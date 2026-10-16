"""Full non-real-BCI pipeline with independent actual simulator outcomes."""
import argparse,json,time
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from step14C_engine import LiberoEngine
from bci_vla.protocols import thaw,ProtocolError
from bci_vla.scene.state import GroundedScene
from bci_vla.benchmark.offline import stamp
from bci_vla.simulation_v3.factory import OBJECTS,GOALS,empirical_outcomes,make_runtime
from bci_vla.simulation_v3.backend import SimulationVLAAdapter
from bci_vla.simulation_v3.interaction import resolve,dispatch,endpoint
from bci_vla.simulation_v3.replay import replay_session_state
from bci_vla.simulation_v3.recovery import repair_replay_mapping

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--checkpoint-root',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
plan=json.loads((Path(__file__).parent/'14C_run_plan.json').read_text());engine=LiberoEngine(out,a.port,a.checkpoint_root,plan['checkpoint_sha256'])
records=[];episodes=[];started=time.perf_counter()
def save(path,value):(out/path).write_text(json.dumps(value,indent=2)+'\n')
def spec():return dict(scope='LIBERO_simulation',physical_execution_allowed=False,interface_version='libero-model/3.0',
    checkpoint_sha256=engine.checkpoint,instruction_to_object=engine.instruction_to_object,
    image_pipeline='RGB uint8 2views 256->224 rotation180',action_pipeline='franka stats + original LIBERO gripper conversion',
    normalization_sha256=plan['normalization_sha256'],task_registry='pick-cream-cheese-or-wine-v1',predicate_version='sim-held-lift-v1')
def persist_session(s,identifier):
    journal=s.journal.to_jsonl();(out/(identifier+'_journal.jsonl')).write_text(journal)
    with patch.object(SimulationVLAAdapter,'__init__',side_effect=AssertionError('replay no model')),patch.object(SimulationVLAAdapter,'submit',side_effect=AssertionError('replay no dispatch')):
        assert replay_session_state(s.journal)==s.snapshot()

try:
    if plan.get('reuse_calibration_json'):
        raw=Path(plan['reuse_calibration_json']).read_bytes()
        import hashlib
        assert hashlib.sha256(raw).hexdigest()==plan['reused_calibration_sha256']
        records=json.loads(raw);assert len(records)==6
        assert all(r['checkpoint_sha256']==engine.checkpoint for r in records)
        save('independent_trajectories.json',records)
    for split,indices in ([] if records else [('calibration',[0,1]),('calibration_holdout',[2])]):
        for index in indices:
            for target,(obj,label) in enumerate(OBJECTS):
                identifier=split+'-init'+str(index)+'-target'+str(target)
                engine.reset(index,identifier,cap=plan['max_execution_steps'])
                instruction='Pick up the '+label+'.'
                while not engine.finished:engine.advance(instruction,8)
                held=engine.held();details=engine.predicate_details();steps=engine.steps;engine.save_video(identifier)
                reset_time=engine.reset(index,identifier+'-reset')
                records.append(dict(record_id=identifier,split=split,init_state_index=index,target_index=target,held=held,
                    predicate_details=details,actual_policy_steps=steps,simulator_reset_wall_s=reset_time,
                    checkpoint_sha256=engine.checkpoint,source_kind='actual_model_and_simulator',human_time_s=None))
                save('independent_trajectories.json',records)
    outcomes=empirical_outcomes(records,engine.checkpoint);save('calibrated_outcomes.json',outcomes)
    for case in plan['cases']:
        identifier=case['id'];engine.reset(3,identifier,cap=1 if case.get('execution_failure') else plan['max_execution_steps'])
        backend=SimulationVLAAdapter(spec(),engine);s,c=make_runtime(identifier,engine.scene(identifier),backend,outcomes,prior=tuple(case['prior']))
        modes=resolve(c,case['goal'],case['channel']);checks={};all_sessions=[s];dispatched=None;reason=None;dispatches=[];repairs=[]
        try:
            if s.state!='AWAITING_AUTHORIZATION' and case['id'] in {'changed_goal','changed_channel','joint_alias_conflict'}:
                s,c,backend,new_sessions,new_modes,proof=repair_replay_mapping(s,c,engine,spec(),outcomes,identifier,
                    emit_goal=case['goal'],emit_channel=case['channel'])
                all_sessions.extend(new_sessions);modes.extend(new_modes);repairs.append(proof)
                checks['mapping_repair_fresh_confirmation']=s.state=='AWAITING_AUTHORIZATION'
            if case.get('late_confirmation') and s.state=='AWAITING_AUTHORIZATION':
                try:s.authorize(now=stamp(c.now+2.01));checks['late_confirmation_blocked']=False
                except ProtocolError:checks['late_confirmation_blocked']=not backend._submissions
            elif s.state=='AWAITING_AUTHORIZATION':
                dispatch(s,c,engine,identifier+'-command');dispatched=s.submission_contract.data['goal_id']
                dispatches.append(dict(goal=dispatched,intended_goal_on_dispatch=GOALS[case['goal']]))
                if case.get('transport_failure'):engine.fail_next=True
                first=True
                while not engine.finished:
                    backend.advance(8);c.now=round(c.now+.1,6);s.refresh_submission(now=stamp(c.now))
                    if s.state=='SUBMISSION_UNKNOWN':
                        checks['inference_failure_not_BCI_relabel']=s.filter.snapshot()==s.confirmation['snapshot']
                        s.request_stop(now=stamp(c.now+.01));s.refresh_submission(now=stamp(c.now+.02));break
                    if case.get('stop'):
                        applied=engine.steps;s.request_stop(now=stamp(c.now+.01));s.refresh_submission(now=stamp(c.now+.02))
                        checks['STOP_queue_empty']=not engine.queue
                        try:backend.advance();checks['STOP_no_further_steps']=False
                        except ProtocolError:checks['STOP_no_further_steps']=engine.steps==applied
                        break
                    if case.get('goal_change_during_execution') and first:
                        first=False;old=s;old_observation=engine.observation(old.context,stamp(c.now))
                        context=dict(old.context,task_revision=2,scene_revision=2);scene=engine.scene(identifier,revision=2)
                        scene['identity']['session_id']=identifier
                        old.revise_context(context,GroundedScene(scene),now=stamp(c.now+.01));old.refresh_submission(now=stamp(c.now+.02))
                        checks['change_stops_and_drains_old']=old.stop_latched and not engine.queue
                        try:old.dispatch('stale-command',old_observation,now=stamp(c.now+.03));checks['old_authorization_rejected']=False
                        except ProtocolError:checks['old_authorization_rejected']=True
                        # Preserve joint belief; rebuild explicitly for a new version, no env reset.
                        mass={(r['goal_id'],r['mode_id']):r['probability'] for r in old.filter.snapshot()['belief']['mass']}
                        prior=tuple(mass.get((g,z),0) for g in GOALS for z in ['normal','flip'])
                        backend=SimulationVLAAdapter(spec(),engine)
                        s,c=make_runtime(identifier+'-recovery',engine.scene(identifier+'-recovery',2),backend,outcomes,
                            prior=prior,now=c.now+.1,task_revision=2);all_sessions.append(s)
                        modes+=resolve(c,1,'normal')
                        if s.state!='AWAITING_AUTHORIZATION':
                            s,c,backend,new_sessions,new_modes,proof=repair_replay_mapping(s,c,engine,spec(),outcomes,identifier+'-active',
                                emit_goal=1,emit_channel='normal')
                            all_sessions.extend(new_sessions);modes.extend(new_modes);repairs.append(proof)
                        checks['recovery_new_confirmation']=s.state=='AWAITING_AUTHORIZATION'
                        if s.state=='AWAITING_AUTHORIZATION':
                            engine.finished=False;engine.cap=engine.steps+plan['max_execution_steps'];engine.held_streak=0
                            dispatch(s,c,engine,identifier+'-recovery-command');dispatched=s.submission_contract.data['goal_id']
                            dispatches.append(dict(goal=dispatched,intended_goal_on_dispatch=GOALS[1]))
                        else:break
                    if engine.finished:
                        before=s.filter.snapshot();backend.finish();c.now+=.01;s.refresh_submission(now=stamp(c.now));endpoint(s,c,engine,identifier)
                        checks['execution_outcome_does_not_relabel_channel']=before==s.filter.snapshot()
            else:reason='planner_or_noisy_confirmation_paused'
        except Exception as exc:reason=type(exc).__name__+':'+str(exc)
        finally:
            engine.clear_queue();engine.save_video(identifier)
            for j,session in enumerate(all_sessions):persist_session(session,identifier+'-'+str(j))
        row=dict(id=identifier,modes=modes,checks=checks,final_state=s.state,reason=reason,
            dispatched_goal=dispatched,goal_after_change=1 if case.get('goal_change_during_execution') else case['goal'],
            task_success=s.state=='SUCCEEDED',actual_sim_steps=engine.steps,planner_decisions=c.decisions,
            final_filter=s.filter.snapshot(),sessions=len(all_sessions),journals_replayed=True,
            physical_execution_allowed=False,real_BCI_trials=0,dispatches=dispatches,mapping_repair_evidence=repairs)
        row['wrong_target_start']=any(d['goal']!=d['intended_goal_on_dispatch'] for d in dispatches)
        row['faithful_task_success']=row['task_success'] and not row['wrong_target_start']
        episodes.append(row);save('episodes.json',episodes)
    save('summary.json',dict(scope='complete_nonreal_BCI_LIBERO_functional_pipeline',episodes=episodes,
        independent_calibration_records=4,independent_holdout_records=2,calibration_holdout_test_init_indices=[[0,1],[2],[3]],
        all_checks_passed=all(all(e['checks'].values()) for e in episodes),
        model_checkpoint=engine.checkpoint,elapsed_wall_s=time.perf_counter()-started,real_BCI_trials=0,physical_robot_commands=0,
        real_user_time_measured=False,statistical_strategy_superiority_claim=False))
finally:engine.close()
