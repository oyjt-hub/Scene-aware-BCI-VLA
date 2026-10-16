"""Exact alias-world, correlated-posterior and failure-transaction regressions."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import unittest

from bci_vla.protocols import JointBelief, ProtocolError
from bci_vla.adapters.bci.replay import ReplayBCIAdapter, IngestResult
from bci_vla.models.channel import ChannelModel, JointTransitionModel
from bci_vla.core.joint_belief import GoalVocabulary, TaskQuestionBinding, JointBeliefFilter, UNKNOWN_CHANNEL

ROOT=Path(__file__).resolve().parents[1]
BASE=datetime(2026,10,15,tzinfo=timezone.utc)
def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def config():return load('configs/belief/simulation_binary_filter.json')
def stamp(seconds):return (BASE+timedelta(seconds=seconds)).isoformat()


def make_filter(*,initial=None,channel=None,dynamic=False,outlier=0):
    c=config();v=GoalVocabulary(c['vocabulary'])
    t=load('configs/belief/simulation_canonical_transition.json' if dynamic else 'examples/joint_belief/fixed_state_transition.json')
    f=JointBeliefFilter(JointBelief(initial or c['initial_joint']),vocabulary=v,channel_model=ChannelModel(channel or load('configs/channels/simulation_binary_v1.json')),transition_model=JointTransitionModel(t),current_context=c['current_context'],layout_id=c['layout_id'],initial_at=BASE,outlier_weight=outlier)
    return f,TaskQuestionBinding(c['task_binding'],v)


def result(mode='TASK_QUERY',*,index=0,symbol='1',status='OBSERVATION',duration=1,mutate=None):
    c=config();name=mode.lower();q=load('examples/protocols/'+name+'_request.json');r=load('examples/protocols/'+name+'_response.json')
    t=index*4
    q.update(c['current_context']);q.update(event_id='request-'+str(index),sequence=index,interaction_id='interaction-'+str(index),query_id=None if mode=='CHANNEL_PROBE' else 'query-'+str(index),probe_id='probe-'+str(index) if mode=='CHANNEL_PROBE' else None,sent_at=stamp(t),expires_at=stamp(t+12))
    if mode=='TASK_QUERY':q['payload'].update(question=c['task_binding']['question'],options=deepcopy(c['task_binding']['options']))
    if mode=='CHANNEL_PROBE':q['payload']['probe_prompt']['presented_at']['at']=stamp(t)
    if mode=='CONFIRM':
        q['payload']['contract'].update(intent=c['vocabulary']['goals'][0]['intent'],valid_until=stamp(t+11))
    for k in ['session_id','task_id','task_revision','scene_id','scene_revision','mapping_id','stimulus_version','channel_profile_id','interaction_id','query_id','probe_id']:
        r[k]=q[k]
    r.update(event_id='response-'+str(index),sequence=index,response_to=q['event_id'],sent_at=stamp(t+duration+2),expires_at=stamp(t+12))
    r['payload'].update(status=status,symbol=symbol if status=='OBSERVATION' else None)
    if status=='OBSERVATION':
        r['payload']['sample_window'].update(window_id='window-'+str(index),sample_start=index*1000+250,sample_end=index*1000+250+duration*250,started_at=stamp(t+1),ended_at=stamp(t+1+duration))
    else:r['payload'].update(raw_scores=None,score_kind=None,calibration_id=None,sample_window=None)
    if mutate:mutate(q,r)
    a=ReplayBCIAdapter(c['current_context'],response_hosts={'bci'},control_hosts={'ui'})
    a.register_request(q,now=BASE+timedelta(seconds=t),monotonic_now=t)
    accepted=a.ingest(r,now=BASE+timedelta(seconds=t+duration+2.01),monotonic_now=t+duration+2.01)
    assert accepted.accepted,accepted.reason
    return accepted


def update(f,r,*,binding=None,**kwargs):
    return f.update(r,binding=binding,assumption='state_constant_within_window',finalized=True,**kwargs)


def masses(f):
    return {(r['goal_id'],r['mode_id']):r['probability'] for r in f.snapshot()['belief']['mass']}


class JointUpdateTests(unittest.TestCase):
    def test_task_alias_stays_correlated_and_equal_after_repeated_queries(self):
        f,b=make_filter();left,right=f.vocabulary.known_goal_ids
        for i in range(3):self.assertTrue(update(f,result(index=i),binding=b)['updated'])
        p=masses(f)
        self.assertAlmostEqual(p[(right,'normal')],0.5);self.assertAlmostEqual(p[(left,'flip')],0.5)
        self.assertEqual(p[(left,'normal')],0);self.assertEqual(p[(right,'flip')],0)
        self.assertAlmostEqual(f.snapshot()['goal_marginals'][left],0.5)

    def test_anchor_changes_goal_marginal_without_assigning_prompt_as_goal(self):
        f,b=make_filter();left,right=f.vocabulary.known_goal_ids
        self.assertTrue(update(f,result(),binding=b)['updated'])
        r=result('CHANNEL_PROBE',index=1,symbol='0');before=r.request.to_dict()
        self.assertTrue(update(f,r,compliance_id='compliant')['updated'])
        self.assertAlmostEqual(f.snapshot()['goal_marginals'][right],float(Fraction(18,19)))
        self.assertEqual(r.request.to_dict(),before)
        self.assertEqual(f.audit_log[-1]['probe_prompt_symbol'],'0')
        self.assertFalse(f.snapshot()['execution_allowed'])

    def test_two_anchors_accumulate_once_and_opposite_evidence_cancels(self):
        f,_=make_filter();right=f.vocabulary.known_goal_ids[1]
        for i in range(2):self.assertTrue(update(f,result('CHANNEL_PROBE',index=i,symbol='0'),compliance_id='compliant')['updated'])
        self.assertAlmostEqual(f.snapshot()['goal_marginals'][right],float(Fraction(324,325)))
        f,_=make_filter()
        update(f,result('CHANNEL_PROBE',symbol='0'),compliance_id='compliant')
        update(f,result('CHANNEL_PROBE',index=1,symbol='1'),compliance_id='compliant')
        self.assertAlmostEqual(f.snapshot()['goal_marginals'][right],0.5)

    def test_uninformative_probe_and_no_response_do_not_invent_diagnosis(self):
        f,_=make_filter();before=masses(f)
        self.assertTrue(update(f,result('CHANNEL_PROBE',symbol='0'),compliance_id='uninformative')['updated'])
        for key,value in before.items():self.assertAlmostEqual(masses(f)[key],value)
        f,b=make_filter();r=result(status='NO_RESPONSE')
        outcome=update(f,r,binding=b,nonresponse_window={'start':stamp(1),'end':stamp(2),'clock_domain':'reference'})
        self.assertTrue(outcome['updated'],outcome['reason'])
        self.assertEqual(f.audit_log[-1]['signal_status'],'NO_RESPONSE')
        for key,value in before.items():self.assertAlmostEqual(masses(f)[key],value)

    def test_noisy_confirmation_is_not_a_true_label_or_authorization(self):
        c=config();left,right=[g['goal_id'] for g in c['vocabulary']['goals'][:2]]
        initial=deepcopy(c['initial_joint']);initial['mass']=[{'goal_id':left,'mode_id':'normal','probability':0.5},{'goal_id':right,'mode_id':'normal','probability':0.5}]
        f,_=make_filter(initial=initial);r=result('CONFIRM',symbol='1')
        self.assertTrue(update(f,r)['updated'])
        self.assertAlmostEqual(f.snapshot()['goal_marginals'][left],16/19)
        self.assertLess(f.snapshot()['goal_marginals'][left],1)
        self.assertEqual(r.request.data['payload']['contract']['status'],'proposed')
        self.assertFalse(f.snapshot()['execution_allowed'])

    def test_elapsed_gap_predicts_joint_changes_without_factorization(self):
        f,b=make_filter(dynamic=True);prior=masses(f);model=f.transition_model
        r=result(index=1,status='NO_RESPONSE')
        out=update(f,r,binding=b,nonresponse_window={'start':stamp(5),'end':stamp(6),'clock_domain':'reference'})
        self.assertTrue(out['updated'],out['reason'])
        states=[(g,z) for g in f.vocabulary.known_goal_ids for z in ['normal','flip']];t=model.matrix(5)
        for j,state in enumerate(states):self.assertAlmostEqual(masses(f)[state],sum(prior[old]*t[i][j] for i,old in enumerate(states)))
        self.assertEqual(f.audit_log[-1]['gap_transition_s'],5)
        self.assertTrue(all(masses(f)[s]>0 for s in states))

    def test_longer_supported_window_is_logged_not_a_fixed_step_transition(self):
        f,b=make_filter();out=update(f,result(duration=2),binding=b)
        self.assertTrue(out['updated']);self.assertEqual(f.audit_log[-1]['stable_window_s'],2)
        self.assertEqual(f.snapshot()['as_of_reference_time'],stamp(3))

    def test_random_error_keeps_multiple_channel_hypotheses(self):
        f,b=make_filter();out=update(f,result(symbol='0'),binding=b)
        self.assertTrue(out['updated']);p=f.snapshot()['channel_marginals']
        self.assertGreater(p['normal'],0);self.assertGreater(p['flip'],0)
        self.assertNotIn('diagnosed_cause',f.snapshot())


class FailureTests(unittest.TestCase):
    def assert_paused_unchanged(self,f,before,out,reason):
        self.assertFalse(out['updated']);self.assertEqual(out['reason'],reason)
        self.assertEqual(masses(f),before);self.assertEqual(f.snapshot()['status'],'paused')

    def test_unknown_compliance_pauses_not_replaced_with_perfect_compliance(self):
        f,_=make_filter();before=masses(f)
        self.assert_paused_unchanged(f,before,update(f,result('CHANNEL_PROBE'),compliance_id='unknown'),'unknown_compliance_use_bounds')

    def test_positive_unknown_none_and_channel_unknown_mass_is_preserved(self):
        for goal,mode in [('unknown','normal'),('none','normal'),(config()['initial_joint']['goal_ids'][0],UNKNOWN_CHANNEL)]:
            c=config();initial=deepcopy(c['initial_joint']);initial['mass'][0]['probability']=0.4;initial['mass'].append({'goal_id':goal,'mode_id':mode,'probability':0.1})
            f,b=make_filter(initial=initial);before=masses(f)
            self.assert_paused_unchanged(f,before,update(f,result(),binding=b),'unmodeled_state_support_requires_pause')

    def test_zero_likelihood_enters_pause_without_argmax_or_uniform_reset(self):
        c=load('configs/channels/simulation_binary_v1.json')
        for z in c['mode_ids']:
            for d,rows in c['matrices']['TASK_QUERY'][z].items():
                for u in rows:rows[u]={'0':1.0,'1':0.0,'NO_RESPONSE':0.0}
        f,b=make_filter(channel=c);before=masses(f)
        self.assert_paused_unchanged(f,before,update(f,result(symbol='1'),binding=b),'zero_evidence_probability')

    def test_numerical_underflow_is_not_silent_hypothesis_deletion(self):
        c=config();left,right=c['initial_joint']['goal_ids'][:2]
        initial=deepcopy(c['initial_joint']);initial['mass']=[{'goal_id':left,'mode_id':'normal','probability':5e-324},{'goal_id':right,'mode_id':'normal','probability':1.0}]
        f,b=make_filter(initial=initial);before=masses(f)
        self.assert_paused_unchanged(f,before,update(f,result(symbol='1'),binding=b),'posterior_underflow_or_nonfinite')

    def test_duplicate_evidence_does_not_update_belief_twice(self):
        f,b=make_filter();r=result();self.assertTrue(update(f,r,binding=b)['updated']);before=masses(f)
        self.assert_paused_unchanged(f,before,update(f,r,binding=b),'duplicate_belief_evidence')

    def test_overlap_failure_does_not_commit_prediction_or_posterior(self):
        f,b=make_filter(dynamic=True);self.assertTrue(update(f,result(),binding=b)['updated']);before=masses(f)
        def overlap(q,r):r['payload']['sample_window'].update(sample_start=375,sample_end=625)
        self.assert_paused_unchanged(f,before,update(f,result(index=1,mutate=overlap),binding=b),'overlapping_prefix_evidence')

    def test_controls_pause_or_stop_without_changing_goal(self):
        for mode in ['STOP','CANCEL']:
            f,_=make_filter();before=masses(f);c=config()
            a=ReplayBCIAdapter(c['current_context'],response_hosts={'bci'},control_hosts={'ui'})
            q=load('examples/protocols/task_query_request.json');q.update(c['current_context'])
            a.register_request(q,now=BASE,monotonic_now=0)
            control=load('examples/protocols/'+mode.lower()+'.json');control.update(c['current_context'])
            control['interaction_id']=q['interaction_id']
            accepted=a.ingest(control,now=BASE+timedelta(seconds=3),monotonic_now=3)
            self.assertTrue(accepted.accepted)
            out=f.update(accepted)
            self.assertFalse(out['updated']);self.assertEqual(masses(f),before)
            self.assertEqual(f.snapshot()['status'],'stopped' if mode=='STOP' else 'paused')
            if mode=='CANCEL':
                stop=load('examples/protocols/stop.json');stop.update(c['current_context'])
                accepted_stop=a.ingest(stop,now=BASE+timedelta(seconds=3),monotonic_now=3)
                self.assertEqual(f.update(accepted_stop)['reason'],'stop')
                self.assertEqual(f.snapshot()['status'],'stopped')

    def test_unknown_signal_and_score_only_pause_without_pseudo_labels(self):
        f,b=make_filter();before=masses(f)
        self.assert_paused_unchanged(f,before,update(f,result(status='INVALID_SIGNAL'),binding=b),'status_requires_pause')
        f,b=make_filter();r=result(mutate=lambda q,r:r['payload'].update(symbol=None));before=masses(f)
        self.assert_paused_unchanged(f,before,update(f,r,binding=b),'score_likelihood_not_calibrated')

    def test_unregistered_question_or_private_alias_does_not_update(self):
        f,b=make_filter();before=masses(f)
        self.assert_paused_unchanged(f,before,update(f,result()),'explicit_task_binding_required')
        f,b=make_filter();r=result(mutate=lambda q,r:q['payload'].update(question='Different semantic question'))
        self.assert_paused_unchanged(f,masses(f),update(f,r,binding=b),'question_text_mismatch')
        v=config()['vocabulary'];v['goals'][0]['goal_id']='old'
        with self.assertRaisesRegex(ProtocolError,'goal_alias_not_canonical'):GoalVocabulary(v)

    def test_transition_aliases_must_be_explicitly_remapped(self):
        c=config();v=GoalVocabulary(c['vocabulary'])
        with self.assertRaisesRegex(ProtocolError,'transition_goal_ids_mismatch'):
            JointBeliefFilter(JointBelief(c['initial_joint']),vocabulary=v,channel_model=ChannelModel(load('configs/channels/simulation_binary_v1.json')),transition_model=JointTransitionModel(load('configs/channels/simulation_transition_v1.json')),current_context=c['current_context'],layout_id=c['layout_id'],initial_at=BASE)

    def test_question_from_another_valid_vocabulary_is_rejected(self):
        f,_=make_filter();c=config();v=deepcopy(c['vocabulary'])
        from bci_vla.scene.tasks import goal_id
        for g in v['goals']:
            if g['kind']=='task':g['intent']['source_id']='different-source';g['goal_id']=goal_id(g['intent'])
        other=GoalVocabulary(v);binding=deepcopy(c['task_binding'])
        for option,g in zip(binding['options'],other.known_goal_ids):option['semantic_ref']='goal:'+g
        b=TaskQuestionBinding(binding,other)
        r=result(mutate=lambda q,r:q['payload'].update(options=binding['options']))
        self.assert_paused_unchanged(f,masses(f),update(f,r,binding=b),'question_goal_mismatch')

    def test_outside_scope_and_unfinalized_prefix_pause(self):
        f,b=make_filter();f.layout_id='different-layout'
        self.assert_paused_unchanged(f,masses(f),update(f,result(),binding=b),'outside_calibration_scope')
        f,b=make_filter();out=f.update(result(),binding=b,assumption='state_constant_within_window',finalized=False)
        self.assert_paused_unchanged(f,masses(f),out,'intermediate_prefix_not_evidence')
        f,b=make_filter();r=result(mutate=lambda q,r:r['payload']['sample_window'].update(source_kind='live_bci'))
        self.assert_paused_unchanged(f,masses(f),update(f,r,binding=b),'offline_evidence_required')

    def test_stale_task_versions_and_reversed_time_cannot_infer(self):
        f,b=make_filter();f._context=dict(f._context,task_revision=2)
        self.assert_paused_unchanged(f,masses(f),update(f,result(),binding=b),'stale_belief_context')
        f,b=make_filter();update(f,result(index=1),binding=b);before=masses(f)
        self.assert_paused_unchanged(f,before,update(f,result(index=0),binding=b),'out_of_order_belief_evidence')

    def test_confirmation_cannot_bind_outside_task_vocabulary(self):
        f,_=make_filter();r=result('CONFIRM',mutate=lambda q,r:q['payload']['contract']['intent'].update(source_id='other-object'))
        self.assert_paused_unchanged(f,masses(f),update(f,r),'contract_outside_goal_vocabulary')

    def test_unmodeled_clock_uncertainty_pauses(self):
        f,b=make_filter();r=result(mutate=lambda q,r:r.update(clock_uncertainty_ms=1))
        self.assert_paused_unchanged(f,masses(f),update(f,r,binding=b),'filter_requires_exact_reference_window')

    def test_paused_filter_does_not_auto_resume_or_accept_new_labels(self):
        f,b=make_filter();update(f,result(status='NO_CONTROL'),binding=b);before=masses(f)
        out=update(f,result(index=1),binding=b)
        self.assertFalse(out['updated']);self.assertEqual(out['reason'],'filter_paused');self.assertEqual(masses(f),before)


class AuditAndImportTests(unittest.TestCase):
    def test_audit_is_immutable_and_snapshot_is_detached(self):
        f,b=make_filter();update(f,result(),binding=b)
        audit=f.audit_log[-1]
        self.assertEqual(audit['mode'],'TASK_QUERY');self.assertTrue(audit['updated'])
        self.assertIn('before',audit);self.assertIn('prediction',audit);self.assertIn('after',audit)
        with self.assertRaises(TypeError):audit['reason']='changed'
        snapshot=f.snapshot();snapshot['belief']['mass'].clear()
        self.assertTrue(f.snapshot()['belief']['mass'])

    def test_joint_filter_import_does_not_touch_model_or_device_runtime(self):
        code='''
import builtins,socket
old=builtins.__import__
def guard(name,*a,**k):
 if name.split('.')[0] in {'torch','jax','numpy','mne','pygame','requests','paramiko','cv2'}:raise AssertionError('external runtime import')
 return old(name,*a,**k)
builtins.__import__=guard
socket.socket=lambda *a,**k:(_ for _ in ()).throw(AssertionError('network forbidden'))
import bci_vla.core.joint_belief
'''
        r=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
