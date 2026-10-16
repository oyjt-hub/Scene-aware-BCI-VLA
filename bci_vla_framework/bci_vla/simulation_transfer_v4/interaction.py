"""Controller chooses diagnostics; only bound actual synthetic observations enter."""
from ..benchmark.offline import stamp,response
from ..protocols import thaw,require
from .factory import GOALS,INTENTS
from ..simulation_v3.records import EndpointEvidence

def interact(controller,action,*,private_goal_index,private_channel_mode):
    q,start,end=controller.begin(action);mode=q.data['mode']
    intended=(action['prompt'] if mode=='CHANNEL_PROBE' else str(private_goal_index) if mode=='TASK_QUERY' else
        '0' if action['goal_id']==GOALS[private_goal_index] else '1')
    observed=str(1-int(intended)) if private_channel_mode=='flip' else intended
    controller.receive(q,observed,action,start,end)
    return {'mode':mode,'request_id':q.data['event_id'],'observed_symbol':observed,'state_after':controller.session.state}

def resolve(controller,goal,channel,*,initial=True):
    modes=[]
    if initial:modes.append(interact(controller,dict(kind='TASK_QUERY',setup_s=1.,duration_s=1.,elapsed_s=2.),
        private_goal_index=goal,private_channel_mode=channel))
    for _ in range(8):
        action=controller.choose()
        if action is None:break
        modes.append(interact(controller,action,private_goal_index=goal,private_channel_mode=channel))
        if action['kind']=='CONFIRM':break
    return modes

def dispatch(session,controller,engine,submission_id):
    controller.now=round(controller.now+.01,6)
    session.authorize(now=stamp(controller.now))
    session.dispatch(submission_id,engine.observation(session.context,stamp(controller.now)),now=stamp(controller.now))
    return session.state

def endpoint(session,controller,engine,identifier):
    controller.now+=.01;observed=stamp(controller.now);controller.now+=.01
    contract=session.submission_contract.data;proposal=contract['proposal']
    trajectory={'trajectory_id':identifier+'-trajectory','executor_id':session.config['backend_id'],
        'checkpoint_ref':engine.checkpoint,'predicate_version':'sim-held-lift-v1','provenance':'independent_annotation',
        'split':'test','source_ref':identifier+'-simulator-held-and-lift-state','command_dispatched':True,
        'commanded_intent':thaw(proposal['intent']),
        'final_predicates':[dict(name='holding',subject_id=obj,object_id=None,value=value,evidence_ref=identifier+'-contact-and-height') for obj,value in engine.held().items()],
        'user_time_s':None,'recovery_by_goal':{}}
    evidence=EndpointEvidence(dict(evidence_id=identifier+'-endpoint',session_id=session.context['session_id'],
        submission_id=session.submission['submission_id'],contract_id=proposal['contract_id'],contract_revision=proposal['revision'],
        scene_id=proposal['scene_id'],scene_revision=proposal['scene_revision'],
        observed_at=dict(host_id='simulator-state',at=observed,offset_ms=0,uncertainty_ms=0),
        source='simulator_state_annotation',trajectory=trajectory))
    return session.verify_endpoint(evidence,now=stamp(controller.now))
