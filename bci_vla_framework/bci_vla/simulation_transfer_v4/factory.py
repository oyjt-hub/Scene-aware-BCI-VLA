"""Real scene/task bindings with unchanged filter and authorization-aware planner."""
from copy import deepcopy
from ..benchmark.offline import read,BASE,stamp,response
from ..protocols import JointBelief,thaw,timestamp
from ..scene.tasks import goal_id
from ..scene.state import GroundedScene
from ..core.joint_belief import GoalVocabulary,JointBeliefFilter
from ..core.diagnostic_policy import DiagnosticPolicy
from ..models.channel import ChannelModel,JointTransitionModel
from ..runtime_v2.controller import RuntimeController
from ..simulation_v3.outcomes import OutcomeTable
from ..simulation_v3.session import InteractionSession

OBJECTS=(('cream_cheese_1','cream cheese'),('wine_bottle_1','wine bottle'))
INTENTS=[dict(action='pick',source_id=o,destination_id=None) for o,_ in OBJECTS]
GOALS=[goal_id(i) for i in INTENTS]

TRANSPORT='direct'
EXECUTOR_ID='Pi05-LIBERO-v4'

class SimulationController(RuntimeController):
    def roundtrip(self,event):
        from ..adapters.bci.packet_replay import PacketReplayTransport
        codec=PacketReplayTransport(known_target_probe_supported=True)
        raw=codec.encode(event);received=[]
        for offset in range(0,len(raw),17):received.extend(codec.feed(raw[offset:offset+17]))
        assert len(received)==1 and received[0]==event
        self.packet_events=getattr(self,'packet_events',0)+1
        return received[0]

    def begin(self,action):
        from ..protocols import Event
        q,start,end=super().begin(action)
        if self.transport=='packet':q=Event.parse(self.roundtrip(q.to_dict()))
        return q,start,end
    def receive(self,q,symbol,action,start,end):
        self.now=end
        normalized=str(1-int(symbol)) if getattr(self,'input_inversion',False) and symbol in {'0','1'} else symbol
        event=response(q,normalized,start=start,duration=action['duration_s'],index=self.index)
        window=event['payload']['sample_window']
        if window is not None:window['recording_id']='synthetic-response-stream:'+self.session.context['session_id']
        if self.transport=='packet':event=self.roundtrip(event)
        result=self.session.ingest(event,now=stamp(end),monotonic_now=end)
        assert result.accepted,result.reason
        if action['kind']!='CONFIRM':
            self.diagnostics+=int(self.index>1);self.memory['diagnostic_count']=self.diagnostics
        if self.session.state=='PAUSED':self.termination='response_or_inference_paused'
        return self.session.state

def remap(value,mapping):
    if isinstance(value,dict):return {mapping.get(k,k):remap(v,mapping) for k,v in value.items()}
    if isinstance(value,list):return [remap(v,mapping) for v in value]
    if isinstance(value,str):
        for old,new in mapping.items():value=value.replace(old,new)
    return value

def empirical_outcomes(records,checkpoint):
    rows=[]
    for target,decision in enumerate(['KEEP_OLD','ACCEPT_NEW']):
        samples=[r for r in records if r['split']=='calibration' and r['target_index']==target]
        assert samples
        for g,gid in enumerate(GOALS):
            failures=sum(not r['held'][OBJECTS[g][0]] for r in samples)
            # Predeclared Beta(1,1) posterior mean, not a fitted success guarantee.
            probability=(1+failures)/(2+len(samples))
            for mode in ['normal','flip']:
                rows.append(dict(decision=decision,stage='terminal',goal_id=gid,channel_mode_id=mode,
                    components=dict(wrong_start=float(g!=target),task_failure=probability,
                        recovery_s=sum(r['simulator_reset_wall_s'] for r in samples)/len(samples),user_time_s=2.),
                    source_ref='independent-calibration:'+','.join(r['record_id'] for r in samples),included_time_phases=['confirmation']))
    return dict(schema_version='outcome-table/1.0',model_id='libero-pick-calibration-v3',executor_id=EXECUTOR_ID,
        checkpoint_ref=checkpoint,predicate_version='sim-held-lift-v1',status='simulation_only',provenance='empirical_simulator_calibration',
        deployment_allowed=False,goal_ids=GOALS,channel_mode_ids=['normal','flip'],rows=rows)

def make_runtime(identifier,scene,backend,outcomes,*,prior=(.45,.05,.45,.05),now=0,task_revision=1,input_inversion=False,profile_revision=0):
    old=read('configs/belief/simulation_binary_filter.json')
    mapping=dict(zip([g['goal_id'] for g in old['vocabulary']['goals'] if g['kind']=='task'],GOALS))
    f=remap(old,mapping)
    context=dict(f['current_context'],session_id=identifier,scene_id=scene['scene_id'],scene_revision=scene['revision'],task_id='libero-pick-task',task_revision=task_revision)
    f['vocabulary'].update(scene_id=context['scene_id'],scene_revision=context['scene_revision'])
    for g,intent in zip(f['vocabulary']['goals'][:2],INTENTS):g['intent']=intent
    initial=deepcopy(f['initial_joint']);initial['as_of_event_id']='declared-simulation-prior'
    initial['mass']=[dict(goal_id=g,mode_id=z,probability=p) for (g,z),p in zip(((g,z) for g in GOALS for z in ['normal','flip']),prior)]
    channel_config=read('configs/channels/simulation_binary_v1.json')
    if profile_revision:
        mapping_id='binary-repaired-'+str(profile_revision)
        channel_config['profile_id']='simulation-binary-repaired-'+str(profile_revision)
        channel_config['scope']['mapping_id']=mapping_id
        context.update(mapping_id=mapping_id,channel_profile_id=channel_config['profile_id'])
    channel=ChannelModel(channel_config)
    transition=JointTransitionModel(remap(read('configs/belief/simulation_canonical_transition.json'),mapping))
    filt=JointBeliefFilter(JointBelief(initial),vocabulary=GoalVocabulary(f['vocabulary']),channel_model=channel,transition_model=transition,
        current_context=context,layout_id=f['layout_id'],initial_at=timestamp(stamp(now)))
    pc=remap(read('configs/diagnosis.json'),mapping);pc['context']=context;pc['actions']['scope']['mapping_id']=context['mapping_id']
    policy=DiagnosticPolicy(pc,channel_model=channel,transition_model=transition,outcome_table=OutcomeTable(outcomes))
    config=remap(read('configs/execution/session.json'),mapping);config.update(scope='simulation_model_only',backend_id=outcomes['executor_id'],
        predicate_version='sim-held-lift-v1',simulation_supported_actions=['pick'],endpoint_hosts=['simulator-state'])
    config['task_binding']['question']='Pick up the cream cheese (0) or wine bottle (1)?'
    config['task_binding']['mapping_id']=context['mapping_id']
    scene=deepcopy(scene);scene['identity']['session_id']=identifier
    session=InteractionSession(current_context=context,scene=GroundedScene(scene),belief_filter=filt,policy=policy,backend=backend,config=config,started_at=stamp(now))
    controller=SimulationController(session,'M',budget_s=45,diagnostic_limit=2)
    controller.contracts={g:dict(contract_id='contract-'+g,task_id=context['task_id'],revision=task_revision,
        scene_id=context['scene_id'],scene_revision=context['scene_revision'],intent=i,status='proposed',valid_until=stamp(60),confirmation_event_id=None) for g,i in zip(GOALS,INTENTS)}
    controller.now=now
    controller.transport=TRANSPORT
    controller.input_inversion=input_inversion
    return session,controller
