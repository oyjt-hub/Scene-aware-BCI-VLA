"""Two fixed analytic evidence pairs, not an experimental run or policy demo."""
from pathlib import Path
import json
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from bci_vla.protocols import JointBelief, timestamp, thaw
from bci_vla.adapters.bci.replay import ReplayBCIAdapter
from bci_vla.models.channel import ChannelModel, JointTransitionModel
from bci_vla.core.joint_belief import GoalVocabulary,TaskQuestionBinding,JointBeliefFilter

def load(p):return json.loads((root/p).read_text(encoding='utf-8-sig'))
c=load('configs/belief/simulation_binary_filter.json');v=GoalVocabulary(c['vocabulary'])
f=JointBeliefFilter(JointBelief(c['initial_joint']),vocabulary=v,channel_model=ChannelModel.load(root/'configs/channels/simulation_binary_v1.json'),transition_model=JointTransitionModel(load('examples/joint_belief/fixed_state_transition.json')),current_context=c['current_context'],layout_id=c['layout_id'],initial_at=timestamp(c['initial_at']))
binding=TaskQuestionBinding(c['task_binding'],v)
a=ReplayBCIAdapter(c['current_context'],response_hosts={'bci'},control_hosts={'ui'})
initial=f.snapshot()
for pair in load('examples/joint_belief/alias_events.json')['events']:
    q,r=pair['request'],pair['response'];qtime=timestamp(q['sent_at']);rtime=timestamp(r['sent_at'])
    a.register_request(q,now=qtime,monotonic_now=(qtime-timestamp(c['initial_at'])).total_seconds())
    accepted=a.ingest(r,now=rtime,monotonic_now=(rtime-timestamp(c['initial_at'])).total_seconds())
    out=f.update(accepted,binding=binding if q['mode']=='TASK_QUERY' else None,compliance_id='compliant' if q['mode']=='CHANNEL_PROBE' else None,assumption='state_constant_within_window',finalized=True)
    if not out['updated']:raise AssertionError(out['reason'])
trace={'scope':'analytic_fixed_evidence_replay','provenance':'not_an_experiment_not_EEG_performance','initial':initial,'audit':thaw(f.audit_log),'final':f.snapshot()}
(root/'execution/08_alias_trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'result':'PASS','updates':len(f.audit_log),'first_task_goal_marginals':dict(f.audit_log[0]['after']['goal_marginals']),'after_probe_goal_marginals':f.snapshot()['goal_marginals'],'execution_allowed':False},ensure_ascii=True,indent=2))
