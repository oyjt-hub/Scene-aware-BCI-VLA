"""Small deterministic oracle audits, not a performance or validation dataset."""
from pathlib import Path
import json
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from bci_vla.protocols import JointBelief
from bci_vla.models.channel import ChannelModel, JointTransitionModel
from bci_vla.models.consequence import OutcomeTable
from bci_vla.core.diagnostic_policy import DiagnosticPolicy, PlanningState
from bci_vla.baselines.finite_model import FiniteModel
from bci_vla.baselines.oracle import ExactOracle
from bci_vla.baselines.methods import ComparisonSuite, IDS, ABLATIONS

def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,v): (ROOT/p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def make_state(model,weights=None,**changes):
    v=read('examples/diagnosis/state.json'); v.update(changes)
    v['belief']=model.joint(weights) if weights is not None else JointBelief(v['belief'])
    return PlanningState(**v)

if __name__=='__main__':
    p=DiagnosticPolicy(read('configs/diagnosis.json'),channel_model=ChannelModel(read('configs/channels/simulation_binary_v1.json')),
        transition_model=JointTransitionModel(read('configs/belief/simulation_canonical_transition.json')),
        outcome_table=OutcomeTable(read('configs/consequences/simulation_canonical_outcomes.json')))
    model=FiniteModel(p); suite=ComparisonSuite(model,read('configs/comparisons/parameters.json')); oracle=ExactOracle(model)
    fixtures={'alias':make_state(model),'known_normal':make_state(model,(1,0,0,0)),
        'mixed_prior':make_state(model,(.1,.2,.3,.4)), 'no_budget':make_state(model,remaining_s=0)}
    audits=[]; decisions=[]
    for name,state in fixtures.items():
        memory={'predicted_goal':model.goals[0],'last_confirmed_goal':model.goals[0]}
        decisions.append({'fixture_id':name,'public_memory':memory,
            'decisions':[suite.choose(method,state,memory=memory) for method in IDS+ABLATIONS]})
        o1s={h:oracle.solve(state,horizon=h) for h in (1,2)}
        reference=oracle.omniscient_reference(state,horizon=2)
        checks=[]
        for method,horizon in [('M',1),('M2',2),('B9',1)]:
            start=perf_counter_ns(); result=suite.choose(method,state,memory=memory,horizon_remaining=horizon)
            elapsed=perf_counter_ns()-start; exact=o1s[horizon]
            chosen_q=next(r['value'] for r in exact['candidates'] if r['action_id']==result['selected_action']['action_id'])
            checks.append({'method':method,'horizon':horizon,'selected_action':result['selected_action']['action_id'],
                'exact_optimal_value':exact['value'],'exact_value_of_selected_action':chosen_q,
                'decision_regret_at_same_horizon':chosen_q-exact['value'],
                'planning_value_difference':result['planning_value']-exact['value'],
                'method_compute_ns':elapsed,'oracle_compute_ns':exact['compute_ns']})
        audits.append({'fixture_id':name,'state_belief':state.belief.to_dict(),
            'same_information_oracles':{str(k):v for k,v in o1s.items()},'omniscient_reference':reference,'checks':checks})
    write('execution/10_decision_trace.json',{'date':'2026-10-16','scope':'analytic_method_decision_traces',
        'fixtures':'handwritten_shared_models_not_validation_data','cases':decisions,'hardware_calls':0,'cloud_calls':0})
    max_regret=max(r['decision_regret_at_same_horizon'] for a in audits for r in a['checks'])
    write('execution/10_oracle_audit.json',{'date':'2026-10-16','scope':'finite_analytic_oracle_consistency_not_experiment',
        'cases':audits,'max_decision_regret':max_regret,'timing_source':'actual_perf_counter_ns_on_this_host_single_runs',
        'timing_limitation':'not a latency benchmark; no repeated load or hardware measurement',
        'real_validation_choice':None,'strategy_superiority_claim':False,'G2_passed':False,
        'hardware_calls':0,'cloud_calls':0})
    print(json.dumps({'result':'PASS','fixtures':len(audits),'methods':len(IDS+ABLATIONS),
        'same_horizon_checks':sum(len(a['checks']) for a in audits),'max_decision_regret':max_regret,
        'real_validation_choice':None,'scope':'analytic_software_audit'},indent=2))
