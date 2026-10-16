"""Deterministic hypothetical planning traces; not a sampled method comparison."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from bci_vla.protocols import JointBelief
from bci_vla.models.channel import ChannelModel, JointTransitionModel
from bci_vla.models.consequence import OutcomeTable
from bci_vla.core.diagnostic_policy import DiagnosticPolicy, PlanningState

def read(p): return json.loads((ROOT / p).read_text(encoding='utf-8-sig'))
def make_state(**changes):
    v = read('examples/diagnosis/state.json'); v.update(changes)
    if isinstance(v['belief'], dict): v['belief'] = JointBelief(v['belief'])
    return PlanningState(**v)

if __name__ == '__main__':
    p = DiagnosticPolicy(read('configs/diagnosis.json'),
        channel_model=ChannelModel(read('configs/channels/simulation_binary_v1.json')),
        transition_model=JointTransitionModel(read('configs/belief/simulation_canonical_transition.json')),
        outcome_table=OutcomeTable(read('configs/consequences/simulation_canonical_outcomes.json')))
    one = p.plan(make_state(), depth=1); two = p.plan(make_state(), depth=2)
    branch = next(b for b in one['selected_action']['branches'] if b['observation'] == '0')
    posterior = JointBelief({'goal_ids': list(p.goals), 'mode_ids': list(p.modes),
        'mass': [{'goal_id': g, 'mode_id': z, 'probability': v}
                 for (g, z), v in zip(p.states, branch['posterior'])],
        'as_of_event_id': 'hypothetical-probe-zero-not-actual-observation'})
    next_prior = p.carry_channel_to_new_task(posterior, {p.goals[0]: 1, p.goals[1]: 0})
    output = {'date': '2026-10-16', 'scope': 'analytic_candidate_plans_not_performance_experiment',
        'models': 'illustrative_not_fitted', 'one_step': one, 'two_step': two,
        'no_budget': p.plan(make_state(remaining_s=0)),
        'all_explanations_pause': p.plan(make_state(contracts={})),
        'stop': p.plan(make_state(stopped=True)),
        'channel_handoff': {'conditional_branch': 'hypothetical probe 0 observation 0',
            'posterior': posterior.to_dict(), 'new_goal_prior': {p.goals[0]: 1, p.goals[1]: 0},
            'next_joint_prior': next_prior.to_dict(), 'new_contracts_required': True},
        'claim': 'This fixture supplies no evidence that depth two outperforms one-step VoI.',
        'hardware_calls': 0, 'cloud_calls': 0}
    (ROOT / 'execution/09_policy_trace.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'result': 'PASS', 'one_step_action': one['selected_action']['action_id'],
        'one_step_loss': one['selected_action']['expected_loss'],
        'two_step_action': two['selected_action']['action_id'],
        'two_step_loss': two['selected_action']['expected_loss'],
        'scope': output['scope']}, indent=2))
