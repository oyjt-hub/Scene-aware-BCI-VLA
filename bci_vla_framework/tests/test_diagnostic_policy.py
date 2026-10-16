"""Decision algebra and fail-closed offline planning; no performance evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from bci_vla.protocols import JointBelief, ProtocolError, thaw
from bci_vla.models.channel import ChannelModel, JointTransitionModel
from bci_vla.models.consequence import OutcomeTable
from bci_vla.core.diagnostic_policy import DiagnosticPolicy, PlanningState

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT / p).read_text(encoding='utf-8-sig'))
def policy(config=None, channel=None, transition=None, table=None):
    return DiagnosticPolicy(config or read('configs/diagnosis.json'),
        channel_model=ChannelModel(channel or read('configs/channels/simulation_binary_v1.json')),
        transition_model=JointTransitionModel(transition or read('configs/belief/simulation_canonical_transition.json')),
        outcome_table=OutcomeTable(table or read('configs/consequences/simulation_canonical_outcomes.json')))
def state(**changes):
    v = read('examples/diagnosis/state.json'); v.update(changes)
    if isinstance(v['belief'], dict): v['belief'] = JointBelief(v['belief'])
    return PlanningState(**v)
def fixed(): return read('examples/joint_belief/fixed_state_transition.json')
def candidate(result, ident): return next(x for x in result['candidates'] if x['action_id'] == ident)
def point(g, z):
    v = read('examples/diagnosis/state.json')['belief']
    v['mass'] = [{'goal_id': g, 'mode_id': z, 'probability': 1.0}]
    return v


class DiagnosticPolicyTests(unittest.TestCase):
    def test_alias_queries_have_no_gain_but_probe_can_change_decision(self):
        r = policy().plan(state())
        self.assertEqual(r['selected_action']['kind'], 'CHANNEL_PROBE')
        self.assertGreater(r['value_over_no_diagnosis'], 0)
        self.assertGreater(candidate(r, 'TASK_QUERY:1.0')['expected_loss'], r['baseline_loss'])

    def test_exact_cost_includes_actual_noisy_confirmation_once(self):
        p = policy(transition=fixed()); g = p.goals[0]
        r = p.plan(state(belief=point(g, 'normal')), depth=0)
        confirm = candidate(r, 'confirm:' + g)
        # normal accept .8, reject .15, omission .05; correct outcome .2;
        # setup+window=2s costs .04, original row's 2s allowance is stripped.
        self.assertAlmostEqual(confirm['expected_loss'], .04 + .8 * .2 + .2 * 1)
        self.assertEqual(confirm['elapsed_s'], 2)
        self.assertFalse(r['execution_allowed'])

    def test_wrong_goal_can_be_falsely_confirmed(self):
        p = policy(transition=fixed()); left, right = p.goals
        r = p.plan(state(belief=point(right, 'normal')), depth=0)
        a = candidate(r, 'confirm:' + left)
        # A false accept occurs with probability .15, not zero.
        accept = next(x for x in a['branches'] if x['observation'] == '0')
        self.assertAlmostEqual(accept['probability'], .15)
        self.assertAlmostEqual(accept['continuation_loss'], 2.4)
        self.assertAlmostEqual(a['expected_loss'], .04 + .15 * 2.4 + .85)

    def test_predictive_probabilities_and_posterior_normalized(self):
        r = policy().plan(state(), depth=2)
        for a in r['candidates']:
            if a['branches']:
                self.assertAlmostEqual(sum(x['probability'] for x in a['branches']), 1)
                for branch in a['branches']:
                    self.assertAlmostEqual(sum(branch['posterior']), 1)
                    self.assertEqual(branch['remaining_s'], 20 - a['elapsed_s'])

    def test_setup_transition_and_stable_window_are_separate(self):
        p = policy(); b = p._vector(state().belief)
        a = next(x for x in p.catalog.candidates() if x.action_id == 'probe:0:1.0')
        branches = p._branches(b, a)
        # Bayes averaging recovers T(setup) prior, not T(total elapsed).
        recovered = [sum(x['probability'] * x['posterior'][i] for x in branches) for i in range(4)]
        self.assertTrue(all(abs(x-y) < 1e-12 for x,y in zip(recovered, p._predict(b, 1))))
        self.assertNotAlmostEqual(recovered[0], p._predict(b, 2)[0])

    def test_longer_duration_same_matrix_has_no_free_benefit(self):
        r = policy().plan(state())
        self.assertAlmostEqual(candidate(r, 'probe:0:2.0')['expected_loss'] -
                               candidate(r, 'probe:0:1.0')['expected_loss'], .02)

    def test_one_step_and_bounded_search_share_models_and_base(self):
        p = policy(); a = p.plan(state(), depth=1); b = p.plan(state(), depth=2)
        self.assertEqual(a['baseline_loss'], b['baseline_loss'])
        self.assertLessEqual(b['selected_action']['expected_loss'], a['selected_action']['expected_loss'] + 1e-12)
        # This fixture does not establish an additional benefit of depth two.
        self.assertEqual(a['selected_action']['action_id'], b['selected_action']['action_id'])

    def test_unmodeled_goal_pauses_without_erasing_support(self):
        v = read('examples/diagnosis/state.json')['belief']
        v['mass'] = [{'goal_id': 'unknown', 'mode_id': 'normal', 'probability': 1}]
        r = policy().plan(state(belief=v))
        self.assertEqual(r['reason'], 'unmodeled_support_pause')
        self.assertEqual(r['unresolved_explanations'], v)

    def test_unknown_channel_pauses(self):
        p = policy(); r = p.plan(state(belief=point(p.goals[0], 'unknown_channel')))
        self.assertEqual(r['selected_action']['kind'], 'PAUSE')

    def test_stop_preempts_everything_even_no_budget(self):
        r = policy().plan(state(stopped=True, remaining_s=0))
        self.assertEqual(r['reason'], 'stop_latched')
        self.assertTrue(all(not x['legal'] for x in r['candidates'] if x['kind'] not in {'PAUSE', 'EXIT'}))

    def test_pending_acquisition_cannot_change_menu(self):
        r = policy().plan(state(pending=True))
        self.assertEqual(r['reason'], 'collection_pending_no_menu_change')

    def test_paused_filter_cannot_be_silently_resumed(self):
        self.assertEqual(policy().plan(state(inference_ready=False))['reason'], 'filter_not_ready')

    def test_budget_exhaustion_never_uses_old_goal(self):
        r = policy().plan(state(remaining_s=1))
        self.assertEqual(r['selected_action']['kind'], 'PAUSE')
        self.assertEqual(candidate(r, 'probe:0:1.0')['reason'], 'insufficient_budget')

    def test_scene_deadline_includes_entire_action(self):
        s = state(); r = policy().plan(state(scene_valid_until_s=s.now_s + 1))
        self.assertEqual(candidate(r, 'probe:0:1.0')['reason'], 'scene_expired_during_action')
        self.assertEqual(r['selected_action']['kind'], 'PAUSE')

    def test_contract_expiry_blocks_confirmation(self):
        v = read('examples/diagnosis/state.json')
        for c in v['contracts'].values(): c['valid_until'] = '2026-10-15T00:00:01Z'
        r = policy().plan(state(contracts=v['contracts']))
        for a in r['candidates']:
            if a['kind'] == 'CONFIRM': self.assertEqual(a['reason'], 'contract_expired_during_confirmation')
        self.assertEqual(r['selected_action']['kind'], 'PAUSE')

    def test_exact_deadline_is_expired_as_in_step08(self):
        v = read('examples/diagnosis/state.json')
        for c in v['contracts'].values(): c['valid_until'] = '2026-10-15T00:00:02Z'
        p = policy(); r = p.plan(state(contracts=v['contracts']))
        self.assertEqual(candidate(r, 'confirm:' + p.goals[0])['reason'], 'contract_expired_during_confirmation')
        r = p.plan(state(scene_valid_until_s=state().now_s + 2))
        self.assertEqual(candidate(r, 'probe:0:1.0')['reason'], 'scene_expired_during_action')

    def test_expiry_at_future_branch_is_rechecked(self):
        v = read('examples/diagnosis/state.json')
        for c in v['contracts'].values(): c['valid_until'] = '2026-10-15T00:00:03Z'
        r = policy().plan(state(contracts=v['contracts']))
        a = candidate(r, 'probe:0:1.0')
        self.assertTrue(all(x['next_action'] == 'pause' or x['observation'] == 'NO_RESPONSE' for x in a['branches']))

    def test_noresponse_never_auto_executes_or_retries(self):
        r = policy().plan(state())
        branch = next(x for x in candidate(r, 'probe:0:1.0')['branches'] if x['observation'] == 'NO_RESPONSE')
        self.assertEqual(branch['next_action'], 'PAUSE_NONRESPONSE')

    def test_noncompliance_unknown_is_not_informative_point_model(self):
        c = read('configs/diagnosis.json'); c['actions']['compliance_id'] = 'unknown'
        r = policy(c).plan(state())
        self.assertEqual(candidate(r, 'probe:0:1.0')['reason'], 'unknown_compliance_use_bounds')

    def test_uninformative_probe_is_pruned(self):
        c = read('configs/diagnosis.json'); c['actions']['compliance_id'] = 'uninformative'
        r = policy(c).plan(state())
        self.assertEqual(candidate(r, 'probe:0:1.0')['reason'], 'probe_cannot_distinguish_modes')
        self.assertEqual(r['selected_action']['kind'], 'PAUSE')

    def test_asymmetric_anchor_selects_distinguishing_prompt(self):
        ch = read('configs/channels/simulation_binary_v1.json')
        for z in ch['mode_ids']:
            for matrix in ch['matrices']['CHANNEL_PROBE'][z].values():
                matrix['0'] = {'0': .475, '1': .475, 'NO_RESPONSE': .05}
        c = read('configs/diagnosis.json'); c['actions']['compliance_id'] = 'compliant'
        r = policy(c, channel=ch).plan(state())
        self.assertEqual(candidate(r, 'probe:0:1.0')['reason'], 'probe_cannot_distinguish_modes')
        self.assertEqual(r['selected_action']['prompt'], '1')

    def test_same_pause_for_all_explanations_skips_diagnosis(self):
        r = policy().plan(state(contracts={}))
        self.assertEqual(r['reason'], 'all_supported_explanations_require_pause')
        self.assertEqual(r['selected_action']['kind'], 'PAUSE')

    def test_retry_requires_fresh_window_permission(self):
        r = policy().plan(state())
        self.assertEqual(candidate(r, 'RETRY:1.0')['reason'], 'retry_requires_fresh_disjoint_window')
        self.assertTrue(candidate(policy().plan(state(retry_allowed=True)), 'RETRY:1.0')['legal'])

    def test_extend_requires_unconsumed_final_window(self):
        self.assertTrue(candidate(policy().plan(state(extension_allowed=True)), 'EXTEND:2.0')['legal'])

    def test_recode_and_repair_have_no_invented_success(self):
        r = policy().plan(state())
        for ident in ('recode', 'recalibrate', 'execute'):
            a = candidate(r, ident)
            self.assertFalse(a['legal']); self.assertIsNone(a['expected_loss'])

    def test_unsupported_duration_rejected(self):
        c = read('configs/diagnosis.json'); c['actions']['durations_s'] = [1.5]
        with self.assertRaises(ProtocolError): policy(c)

    def test_mapping_and_profile_must_match(self):
        for key in ('mapping_id', 'channel_profile_id'):
            c = read('configs/diagnosis.json'); c['context'][key] = 'unvalidated'
            with self.assertRaises(ProtocolError): policy(c)

    def test_wrong_contract_revision_or_scene_rejected(self):
        for key in ('revision', 'scene_revision'):
            v = read('examples/diagnosis/state.json'); next(iter(v['contracts'].values()))[key] = 2
            with self.assertRaises(ProtocolError): state(contracts=v['contracts'])

    def test_contract_cannot_silently_change_goal(self):
        v = read('examples/diagnosis/state.json'); items = list(v['contracts'].values())
        items[0]['intent'] = deepcopy(items[1]['intent'])
        with self.assertRaises(ProtocolError): state(contracts=v['contracts'])

    def test_confirmation_phase_accounting_scope_must_be_exclusive(self):
        table = read('configs/consequences/simulation_canonical_outcomes.json')
        table['rows'][0]['included_time_phases'].append('diagnosis')
        with self.assertRaises(ProtocolError): policy(table=table)

    def test_live_mode_and_unbounded_search_rejected(self):
        p = policy()
        with self.assertRaises(ProtocolError): p.plan(state(), execution_mode='live')
        with self.assertRaises(ProtocolError): p.plan(state(), depth=3)

    def test_channel_handoff_does_not_freeze_previous_goal(self):
        p = policy(); left, right = p.goals
        new = p.carry_channel_to_new_task(JointBelief(point(left, 'flip')), {left: 0, right: 1})
        masses = {(r['goal_id'], r['mode_id']): r['probability'] for r in new.data['mass']}
        self.assertEqual(masses[(right, 'flip')], 1)
        self.assertEqual(masses[(left, 'flip')], 0)

    def test_snapshot_time_alignment_and_pause_status(self):
        from tests.test_joint_belief import make_filter
        f, _ = make_filter(); v = read('examples/diagnosis/state.json'); v.pop('belief')
        s = PlanningState.from_filter_snapshot(f.snapshot(), **v)
        self.assertTrue(s.inference_ready)
        v['now_s'] += 1
        with self.assertRaises(ProtocolError): PlanningState.from_filter_snapshot(f.snapshot(), **v)

    def test_trace_is_json_and_input_state_unchanged(self):
        p = policy(); s = state(); before = s.belief.to_dict(); r = p.plan(s, depth=2)
        json.dumps(r, allow_nan=False)
        self.assertEqual(s.belief.to_dict(), before)
        self.assertEqual(s.used_s, 0)
        self.assertEqual(len(r['candidates']), len(p.catalog.candidates()))


if __name__ == '__main__': unittest.main()
