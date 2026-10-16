"""Independent finite-oracle algebra, fair scoring, variants and selection gates."""
from copy import deepcopy
from unittest.mock import patch
import json
import unittest

from bci_vla.protocols import JointBelief, ProtocolError
from bci_vla.baselines.finite_model import FiniteModel, entropy
from bci_vla.baselines.oracle import ExactOracle
from bci_vla.baselines.methods import ComparisonSuite, IDS, ABLATIONS
from bci_vla.baselines.selection import select_validation, digest
from tests.test_diagnostic_policy import policy, state, read, fixed, point

def suite(p=None): return ComparisonSuite(FiniteModel(p or policy()),read('configs/comparisons/parameters.json'))
def memory(m): return {'predicted_goal':m.goals[0],'last_confirmed_goal':m.goals[0]}
def selection_fixture():
    partitions={'development':['dev'],'calibration':['cal'],'validation':['val'],'test':['test']}
    rule={'status':'predeclared','source_split':'calibration','source_ref':'unit-test-only-pilot',
          'wrong_start_rate_ceiling':.2,'all_episode_mean_time_ceiling_s':10}
    records=[]
    for method,n in [('B5',6),('B9',8)]:
        records.append({'candidate_id':method+'-fixture','method':method,'config_sha256':digest({'method':method}),
            'split':'validation','provenance':'simulation_fixture','group_ids':['val'],'episode_ids':['e'+str(i) for i in range(10)],
            'episodes':10,'faithful_completions':n,'wrong_starts':1,'abstentions':2,'total_all_episode_time_s':60,
            'common_protocol_checks_pass':True})
    return records,partitions,rule


class OracleTests(unittest.TestCase):
    def test_independent_from_step09_solver(self):
        p=policy(); m=FiniteModel(p)
        with patch.object(p,'plan',side_effect=AssertionError('must not delegate')), \
             patch.object(p,'_evaluate',side_effect=AssertionError('must not delegate')), \
             patch.object(p,'_branches',side_effect=AssertionError('must not delegate')):
            r=ExactOracle(m).solve(state(),horizon=2)
        self.assertEqual(r['selected_action']['kind'],'CHANNEL_PROBE')

    def test_closed_form_confirmation_with_errors_and_once_only_time(self):
        p=policy(transition=fixed()); m=FiniteModel(p); g=m.goals[0]
        r=ExactOracle(m).solve(state(belief=point(g,'normal')),horizon=0)
        self.assertAlmostEqual(r['value'],.04+.8*.2+.2)
        self.assertEqual(r['selected_action']['kind'],'CONFIRM')

    def test_matches_step09_at_equal_horizons_and_several_priors(self):
        p=policy(); m=FiniteModel(p); o=ExactOracle(m)
        for weights in [(0,.5,.5,0),(.1,.2,.3,.4),(.7,0,0,.3)]:
            b=m.joint(weights); s=state(belief=b)
            for h in (0,1,2):
                with self.subTest(weights=weights,horizon=h):
                    self.assertAlmostEqual(o.solve(s,horizon=h)['value'],p.plan(s,depth=h)['selected_action']['expected_loss'])

    def test_marginalizing_observation_recovers_predicted_prior(self):
        m=FiniteModel(policy()); b=m.vector(state().belief)
        a=next(x for x in m.catalog.candidates() if x.action_id=='probe:1:1.0')
        branches=m.branches(b,a)
        for i,v in enumerate(m.predict(b,a.setup_s)):
            self.assertAlmostEqual(sum(p*post[i] for _,p,post in branches),v)

    def test_o2_is_explicit_state_conditional_non_deployable_bound(self):
        o=ExactOracle(FiniteModel(policy())); s=state()
        same=o.solve(s,horizon=2); omn=o.omniscient_reference(s,horizon=2)
        self.assertLessEqual(omn['value'],same['value'])
        self.assertEqual(omn['scope'],'non_deployable_omniscient_reference')
        self.assertTrue(omn['conditional_actions']); self.assertFalse(omn['execution_allowed'])
        self.assertEqual(omn['confirmation_model'],'same_noisy_CONFIRM')

    def test_more_information_does_not_mean_free_confirmation(self):
        p=policy(transition=fixed()); o=ExactOracle(FiniteModel(p)); g=p.goals[0]
        r=o.omniscient_reference(state(belief=point(g,'normal')),horizon=0)
        self.assertAlmostEqual(r['value'],.4)

    def test_stop_unknown_and_expiry_guard_oracle(self):
        p=policy(); o=ExactOracle(FiniteModel(p))
        for s in [state(stopped=True),state(remaining_s=0),state(inference_ready=False),state(belief=point(p.goals[0],'unknown_channel'))]:
            self.assertEqual(o.solve(s)['selected_action']['kind'],'PAUSE')

    def test_unknown_compliance_probe_not_point_scored(self):
        c=read('configs/diagnosis.json'); c['actions']['compliance_id']='unknown'
        r=ExactOracle(FiniteModel(policy(c))).solve(state(),horizon=1)
        self.assertFalse(any(x['kind']=='CHANNEL_PROBE' for x in r['candidates']))

    def test_oracle_horizon_is_bounded(self):
        with self.assertRaises(ProtocolError): ExactOracle(FiniteModel(policy())).solve(state(),horizon=4)


class BaselineTests(unittest.TestCase):
    def test_registry_all_methods_and_ablation_ids(self):
        r=read('configs/comparisons/method_registry.json')
        self.assertEqual({x['id'] for x in r['methods']},set(IDS)|{'O1','O2'})
        self.assertEqual({x['id'] for x in read('configs/comparisons/ablations.json')['methods']},set(ABLATIONS))

    def test_all_implemented_methods_produce_legal_offline_requests(self):
        s=suite(); st=state()
        for method in IDS+ABLATIONS:
            r=s.choose(method,st,memory=memory(s.model))
            with self.subTest(method=method):
                self.assertFalse(r['execution_allowed']); json.dumps(r,allow_nan=False)
                if method=='B0': self.assertIsNone(r['selected_action']); continue
                a=next(a for a in s.model.catalog.candidates() if a.action_id==r['selected_action']['action_id'])
                self.assertIsNone(s.model.catalog.legality(a,st))
                self.assertGreaterEqual(r['compute_ns'],0)

    def test_b9_and_m_share_one_step_task_risk(self):
        s=suite(); a=s.choose('B9',state(),horizon_remaining=1); b=s.choose('M',state(),horizon_remaining=1)
        self.assertEqual(a['selected_action']['action_id'],b['selected_action']['action_id'])
        self.assertAlmostEqual(a['planning_value'],b['planning_value'])

    def test_belief_view_does_not_consume_retry_extension_permissions(self):
        s=suite(); original=state(retry_allowed=True,extension_allowed=True)
        for method in IDS[1:]+ABLATIONS:
            view=s.inference_state(method,original)
            self.assertTrue(view.retry_allowed); self.assertTrue(view.extension_allowed)
        self.assertAlmostEqual(s.choose('B9',original,horizon_remaining=1)['planning_value'],s.oracle.solve(original,horizon=1)['value'])

    def test_public_api_rejects_private_truth_and_o2(self):
        s=suite()
        with self.assertRaises(ProtocolError): s.choose('B9',state(),memory={'true_goal':s.model.goals[0]})
        with self.assertRaises(ProtocolError): s.choose('O2',state())
        with self.assertRaises(TypeError): s.choose('B9',state(),true_channel='flip')

    def test_live_ablations_refused(self):
        s=suite()
        for method in ABLATIONS:
            with self.assertRaises(ProtocolError): s.choose(method,state(),execution_mode='live')

    def test_fixed_channel_inference_does_not_change_true_world(self):
        s=suite(); original=state(); before=original.belief.to_dict()
        changed=s.inference_state('B2',original)
        _,z=s.model.marginals(s.model.vector(changed.belief))
        self.assertEqual(z['normal'],1); self.assertEqual(z['flip'],0)
        self.assertEqual(original.belief.to_dict(),before)
        after,_=s.observe('B2',changed,'TASK_QUERY:1.0','1')
        _,z=s.model.marginals(s.model.vector(after.belief)); self.assertEqual(z['flip'],0)

    def test_factorized_ablation_invents_alias_combinations_and_persists(self):
        s=suite(); original=s.model.vector(state().belief)
        fact=s.model.vector(s.inference_state('A_factorized',state()).belief)
        self.assertEqual(original[0],0); self.assertGreater(fact[0],0)
        after,_=s.observe('A_factorized',state(),'TASK_QUERY:1.0','1')
        b=s.model.vector(after.belief)
        for a,v in zip(s.model.factorize(b),b): self.assertAlmostEqual(a,v)

    def test_periodic_rule_changes_phase_without_truth(self):
        s=suite()
        self.assertEqual(s.choose('B5',state(),memory={'diagnostic_count':0})['selected_action']['kind'],'CHANNEL_PROBE')
        self.assertNotEqual(s.choose('B5',state(),memory={'diagnostic_count':1})['selected_action']['kind'],'CHANNEL_PROBE')

    def test_conflict_repeats_and_quality_trigger_uses_public_signal(self):
        s=suite(); mem=memory(s.model)
        self.assertEqual(s.choose('B6',state(),memory=mem)['selected_action']['kind'],'TASK_QUERY')
        self.assertEqual(s.choose('B7',state(),memory={'quality_score':.1})['selected_action']['kind'],'CHANNEL_PROBE')
        with self.assertRaises(ProtocolError): s.choose('B7',state(),memory={'quality_score':2})

    def test_channel_information_not_task_entropy_substitute(self):
        s=suite(); a=next(a for a in s.model.catalog.candidates() if a.action_id=='probe:0:1.0')
        self.assertGreater(s.model.information(s.model.vector(state().belief),a,'channel'),0)
        self.assertEqual(s.choose('B8',state())['selected_action']['kind'],'CHANNEL_PROBE')

    def test_no_probe_baselines_and_ablations_retain_confirmation(self):
        s=suite(policy(transition=fixed())); g=s.model.goals[0]
        for method in ('B3','B4','A_no_probe'):
            self.assertNotEqual(s.choose(method,state())['selected_action']['kind'],'CHANNEL_PROBE')
        r=s.choose('A_no_probe',state(belief=point(g,'normal')))
        self.assertEqual(r['selected_action']['kind'],'CONFIRM')

    def test_prediction_verification_and_rejection_correction_are_not_final_confirmation(self):
        s=suite(policy(transition=fixed())); left,right=s.model.goals
        original=state(belief=s.model.joint((.1,0,.9,0)))
        mem={'predicted_goal':left}
        r=s.choose('B1P',original,memory=mem)
        self.assertEqual(r['selected_action']['kind'],'TASK_QUERY')
        after,mem=s.observe('B1P',original,r['selected_action']['action_id'],'1',memory=mem)
        self.assertTrue(mem['prediction_checked']); self.assertTrue(mem['prediction_rejected'])
        self.assertGreater(mem['correction_queries'],0)
        final=s.choose('B1P',after,memory=mem,horizon_remaining=0)
        self.assertEqual(final['selected_action']['kind'],'CONFIRM')
        self.assertEqual(final['selected_action']['goal_id'],right)
        self.assertFalse(final['execution_allowed'])

    def test_prediction_input_required_and_no_free_horizon_verification(self):
        s=suite()
        with self.assertRaises(ProtocolError): s.choose('B1P',state())
        self.assertEqual(s.choose('B1P',state(),memory=memory(s.model),horizon_remaining=0)['selected_action']['kind'],'PAUSE')

    def test_consistent_prediction_can_skip_correction_but_not_confirmation(self):
        s=suite(policy(transition=fixed())); left,right=s.model.goals
        st=state(belief=s.model.joint((.95,0,.05,0)))
        consistent=s.choose('B1P',st,memory={'predicted_goal':left})
        conflicting=s.choose('B1P',st,memory={'predicted_goal':right})
        self.assertEqual(consistent['selected_action']['kind'],'CONFIRM')
        self.assertEqual(conflicting['selected_action']['kind'],'TASK_QUERY')
        self.assertFalse(consistent['execution_allowed'])

    def test_legacy_results_are_not_fabricated(self):
        r=suite().choose('B0',state())
        self.assertEqual(r['status'],'unavailable_external_legacy_trace'); self.assertIsNone(r['selected_action'])

    def test_source_mixing_changes_inference_not_world_emissions(self):
        s=suite(); a=next(a for a in s.model.catalog.candidates() if a.action_id=='probe:0:1.0')
        b=s.model.vector(state().belief); original=s.model.branches(b,a)
        mixed,_=s.observe('A_mix_sources',state(),a.action_id,'0')
        correct,_=s.observe('B9',state(),a.action_id,'0')
        self.assertNotEqual(mixed.belief.to_dict()['mass'],correct.belief.to_dict()['mass'])
        self.assertEqual(original,s.model.branches(b,a))

    def test_pseudo_adaptation_is_branch_local_and_never_confirmation_labels(self):
        s=suite(); mem={}
        _,a=s.observe('A_pseudo_labels',state(),'TASK_QUERY:1.0','1',memory=mem)
        _,b=s.observe('A_pseudo_labels',state(),'TASK_QUERY:1.0','0',memory=mem)
        self.assertEqual(mem,{}); self.assertNotEqual(a['pseudo_counts'],b['pseudo_counts'])
        target=s.model.goals[0]
        _,c=s.observe('A_pseudo_labels',state(),'confirm:'+target,'0',memory=a)
        self.assertEqual(a['pseudo_counts'],c['pseudo_counts'])

    def test_ignored_probe_cost_still_spends_budget_and_actual_cost(self):
        s=suite(); a=next(a for a in s.model.catalog.candidates() if a.action_id=='probe:0:1.0')
        self.assertEqual(s.model.cost(a,'ignore_probe_cost'),0)
        self.assertAlmostEqual(s.model.cost(a),.04)
        after,_=s.observe('A_ignore_probe_cost',state(),a.action_id,'0')
        self.assertEqual(after.remaining_s,18)
        value=s.evaluate('A_ignore_probe_cost',state(),horizon=1)['actual_model_expected_loss']
        planning=s.choose('A_ignore_probe_cost',state(),horizon_remaining=1)['planning_value']
        self.assertAlmostEqual(value-planning,.04)

    def test_no_consequence_planner_does_not_change_evaluator_loss(self):
        s=suite(); target=s.model.goals[0]; b=s.model.vector(state().belief)
        self.assertNotEqual(s.model.terminal_loss(target,b),s.model.terminal_loss(target,b,no_task_consequence=True))
        r=s.evaluate('A_no_task_consequence',state(),horizon=1)
        self.assertEqual(r['scope'],'analytic_fixed_policy_audit_not_experiment')

    def test_fixed_duration_never_selects_long_probe(self):
        s=suite(); r=s.choose('A_fixed_duration',state(extension_allowed=True))
        self.assertIn(r['selected_action']['duration_s'],(0,1))

    def test_all_baselines_obey_stop_even_without_external_dependencies(self):
        s=suite()
        for method in IDS+ABLATIONS:
            r=s.choose(method,state(stopped=True))
            self.assertEqual(r['selected_action']['kind'],'PAUSE')
            self.assertEqual(r['reason'],'stop_latched')

    def test_exhausted_budget_no_implicit_old_goal(self):
        s=suite()
        for method in IDS[1:]+ABLATIONS:
            r=s.choose(method,state(remaining_s=0),memory=memory(s.model))
            self.assertEqual(r['selected_action']['kind'],'PAUSE')

    def test_fixed_policy_audit_respects_horizon_and_exact_one_step_value(self):
        s=suite(); o=s.oracle.solve(state(),horizon=1)
        self.assertAlmostEqual(s.evaluate('B9',state(),horizon=1)['actual_model_expected_loss'],o['value'])
        self.assertAlmostEqual(s.evaluate('M2',state(),horizon=2)['actual_model_expected_loss'],s.oracle.solve(state(),horizon=2)['value'])


class SelectionTests(unittest.TestCase):
    def call(self,records=None,partitions=None,rule=None,sim=True,expected=None):
        r,p,c=selection_fixture()
        return select_validation(records or r,partitions=partitions or p,rule=rule or c,
            expected_candidates=expected or [x['candidate_id'] for x in r],simulation_only=sim)

    def test_simulated_validation_can_check_code_but_not_deploy(self):
        r=self.call(); self.assertEqual(r['chosen_method'],'B9')
        self.assertEqual(r['status'],'simulation_selection_check_only'); self.assertFalse(r['deployment_allowed'])

    def test_formal_test_cannot_select(self):
        r,p,c=selection_fixture(); r[0]['split']='test'
        with self.assertRaises(ProtocolError): self.call(r)

    def test_real_choice_rejects_synthetic_validation(self):
        with self.assertRaises(ProtocolError): self.call(sim=False)

    def test_group_leakage_rejected(self):
        r,p,c=selection_fixture(); p['test']=['val']
        with self.assertRaises(ProtocolError): self.call(partitions=p)

    def test_partial_candidate_grid_rejected(self):
        r,p,c=selection_fixture()
        with self.assertRaises(ProtocolError): self.call(records=r[:1])

    def test_unpaired_episodes_rejected(self):
        r,p,c=selection_fixture(); r[0]['episode_ids'][0]='different-episode'
        with self.assertRaises(ProtocolError): self.call(r)

    def test_failed_common_protection_not_eligible(self):
        r,p,c=selection_fixture(); r[1]['common_protocol_checks_pass']=False
        self.assertEqual(self.call(r)['chosen_method'],'B5')

    def test_no_eligible_method_does_not_relax_rule(self):
        r,p,c=selection_fixture(); c['wrong_start_rate_ceiling']=0
        with self.assertRaises(ProtocolError): self.call(rule=c)

    def test_missing_pilot_rule_rejected(self):
        r,p,c=selection_fixture(); c['status']='awaiting_independent_pilot_limits'
        with self.assertRaises(ProtocolError): self.call(rule=c)

    def test_config_and_rule_hashes_retained(self):
        r,p,c=selection_fixture(); v=self.call()
        self.assertEqual(v['rule_sha256'],digest(c))
        self.assertEqual(v['chosen_config_sha256'],r[1]['config_sha256'])


if __name__=='__main__': unittest.main()
