"""Regression tests for authorization-aware planning; no threshold relaxation."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from bci_vla.protocols import ProtocolError,thaw
from bci_vla.core.diagnostic_policy import PlanningState
from bci_vla.baselines.oracle import ExactOracle
from bci_vla.benchmark.offline import make_session,read
from bci_vla.benchmark.synthetic_world import CASES
from bci_vla.runtime_v2.controller import RuntimeController
from bci_vla.runtime_v2.planning import AuthorizationAwarePolicy,VERSION,checked_gate
from bci_vla.runtime_v2.audit import components,gate_replay,alignment_audit
from bci_vla.runtime_v2.runner import run_episode


def changed(state,**updates):
    values={field:getattr(state,field) for field in state.__dataclass_fields__}
    values['contracts']=read('examples/diagnosis/state.json')['contracts'];values.update(updates)
    return PlanningState(**values)


class GateCostTests(unittest.TestCase):
    def test_original_low_posterior_counterexample_now_costs_uncompleted_pause(self):
        row=gate_replay()
        self.assertAlmostEqual(row['predicted_goal_probability'],.32/.41)
        self.assertFalse(row['candidate_authorizes']);self.assertFalse(row['actual_authorizes'])
        self.assertEqual(row['candidate_continuation_loss'],1)
        self.assertEqual(row['independent_O1_continuation_loss'],1)

    def test_whole_confirm_row_includes_blocked_accept_reject_and_return_time(self):
        _,policy,model,state=components();_,rows=policy._evaluate(model.vector(state.belief),state,0)
        row=next(r for r in rows if r['kind']=='CONFIRM' and r['goal_id']==policy.goals[0])
        self.assertAlmostEqual(row['elapsed_s'],2.01);self.assertAlmostEqual(row['expected_loss'],1+.02*2.01)
        accept=next(b for b in row['branches'] if b['observation']=='0')
        self.assertEqual(accept['next_action'],'PAUSE_AUTHORIZATION_BLOCKED')

    def test_valid_high_probability_accept_remains_possible(self):
        row=gate_replay((.9,0,.1,0))
        self.assertTrue(row['candidate_authorizes']);self.assertTrue(row['actual_authorizes'])
        self.assertLess(row['candidate_continuation_loss'],row['pause_loss'])

    def test_return_delay_over_age_limit_blocks_even_certain_goal(self):
        row=gate_replay((1,0,0,0),return_s=2.01)
        self.assertFalse(row['actual_authorizes']);self.assertEqual(row['candidate_continuation_loss'],1)
        _,policy,model,state=components((1,0,0,0),return_s=2.01)
        confirms=[a for a in model.catalog.candidates() if a.kind=='CONFIRM']
        self.assertTrue(all(model.catalog.legality(a,state)=='confirmation_expires_before_authorization' for a in confirms))

    def test_delayed_prediction_matches_actual_dynamic_authorizer(self):
        for prior in ((.4,0,.6,0),(.6,0,.4,0),(1,0,0,0)):
            for dt in (.01,2):
                with self.subTest(prior=prior,delay=dt):
                    row=gate_replay(prior,fixed=False,return_s=dt)
                    self.assertEqual(row['candidate_authorizes'],row['actual_authorizes'])

    def test_gate_spec_rejects_extra_or_invalid_fields(self):
        _,_,model,_=components();gate=thaw(model.gate)
        for key,value in [('minimum_goal_probability',-1),('return_s',-1),('version','live')]:
            bad=dict(gate);bad[key]=value
            with self.assertRaises(ProtocolError):checked_gate(bad)
        with self.assertRaises(ProtocolError):checked_gate(dict(gate,truth=True))

    def test_threshold_and_age_are_copied_from_unchanged_session(self):
        s=make_session('opaque');c=RuntimeController(s,'M')
        self.assertEqual(c.model.gate['minimum_goal_probability'],.8)
        self.assertEqual(c.model.gate['max_confirmation_age_s'],2)
        with self.assertRaises(TypeError):c.model.gate['minimum_goal_probability']=.1


class GuardAndIndependentOracleTests(unittest.TestCase):
    def test_O1_runs_when_candidate_gate_and_solver_are_forbidden(self):
        _,policy,model,state=components()
        with patch.object(AuthorizationAwarePolicy,'_evaluate',side_effect=AssertionError('candidate solver forbidden')), \
             patch.object(AuthorizationAwarePolicy,'confirmation_value',side_effect=AssertionError('candidate terminal algebra forbidden')):
            result=ExactOracle(model).solve(state,horizon=2)
        self.assertEqual(result['scope'],'offline_exact_same_information')

    def test_missing_contracts_cannot_be_bypassed_by_accept_probability(self):
        _,policy,model,state=components();state=changed(state,contracts={})
        result=policy.plan(state,depth=0);self.assertEqual(result['selected_action']['kind'],'PAUSE')

    def test_stop_pending_or_unready_never_yield_execution_authority(self):
        _,policy,model,state=components()
        for update in ({'stopped':True},{'pending':True},{'inference_ready':False}):
            v=changed(state,**update)
            self.assertEqual(policy.plan(v,depth=1)['selected_action']['kind'],'PAUSE')
            self.assertEqual(ExactOracle(model).solve(v,horizon=1)['selected_action']['kind'],'PAUSE')

    def test_remaining_budget_includes_authorization_return_delay(self):
        _,policy,model,state=components((1,0,0,0));state=changed(state,remaining_s=2.)
        self.assertEqual(policy.plan(state,depth=0)['selected_action']['kind'],'PAUSE')
        self.assertTrue(all(model.catalog.legality(a,state)=='insufficient_budget' for a in model.catalog.candidates() if a.kind=='CONFIRM'))

    def test_scene_or_contract_expiry_includes_return_delay(self):
        _,policy,model,state=components((1,0,0,0))
        scene=changed(state,scene_valid_until_s=state.now_s+2.)
        self.assertEqual(policy.plan(scene,depth=0)['selected_action']['kind'],'PAUSE')
        contracts=read('examples/diagnosis/state.json')['contracts']
        for c in contracts.values():c['valid_until']='2026-10-15T00:00:02+00:00'
        contract=changed(state,contracts=contracts)
        self.assertEqual(policy.plan(contract,depth=0)['selected_action']['kind'],'PAUSE')

    def test_stale_contract_versions_are_rejected_before_planning(self):
        _,policy,model,state=components();contracts=read('examples/diagnosis/state.json')['contracts']
        for c in contracts.values():c['scene_revision']=2
        with self.assertRaises(ProtocolError):changed(state,contracts=contracts)

    def test_original_observation_and_actual_evidence_filter_remain_unmodified(self):
        s=make_session('opaque');before=s.filter.snapshot();c=RuntimeController(s,'M')
        c.choose();self.assertEqual(s.filter.snapshot(),before)
        self.assertIsNot(s.policy,c.model.policy)
        self.assertEqual(c.planner_ref['role'],'external_advisory_proposer_not_execution_authority')

    def test_post_compute_recheck_uses_current_public_view(self):
        s=make_session('opaque');c=RuntimeController(s,'M');a=c.choose()
        self.assertIsNotNone(a)
        if a['kind']=='CONFIRM':self.assertTrue(c.decisions[-1]['post_compute_authorization_prediction']['allowed'])
        self.assertEqual(c.decisions[-1]['planner_ref']['version'],VERSION)


class FullRepairAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.audit=alignment_audit()

    def test_all29_actual_authorization_cases_match_candidate_and_independent_O1(self):
        self.assertEqual(len(self.audit['gate_cases']),29)
        self.assertTrue(self.audit['all_actual_gate_predictions_match'])
        self.assertTrue(self.audit['counterexample_fixed'])
        for row in self.audit['gate_cases']:
            self.assertAlmostEqual(row['candidate_continuation_loss'],row['independent_O1_continuation_loss'])

    def test_all80_same_horizon_queries_match_independent_oracle(self):
        self.assertEqual(len(self.audit['same_horizon_rows']),80)
        self.assertLessEqual(self.audit['max_same_horizon_regret'],1e-9)
        self.assertTrue(self.audit['all_thresholds_unchanged'])
        self.assertFalse(self.audit['full_runtime_optimality_claim'])

    def test_all_methods_share_fixed_authorization_rules_and_replay(self):
        for method in ('B4','B5','B7','B9','M','M2'):
            with self.subTest(method=method):
                p,e,private,journal=run_episode(CASES[1],0,method,'opaque')
                self.assertEqual(p['planner_ref']['gate']['minimum_goal_probability'],.8)
                self.assertTrue(p['replay_matches_each_checkpoint'])
                self.assertFalse(p['termination'].startswith('authorization_or_dispatch_blocked:'))

    def test_C11_negative_result_is_not_erased_by_cost_fix(self):
        p,e,_,_=run_episode(CASES[3],0,'M','opaque')
        self.assertEqual(p['final_state'],'SUCCEEDED');self.assertTrue(e['wrong_start'])
        self.assertFalse(e['faithful_virtual_completion'])


if __name__=='__main__':unittest.main()
