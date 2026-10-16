"""Step12 mechanisms and full offline episodes with an isolated truth join."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from bci_vla.protocols import ProtocolError,thaw
from bci_vla.benchmark.offline import make_session,OfflineController,union_duration,METHODS
from bci_vla.benchmark.synthetic_world import PrivateWorld,CASES
from bci_vla.benchmark.runner import run_episode,summarize
from bci_vla.benchmark.audits import mechanisms,faults,oracle_audit
from bci_vla.benchmark.runtime_gate import authorization_gate_counterexample


class MechanismBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.result=mechanisms()

    def test_both_binary_codebooks_leave_competing_worlds_identical(self):
        rows=self.result['alias_rows'];self.assertEqual(len(rows),6)
        for row in rows:self.assertAlmostEqual(row['right_normal'],row['left_flip'])

    def test_informative_probe_changes_goal_through_correlation(self):
        # Imperfect compliance mixes intended and opposite markers. The analytic
        # posterior is .7675/(.7675+.0875), below the perfect-compliance value.
        self.assertAlmostEqual(max(self.result['after_goal'].values()),.7675/.855)
        self.assertGreater(max(self.result['after_goal'].values()),.5)
        self.assertTrue(self.result['checks']['probe_is_goal_independent_given_channel'])

    def test_uninformative_probe_preserves_every_joint_cell(self):
        for p,q in zip(self.result['prior'],self.result['uninformative_posterior']):self.assertAlmostEqual(p,q)
        self.assertTrue(self.result['checks']['uninformative_probe_has_zero_mutual_information'])

    def test_goal_independence_is_preserved_for_uncorrelated_prior(self):
        self.assertTrue(self.result['checks']['independent_goal_stays_half'])

    def test_new_codebook_cannot_be_used_as_an_uncalibrated_repair(self):
        self.assertTrue(self.result['checks']['recode_unavailable_without_separate_calibration'])


class TruthIsolationTests(unittest.TestCase):
    def test_online_module_imports_no_private_world(self):
        path=Path(__file__).resolve().parents[1]/'bci_vla/benchmark/offline.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        self.assertFalse(any(name and ('synthetic_world' in name or 'runner' in name) for name in imports))

    def test_controller_operates_when_private_world_access_is_forbidden(self):
        s=make_session('opaque');c=OfflineController(s,'M')
        with patch.object(PrivateWorld,'truth_at',side_effect=AssertionError('private truth forbidden')):
            decision=c.choose()
        self.assertIsNotNone(decision);self.assertFalse(hasattr(c,'seed'));self.assertFalse(hasattr(c,'world'))

    def test_identical_public_inputs_do_not_depend_on_private_schedule(self):
        s1=make_session('opaque');s2=make_session('opaque')
        c1=OfflineController(s1,'M');c2=OfflineController(s2,'M')
        unused=PrivateWorld(CASES[3],987,s1.policy.channel,s1.policy.goals)
        self.assertEqual(c1.choose()['action_id'],c2.choose()['action_id'])
        self.assertEqual(c1.decisions[0]['public_input'],c2.decisions[0]['public_input'])

    def test_simultaneous_goal_channel_switch_can_leave_initial_output_identical(self):
        s=make_session('opaque');c=OfflineController(s,'M')
        q,_,_=c.begin({'kind':'TASK_QUERY','setup_s':1,'duration_s':1,'elapsed_s':2})
        stable=PrivateWorld(CASES[0],42,s.policy.channel,s.policy.goals)
        both=PrivateWorld(CASES[3],42,s.policy.channel,s.policy.goals)
        self.assertEqual(stable.emit(q,start=1,duration=1),both.emit(q,start=1,duration=1))
        self.assertNotEqual(stable.truth_at(1),both.truth_at(1))
        self.assertEqual(stable.draws[0]['distribution'],both.draws[0]['distribution'])


class FullEpisodeTests(unittest.TestCase):
    def test_legal_full_pipeline_retains_independent_truth_evaluation(self):
        public,evaluation,private,journal=run_episode(CASES[0],0,'M','opaque')
        self.assertEqual(public['final_state'],'SUCCEEDED');self.assertTrue(evaluation['faithful_virtual_completion'])
        self.assertTrue(public['replay_matches_each_checkpoint']);self.assertEqual(public['backend_submissions'],1)
        self.assertNotIn('seed',public);self.assertNotIn('case',public);self.assertIn('draws',private)
        self.assertFalse(public['physical_execution_allowed'])

    def test_contract_success_does_not_override_private_goal_truth(self):
        public,evaluation,_,_=run_episode(CASES[3],0,'M','opaque')
        self.assertEqual(public['final_state'],'SUCCEEDED')
        self.assertTrue(evaluation['wrong_start']);self.assertFalse(evaluation['faithful_virtual_completion'])

    def test_same_probe_methods_all_use_real_replay_session(self):
        for method in METHODS:
            with self.subTest(method=method):
                public,row,_,_=run_episode(CASES[1],0,method,'opaque')
                self.assertTrue(public['replay_matches_each_checkpoint'])
                self.assertLessEqual(public['backend_submissions'],1)
                self.assertGreaterEqual(row['elapsed_s'],2)

    def test_all_started_episodes_remain_in_summary_including_abstentions(self):
        rows=[run_episode(CASES[0],0,'M','opaque-a')[1],run_episode(CASES[1],0,'M','opaque-b')[1]]
        summary=summarize(rows);self.assertEqual(sum(g['episodes'] for g in summary),2)
        self.assertEqual(sum(r['unresolved'] for r in rows),1)

    def test_time_union_handles_overlap_and_includes_non_user_phases(self):
        spans=[{'start_s':0,'end_s':2},{'start_s':1,'end_s':3},{'start_s':5,'end_s':6}]
        self.assertEqual(union_duration(spans),4)
        public,_,_,_=run_episode(CASES[0],0,'M','opaque')
        self.assertGreater(public['elapsed_s'],public['user_busy_s'])
        self.assertAlmostEqual(public['elapsed_s'],public['timeline_union_s'],places=4)

    def test_zero_compute_budget_discards_result_before_collecting_or_submitting(self):
        s=make_session('opaque');c=OfflineController(s,'M',compute_budget_ns=0)
        self.assertIsNone(c.choose());self.assertEqual(s.backend.submission_count,0)
        self.assertEqual(c.termination,'policy_compute_budget_exceeded');self.assertTrue(s.stop_latched)

    def test_budget_blocks_next_request_after_actual_compute(self):
        s=make_session('opaque');c=OfflineController(s,'M',budget_s=0)
        self.assertIsNone(c.choose());self.assertEqual(s.backend.submission_count,0)


class OfflineFaultAndOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.faults=faults();cls.oracle=oracle_audit()

    def test_required_faults_present_and_all_replay(self):
        names={r['fault'] for r in self.faults['cases']}
        self.assertEqual(len(names),10)
        self.assertTrue({'no_response','old_menu','expired_confirmation','duplicate_confirmation_and_dispatch',
            'target_disappeared','policy_timeout','illegal_action','cancel_late_response','stop_during_execution',
            'unknown_submission_no_retry'}<=names)
        self.assertTrue(all(r['passed'] and r['replay_matches'] and not r['physical_stop_verified'] for r in self.faults['cases']))

    def test_same_horizon_oracle_equal_on_expanded_prior_and_menu_grid(self):
        self.assertEqual(len(self.oracle['rows']),80)
        self.assertLessEqual(self.oracle['max_same_horizon_regret'],1e-9)
        self.assertLessEqual(self.oracle['max_planning_value_gap'],1e-9)

    def test_scale_and_actual_compute_are_reported_with_model_scope(self):
        self.assertEqual({r['hidden_states'] for r in self.oracle['rows']},{4})
        self.assertEqual({r['horizon'] for r in self.oracle['rows']},{0,1,2})
        self.assertTrue(all(r['oracle_compute_ns']>0 and r['method_compute_ns']>0 for r in self.oracle['rows']))
        self.assertFalse(self.oracle['optimal_pomdp_claim']);self.assertIn('authorization',self.oracle['runtime_gap'])

    def test_noisy_accept_counterexample_detects_planning_runtime_gap(self):
        result=authorization_gate_counterexample()
        self.assertAlmostEqual(result['posterior_after_noisy_confirm'],.32/.41)
        self.assertLess(result['posterior_after_noisy_confirm'],result['authorization_threshold'])
        self.assertEqual(result['backend_submissions'],0);self.assertFalse(result['gate_weakened'])
        self.assertTrue(result['requires_versioned_planner_and_O1_alignment'])


if __name__=='__main__':unittest.main()
