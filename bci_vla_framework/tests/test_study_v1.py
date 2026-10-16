import unittest
from copy import deepcopy
from unittest.mock import patch
from bci_vla.study_v1.world import StudyWorld,CASES
from bci_vla.study_v1.runner import run_episode
from bci_vla.study_v1.statistics import paired_summary,judge
from bci_vla.benchmark.offline import make_session

class StudyTests(unittest.TestCase):
 def world(self,case):
  s=make_session('unit');return StudyWorld(case,0,s.policy.channel,s.policy.goals),s
 def test_observable_change_orders_have_distinct_initial_truth(self):
  a,s=self.world(CASES[4]);b,_=self.world(CASES[5]);self.assertNotEqual(a.truth_at(1),b.truth_at(1));self.assertEqual(a.truth_at(3),b.truth_at(3))
 def test_repeated_changes_follow_private_timeline(self):
  w,s=self.world(CASES[12]);self.assertNotEqual(w.truth_at(1),w.truth_at(3));self.assertNotEqual(w.truth_at(3),w.truth_at(5))
 def test_invalid_window_returns_no_response_instead_of_mislabeled_sample(self):
  w,s=self.world(CASES[11]);q=s.begin_interaction('TASK_QUERY',now='2026-10-15T00:00:00+00:00',monotonic_now=0)
  self.assertEqual(w.emit(q,start=1,duration=1),'NO_RESPONSE');self.assertTrue(w.draws[-1]['invalid_nonstationary_window_suppressed'])
 def test_no_probe_ablation_does_not_select_probe(self):
  p,e,_,_=run_episode(CASES[2],0,'A_no_probe','unit-ablation')
  self.assertIsNone(e['runtime_error']);self.assertTrue(all(d['selected_action']['kind']!='CHANNEL_PROBE' for d in p['decisions']))
 def test_executor_failure_never_changes_channel(self):
  case=dict(CASES[0],prior=[1,0,0,0],execution_failure_probability=1.)
  p,e,_,_=run_episode(case,0,'M','unit-failure')
  self.assertIsNone(e['runtime_error']);self.assertEqual(e['final_state'],'FAILED');self.assertTrue(e['execution_does_not_update_channel'])
 def test_all_methods_replay_on_development_seed(self):
  from bci_vla.study_v1.runner import METHODS
  for m in METHODS:
   p,e,_,_=run_episode(CASES[0],1,m,'unit-'+m);self.assertIsNone(e['runtime_error']);self.assertTrue(p['replay_matches_each_checkpoint'])
 def test_paired_seed_cluster_ties_are_not_equivalence_or_G4(self):
  rows=[dict(seed=s,case_id=c,method=m,faithful_virtual_completion=True,wrong_start=False,elapsed_s=3.) for s in range(4) for c in ['a','b'] for m in ['M','B9']]
  p=[paired_summary(rows,'M','B9',metric,bootstrap=100) for metric in ['faithful_virtual_completion','wrong_start','elapsed_s']]
  self.assertEqual(p[0]['cluster_percentile_interval'],[0,0]);self.assertGreater(p[0]['bounded_difference_Hoeffding_interval'][1],0)
  self.assertFalse(judge(p)['synthetic_primary_acceptance_passed']);self.assertFalse(judge(p)['formal_human_or_robot_G4_passed'])
 def test_incomplete_pairs_cannot_be_silently_dropped(self):
  with self.assertRaises(KeyError):paired_summary([dict(seed=1,case_id='a',method='M',wrong_start=False)],'M','B9','wrong_start')
