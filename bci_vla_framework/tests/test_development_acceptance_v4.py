from copy import deepcopy
import unittest
from bci_vla.evaluation_v4.development import aggregate, paired_scene_interval, sample_size_draft, score_trace, numerically_equal


def row(group,episode,scenario,resolved=False,wrong=False):
    return {'group_id':group,'episode_id':episode,'scenario':scenario,'resolved':resolved,'wrong_episode':wrong,
            'submissions':int(resolved or wrong),'wrong_submissions':int(wrong),'questions':3,'waits':9,
            'no_responses':1,'rejects':0,'clock_cost_s':24.}


class DevelopmentAcceptanceTests(unittest.TestCase):
    def test_floating_tolerance_never_hides_a_different_choice_or_context(self):
        first={'choice':'o0','context':1,'score':.27987093159073534}
        self.assertTrue(numerically_equal(first,{**first,'score':.2798709315907354}))
        self.assertFalse(numerically_equal(first,{**first,'choice':'o1'}))
        self.assertFalse(numerically_equal(first,{**first,'context':2}))
        self.assertFalse(numerically_equal(first,{**first,'score':.279871}))
    def test_waits_and_failure_cases_stay_in_resolution_and_risk_denominators(self):
        rows=[row('g','e',str(i),resolved=i==0,wrong=i==1) for i in range(4)]
        result=aggregate(rows)
        self.assertEqual(result['resolution'],.25)
        self.assertEqual(result['wrong_episode_rate'],1.)
        self.assertEqual(result['base_episodes'],1)
        self.assertEqual(result['condition_instances'],4)
        self.assertEqual(result['no_responses'],4)

    def test_always_wait_is_not_success_and_has_no_conditional_submission_estimate(self):
        result=aggregate([row('g','e',str(i)) for i in range(4)])
        self.assertEqual(result['resolution'],0.)
        self.assertIsNone(result['conditional_submission_error'])

    def test_duplicate_conditions_rejected_instead_of_increasing_sample_size(self):
        r=row('g','e','stable')
        with self.assertRaisesRegex(ValueError,'duplicate'):aggregate([r,deepcopy(r)])

    def test_interval_resamples_paired_scenes_not_variants(self):
        a=[row(g,'e',str(i),resolved=True) for g in ['g0','g1'] for i in range(4)]
        b=[row(g,'e',str(i),resolved=False) for g in ['g0','g1'] for i in range(4)]
        result=paired_scene_interval(a,b,replicates=100)
        self.assertEqual(result['difference'],1.)
        self.assertEqual(result['scene_groups'],2)
        self.assertFalse(result['inferential_claim_allowed'])
        with self.assertRaisesRegex(ValueError,'same paired'):paired_scene_interval(a,b[:4])

    def test_zero_errors_eight_episodes_does_not_prove_ten_percent_population_risk(self):
        result=sample_size_draft()
        self.assertGreater(result['currently_observed_zero_errors_8_episode_upper_if_independent'],.3)
        self.assertEqual(result['zero_scene_errors_min_scenes_for_one_sided_95_upper_le_10_percent'],29)
        self.assertFalse(result['formal_sample_count_locked'])

    def test_goal_update_is_censored_without_a_trusted_public_anchor(self):
        trace={'journal':[{'kind':'WAIT'}]*12,'submissions':[],'private_scoring_only':[],
               'compute_ns':[1]*12,'partition':'validation','method':'wait','base_episode_id':'e','horizon_s':24}
        world={'initial_goal':'a','later_goal':'b','initial_success':.9,'later_success':.9,'change_at_s':8}
        result=score_trace(trace,world,'g')
        self.assertFalse(result['resolved'])
        self.assertTrue(result['update_censored'])
        self.assertEqual(result['censor_after_change_s'],16)
        self.assertIsNone(result['identifiable_response_latency_s'])


if __name__=='__main__':unittest.main()
