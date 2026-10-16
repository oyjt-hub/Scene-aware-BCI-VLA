"""Seed-cluster paired estimates; never interpret repeated seeds as subjects."""
import math,random
from statistics import mean

def paired_summary(rows,left,right,metric,*,bootstrap=2000,alpha=.05,bootstrap_seed=16016):
 lookup={(r['seed'],r['case_id'],r['method']):r for r in rows};seeds=sorted({r['seed'] for r in rows});cases=sorted({r['case_id'] for r in rows})
 differences=[]
 for seed in seeds:
  differences.append(mean(float(lookup[seed,c,left][metric])-float(lookup[seed,c,right][metric]) for c in cases))
 rng=random.Random(bootstrap_seed);samples=sorted(mean(rng.choices(differences,k=len(differences))) for _ in range(bootstrap))
 lo=samples[max(0,int(bootstrap*alpha/2))];hi=samples[min(bootstrap-1,int(bootstrap*(1-alpha/2)))]
 result=dict(left=left,right=right,metric=metric,paired_seed_clusters=len(seeds),cases_per_cluster=len(cases),mean_difference=mean(differences),
  cluster_percentile_interval=[lo,hi],alpha=alpha,bootstrap_resamples=bootstrap,all_seed_differences_equal=len(set(differences))==1,
  scope='conditional synthetic generator; scenarios fixed; not subject inference')
 if metric not in {'elapsed_s','user_busy_s','probe_s'}:
  width=math.sqrt(2*math.log(2/alpha)/len(seeds))
  result['bounded_difference_Hoeffding_interval']=[max(-1,result['mean_difference']-width),min(1,result['mean_difference']+width)]
 return result

def judge(primary,*,margin=.05,safety=.02,time_margin=1.):
 p={r['metric']:r for r in primary}
 gain=p['faithful_virtual_completion']['bounded_difference_Hoeffding_interval'][0]>margin
 safe=p['wrong_start']['bounded_difference_Hoeffding_interval'][1]<=safety
 timely=p['elapsed_s']['cluster_percentile_interval'][1]<=time_margin
 return dict(synthetic_primary_acceptance_passed=gain and safe and timely,completion_gain_supported=gain,wrong_start_noninferiority_supported=safe,
  full_episode_time_condition_supported=timely,formal_human_or_robot_G4_passed=False,
  verdict='SUPPORTED_ONLY_IN_REGISTERED_SYNTHETIC_SCOPE' if gain and safe and timely else 'BENEFIT_NOT_ESTABLISHED',
  zero_bootstrap_width_is_not_equivalence_proof=True)
