"""Development-only outcome scoring. Private worlds never enter a policy."""
from collections import Counter, defaultdict
from statistics import NormalDist
import math
import random

from bci_vla.data_v4.channel import world_at
from bci_vla.data_v4.contracts import validate_public
from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.calibration_v4.evidence_gate import confirmation_evidence_gate


def numerically_equal(first, second, tolerance=1e-12):
    """Only floats have a tolerance. Roles, choices, versions and IDs are exact."""
    if type(first) is not type(second):return False
    if isinstance(first,float):return math.isfinite(first) and math.isfinite(second) and abs(first-second)<=tolerance
    if isinstance(first,dict):return first.keys()==second.keys() and all(numerically_equal(first[k],second[k],tolerance) for k in first)
    if isinstance(first,list):return len(first)==len(second) and all(numerically_equal(a,b,tolerance) for a,b in zip(first,second))
    return first==second


def score_trace(trace, world, group_id):
    rows = [j for j in trace['journal'] if j['kind']=='INTERACTION']
    waits = [j for j in trace['journal'] if j['kind']=='WAIT']
    changes = world['initial_goal'] != world['later_goal']
    change_time = world['change_at_s']
    horizon = trace['horizon_s']
    scoring = {s['request_id']:s for s in trace['private_scoring_only']}
    submissions = trace['submissions']
    goal, _ = world_at(world,horizon)
    last = submissions[-1] if submissions else None
    resolved = bool(last and last['task']['source_id']==goal and
                    (not changes or last['submitted_at_s']>=change_time))
    correct_updates = [s['submitted_at_s'] for s in submissions if changes and
                       s['submitted_at_s']>=change_time and s['task']['source_id']==world['later_goal']]
    first_update = min(correct_updates) if correct_updates else None
    old_submissions = sum(changes and s['submitted_at_s']>=change_time and
                          s['task']['source_id']==world['initial_goal'] for s in submissions)
    roles = Counter(j['decision']['action'] for j in rows)
    wrong = sum(not scoring[s['request_id']]['source_matches_latent_goal'] for s in submissions)
    return {'split':trace['partition'],'method':trace['method'],'group_id':group_id,
            'episode_id':trace['base_episode_id'],'resolved':resolved,
            'wrong_episode':wrong>0,'wrong_submissions':wrong,'submissions':len(submissions),
            'questions':len(rows),'waits':len(waits),'roles':dict(roles),
            'no_responses':sum(j['event']['role']=='NO_RESPONSE' for j in rows),
            'rejects':sum(j['event']['role']=='REJECT' for j in rows),
            'parsing_failures':0,'condition_failures':0,
            'clock_cost_s':2.*(len(rows)+len(waits)),
            'goal_changed':changes,'change_at_s':change_time if changes else None,
            'first_correct_update_s':first_update,
            'total_update_latency_s':first_update-change_time if first_update is not None else None,
            'update_censored':bool(changes and first_update is None),
            'censor_after_change_s':max(0.,horizon-change_time) if changes else None,
            'old_target_submissions_after_change':int(old_submissions),
            # A finite set of noisy responses does not identify a hidden goal.
            'identified_public_change_anchor_s':None,
            'identifiable_anchor_status':'NOT_AVAILABLE_NO_TRUSTED_EXPLICIT_DECLARATION' if changes else 'NOT_APPLICABLE',
            'identifiable_response_latency_s':None,
            'policy_runtime_ns':sum(trace['compute_ns']),
            'VLA_called':False,'real_dynamics_advanced':False}


def audit_trace(trace, config=None):
    checks = Counter(); receipts = set()
    assert trace['partition'] in ('validation','calibration_audit')
    assert trace['VLA_called'] is False and trace['real_dynamics_advanced'] is False
    submissions = {s['request_id']:s for s in trace['submissions']}
    for j in trace['journal']:
        if j['kind'] not in ('WAIT','INTERACTION'):
            continue
        public = j['input'];validate_public(public);checks['public_records']+=1
        if j['kind']=='WAIT':
            continue
        event=j['event'];q=event['question'];checks['questions']+=1
        assert q['context']==public['context'] and q['presentation_ack']
        if j['decision']['action']=='CONFIRM':
            checks['confirmations']+=1
            assert j['decision']['source_id']==q['proposal']['source_id']
            if config is not None:
                assert confirmation_evidence_gate(public,j['decision']['source_id'],config) is None
                checks['evidence_contracts']+=1
        request=j['simulated_submission']
        if request is None:
            continue
        assert event['role']=='CONFIRM' and event['valid']
        assert request['request_id'] in submissions and request==submissions[request['request_id']]
        assert request['task']==q['proposal']
        assert request['task']['context']==j['public_after']['context']
        assert request['confirmation_event_id']==event['event_id']
        assert request['execution_authorized'] is False and request['VLA_called'] is False and request['physical_dispatched'] is False
        assert request['request_id'] not in receipts
        receipts.add(request['request_id']);checks['simulated_requests']+=1
    assert len(receipts)==len(submissions)
    return dict(checks)


def aggregate(rows):
    """One base episode first, then equal scene weights. Variants never inflate N."""
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        groups[r['group_id']][r['episode_id']].append(r)
    base=[]
    for group,episodes in sorted(groups.items()):
        for episode,items in sorted(episodes.items()):
            if all('scenario' in r for r in items) and len({r['scenario'] for r in items})!=len(items):
                raise ValueError('duplicate condition cannot inflate a base episode')
            base.append({'group_id':group,'episode_id':episode,
                         'resolution':sum(r['resolved'] for r in items)/len(items),
                         'wrong_episode':any(r['wrong_episode'] for r in items),
                         'condition_count':len(items)})
    def macro(key):
        return sum(sum(float(r[key]) for r in base if r['group_id']==g)/len(groups[g]) for g in groups)/len(groups)
    return {'scene_groups':len(groups),'base_episodes':len(base),'condition_instances':len(rows),
            'resolution':macro('resolution'),'wrong_episode_rate':macro('wrong_episode'),
            'resolved_conditions':sum(r['resolved'] for r in rows),
            'wrong_base_episodes':sum(r['wrong_episode'] for r in base),
            'submissions':sum(r['submissions'] for r in rows),
            'wrong_submissions':sum(r['wrong_submissions'] for r in rows),
            'conditional_submission_error':sum(r['wrong_submissions'] for r in rows)/sum(r['submissions'] for r in rows) if sum(r['submissions'] for r in rows) else None,
            'mean_questions':sum(r['questions'] for r in rows)/len(rows),
            'mean_waits':sum(r['waits'] for r in rows)/len(rows),
            'no_responses':sum(r['no_responses'] for r in rows),'rejects':sum(r['rejects'] for r in rows),
            'clock_cost_s_per_condition':sum(r['clock_cost_s'] for r in rows)/len(rows),
            'base_rows':base,'risk_basis':'scene-macro base-episode point estimate; not a population guarantee'}


def paired_scene_interval(first, second, seed=10, replicates=5000):
    a=aggregate(first);b=aggregate(second)
    groups=sorted({r['group_id'] for r in a['base_rows']})
    def mapping(rows):return {(r['group_id'],r['episode_id']):r for r in rows}
    aa,bb=mapping(a['base_rows']),mapping(b['base_rows'])
    if set(aa)!=set(bb):raise ValueError('same paired scene/base episodes required')
    differences=[sum(aa[k]['resolution']-bb[k]['resolution'] for k in aa if k[0]==g)/sum(k[0]==g for k in aa) for g in groups]
    rng=random.Random(seed)
    boot=sorted(sum(rng.choice(differences) for _ in groups)/len(groups) for _ in range(replicates))
    return {'difference':sum(differences)/len(groups),'percentile_95':[boot[int(.025*(replicates-1))],boot[int(.975*(replicates-1))]],
            'resampling_unit':'paired scene; all episode/variant rows retained together','scene_groups':len(groups),
            'replicates':replicates,'inferential_claim_allowed':False,
            'warning':'Two development scenes are insufficient for a reliable population interval or significance claim'}


def sample_size_draft(delta=.1, alpha=.05, power=.8, risk=.1):
    z=NormalDist().inv_cdf(1-alpha/2)+NormalDist().inv_cdf(power)
    sensitivity={str(sd):math.ceil((z*sd/delta)**2) for sd in (.2,.25,.35,1.)}
    n_zero=math.ceil(math.log(alpha)/math.log(1-risk))
    return {'status':'DRAFT_ASSUMPTIONS_NOT_CAPACITY_CONFIRMED','effect_delta':delta,'two_sided_alpha':alpha,
            'power':power,'independent_unit':'scene/task cluster, not augmented condition or training seed',
            'normal_approximation_scene_counts_by_assumed_paired_sd':sensitivity,
            'assumed_paired_sd_not_estimated_from_two_dev_scenes':.25,
            'illustrative_effect_target_scenes':math.ceil(sensitivity['0.25']/.9),
            'zero_scene_errors_min_scenes_for_one_sided_95_upper_le_10_percent':n_zero,
            'risk_model':'independent scene-any-error Bernoulli bound; conservative for average episode risk; independence and sampling assumptions required',
            'currently_observed_zero_errors_8_episode_upper_if_independent':1-alpha**(1/8),
            'upper_bound_caveat':'illustration only: present episodes share only two scene groups',
            'formal_sample_count_locked':False}
