"""Predeclared validation-only candidate selection, never formal-test tuning."""
import hashlib
import json

from ..protocols import fields, number, require


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def select_validation(records,*,partitions,rule,expected_candidates,simulation_only=False):
    fields(partitions,{'development','calibration','validation','test'})
    sets={k:set(v) for k,v in partitions.items()}
    require(all(len(sets[k])==len(partitions[k]) for k in sets),'duplicate_partition_group')
    names=list(sets)
    require(all(not sets[a]&sets[b] for i,a in enumerate(names) for b in names[i+1:]),'partition_group_leakage')
    require(rule['status']=='predeclared' and rule['source_split']=='calibration','selection_rule_not_frozen_from_pilot')
    require(bool(rule['source_ref']),'pilot_rule_evidence_required')
    number(rule['wrong_start_rate_ceiling'],0,1); number(rule['all_episode_mean_time_ceiling_s'],0)
    require(bool(expected_candidates),'candidate_grid_required')
    require(len(expected_candidates)==len(set(expected_candidates)),'duplicate_candidate_grid')
    require({r['candidate_id'] for r in records}==set(expected_candidates) and len(records)==len(expected_candidates),'candidate_grid_incomplete')
    eligible=[]; common_ids=None
    for r in records:
        fields(r,{'candidate_id','method','config_sha256','split','provenance','group_ids','episode_ids',
                  'episodes','faithful_completions','wrong_starts','abstentions','total_all_episode_time_s','common_protocol_checks_pass'})
        require(r['method'] in {'B'+str(i) for i in range(11)}|{'B1P'},'not_a_baseline_candidate')
        require(r['split']=='validation','formal_test_cannot_select_baseline')
        require(r['provenance']=='independent_measurement' or (simulation_only and r['provenance']=='simulation_fixture'),'real_validation_required')
        require(set(r['group_ids'])<=sets['validation'] and bool(r['group_ids']),'validation_group_mismatch')
        require(type(r['common_protocol_checks_pass']) is bool,'invalid_protocol_check_flag')
        for k in ('episodes','faithful_completions','wrong_starts','abstentions'):
            require(type(r[k]) is int and 0<=r[k]<=r['episodes'],'invalid_episode_counts')
        require(r['episodes']>0 and len(r['episode_ids'])==len(set(r['episode_ids']))==r['episodes'],'episode_denominator_mismatch')
        require(len(r['config_sha256'])==64 and all(c in '0123456789abcdef' for c in r['config_sha256']),'config_fingerprint_required')
        ids=set(r['episode_ids'])
        if common_ids is None: common_ids=ids
        require(ids==common_ids,'unpaired_validation_episodes')
        number(r['total_all_episode_time_s'],0)
        n=r['episodes']; wrong=r['wrong_starts']/n; time=r['total_all_episode_time_s']/n
        if r['common_protocol_checks_pass'] and wrong<=rule['wrong_start_rate_ceiling'] and time<=rule['all_episode_mean_time_ceiling_s']:
            eligible.append(((-r['faithful_completions']/n,wrong,time,r['abstentions']/n,r['candidate_id']),r))
    require(bool(eligible),'no_eligible_baseline_do_not_relax_after_test')
    chosen=min(eligible,key=lambda x:x[0])[1]
    return {'status':'simulation_selection_check_only' if simulation_only else 'validation_choice_pending_deployment_audit',
        'chosen_method':chosen['method'],'chosen_candidate':chosen['candidate_id'],
        'chosen_config_sha256':chosen['config_sha256'],'rule_sha256':digest(rule),'records_sha256':digest(records),
        'partitions_sha256':digest(partitions),'deployment_allowed':False}
