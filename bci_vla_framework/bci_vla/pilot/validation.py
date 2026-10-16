"""Check declared pilot records without inventing a real calibrated channel.

These tools validate supplied metadata, never prove device accuracy, label
truth, participant understanding, clock synchronization or physical events.
"""
from collections import Counter,defaultdict
from ..protocols import fields,integer,number,text,require,json_object

SPLITS={'calibration','calibration_holdout','pilot'}
MODES={'TASK_QUERY','CHANNEL_PROBE','CONFIRM'}
STATUSES={'OBSERVATION','NO_RESPONSE','INVALID_SIGNAL','NO_CONTROL','TRANSPORT_FAILURE','TIMEOUT'}


def sampling_segment(raw):
    v=json_object(raw)
    fields(v,{'sample_rate_hz','sample_start','sample_end','start_s','end_s','clock_id','channels','recording_ref'})
    number(v['sample_rate_hz'],1);integer(v['sample_start'],0);integer(v['sample_end'],1)
    require(v['sample_end']>v['sample_start'],'empty_sampling_window')
    for k in ['start_s','end_s']:number(v[k],0)
    require(v['end_s']>v['start_s'],'invalid_sampling_time')
    for k in ['clock_id','recording_ref']:text(v[k])
    require(isinstance(v['channels'],list) and bool(v['channels']) and len(set(v['channels']))==len(v['channels']),'invalid_channels')
    for channel in v['channels']:text(channel)
    expected=(v['sample_end']-v['sample_start'])/v['sample_rate_hz']
    require(abs(expected-(v['end_s']-v['start_s']))<=1/v['sample_rate_hz']+1e-6,'sample_count_time_rate_mismatch')
    return v


def validate_sampling_chain(raw,processed,transform):
    a,b=sampling_segment(raw),sampling_segment(processed);t=json_object(transform)
    fields(t,{'operation','input_rate_hz','output_rate_hz','anti_alias_evidence_ref','trace_ref'})
    require(t['operation'] in {'identity','resample'},'unsupported_sampling_operation')
    require(t['input_rate_hz']==a['sample_rate_hz'] and t['output_rate_hz']==b['sample_rate_hz'],'transform_rate_mismatch')
    text(t['trace_ref'])
    require(a['clock_id']==b['clock_id'],'cross_clock_requires_prior_alignment')
    require(a['channels']==b['channels'],'channel_order_changed_without_mapping')
    tolerance=max(1/a['sample_rate_hz'],1/b['sample_rate_hz'])+1e-6
    require(abs(a['start_s']-b['start_s'])<=tolerance and abs(a['end_s']-b['end_s'])<=tolerance,'processed_window_not_same_source_interval')
    if t['operation']=='identity':
        require(a['sample_rate_hz']==b['sample_rate_hz'] and a['sample_end']-a['sample_start']==b['sample_end']-b['sample_start'],'identity_changed_sampling')
        require(t['anti_alias_evidence_ref'] is None,'identity_has_unexpected_filter_evidence')
    else:text(t['anti_alias_evidence_ref'])
    return {'scope':'metadata_consistency_only_not_I01_resolved','raw_samples':a['sample_end']-a['sample_start'],
        'processed_samples':b['sample_end']-b['sample_start'],'duration_s':a['end_s']-a['start_s'],
        'raw_rate_hz':a['sample_rate_hz'],'processed_rate_hz':b['sample_rate_hz'],'declared_transform_consistent':True,
        'raw_signal_verified':False,'filter_quality_verified':False}


def validate_record(raw,*,allow_synthetic=False):
    v=json_object(raw)
    fields(v,{'record_id','participant_alias','session_group','recording_group','split','mode','source_kind',
        'label_source','label_validity','intended_symbol','probe_prompt','compliance_annotation',
        'response_status','observed_symbol','window','source_ref','stimulus_context','calibration_condition_ref'})
    for k in ['record_id','participant_alias','session_group','recording_group','source_ref']:text(v[k])
    fields(v['stimulus_context'],{'layout_id','mapping_id','stimulus_version','decoder_version'})
    for value in v['stimulus_context'].values():text(value)
    text(v['calibration_condition_ref'])
    require(v['split'] in SPLITS,'formal_test_not_calibration_or_pilot')
    require(v['mode'] in MODES and v['response_status'] in STATUSES,'invalid_pilot_mode_or_status')
    require(v['source_kind'] in {'live_bci','real_eeg_replay','synthetic_fixture'},'invalid_evidence_source')
    require(v['source_kind']!='synthetic_fixture' or allow_synthetic,'synthetic_not_real_pilot')
    require(v['label_source'] in {'independent_task_card','independent_probe_annotation','independent_confirmation_annotation'},'independent_label_required')
    expected={'TASK_QUERY':'independent_task_card','CHANNEL_PROBE':'independent_probe_annotation','CONFIRM':'independent_confirmation_annotation'}
    require(v['label_source']==expected[v['mode']],'label_mode_mismatch')
    require(v['label_validity'] in {'known','unknown'},'invalid_label_validity')
    require(v['intended_symbol'] in {'0','1',None},'invalid_intended_symbol')
    require((v['label_validity']=='known')==(v['intended_symbol'] is not None),'unknown_label_cannot_supply_intended_symbol')
    require(v['observed_symbol'] in {'0','1',None},'invalid_observed_symbol')
    require((v['response_status']=='OBSERVATION')==(v['observed_symbol'] is not None),'status_symbol_mismatch')
    if v['mode']=='CHANNEL_PROBE':
        require(v['probe_prompt'] in {'0','1'},'probe_prompt_required')
        require(v['compliance_annotation'] in {'0','1','NO_RESPONSE','UNKNOWN'},'independent_compliance_or_unknown_required')
        if v['label_validity']=='known':require(v['intended_symbol']==v['compliance_annotation'],'probe_intended_must_match_independent_annotation')
    else:require(v['probe_prompt'] is None and v['compliance_annotation'] is None,'probe_fields_outside_probe')
    if v['window'] is None:
        require(v['response_status'] in {'INVALID_SIGNAL','NO_CONTROL','TRANSPORT_FAILURE','TIMEOUT'},'measured_window_required_for_observation_or_omission')
    else:sampling_segment(v['window'])
    return v


def summarize_records(records,*,allow_synthetic=False):
    checked=[validate_record(row,allow_synthetic=allow_synthetic) for row in records]
    seen=set();groups={};ranges=defaultdict(list)
    intended=defaultdict(Counter);effective_probe=defaultdict(Counter);statuses=Counter();unknown=0
    for row in checked:
        require(row['record_id'] not in seen,'duplicate_pilot_record');seen.add(row['record_id'])
        group=row['recording_group'];previous=groups.setdefault(group,row['split'])
        require(previous==row['split'],'recording_group_split_leakage')
        statuses[row['response_status']]+=1
        if row['label_validity']=='unknown':unknown+=1
        w=row['window']
        if w is None:continue
        a,b=w['sample_start'],w['sample_end'];key=w['recording_ref']
        for low,high in ranges[key]:require(b<=low or a>=high,'overlapping_recording_samples')
        ranges[key].append((a,b))
        output=row['observed_symbol'] if row['response_status']=='OBSERVATION' else 'NO_RESPONSE' if row['response_status']=='NO_RESPONSE' else None
        if output is None:continue
        scope=(row['participant_alias'],row['session_group'],row['split'],row['mode'],tuple(sorted(row['stimulus_context'].items())),
            row['calibration_condition_ref'],w['sample_rate_hz'],tuple(w['channels']),
            (w['sample_end']-w['sample_start'])/w['sample_rate_hz'])
        if row['label_validity']=='known':intended[(*scope,row['intended_symbol'])][output]+=1
        if row['mode']=='CHANNEL_PROBE':effective_probe[(*scope,row['probe_prompt'])][output]+=1
    def rows(table):
        return [{'scope':list(key[:-1]),'intended_or_prompt':key[-1],'counts':{y:counts[y] for y in ['0','1','NO_RESPONSE']},
            'n':sum(counts.values())} for key,counts in sorted(table.items(),key=lambda x:str(x[0]))]
    synthetic=any(row['source_kind']=='synthetic_fixture' for row in checked)
    return {'scope':'synthetic_software_check' if synthetic else 'offline_pilot_record_summary' if checked else 'not_collected',
        'records':len(checked),'real_records':sum(row['source_kind']!='synthetic_fixture' for row in checked),
        'status_counts':dict(statuses),'unknown_intent_labels':unknown,
        'records_without_sampling_window':sum(row['window'] is None for row in checked),'conditional_intended_counts':rows(intended),
        'effective_probe_counts':rows(effective_probe),'real_channel_model_fitted':False,
        'decomposition_C_probe_and_compliance_identified':False,'transfer_validated':False,'G3_passed':False,
        'note':'effective probe counts do not identify decoder confusion and compliance separately; no automatic deployment approval'}
