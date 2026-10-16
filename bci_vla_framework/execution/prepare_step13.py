"""Create pilot preparation templates without overwriting filled user files."""
from pathlib import Path
import json
import importlib.util
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.models.channel import ChannelModel
from bci_vla.protocols import ProtocolError

def write(path,v):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def template(path,v):
    if not (ROOT/path).exists():write(path,v)


def main():
    template('configs/pilot/step13_site.template.json',{
        'schema_version':'pilot-site-preparation/1.0','scope':'preparation_only','user_requested_step13':True,
        'executor_mode':'virtual_only','physical_execution_allowed':False,
        'device':{'model':None,'acquisition_program_path':None,'source_sampling_hz':None,'decoder_sampling_hz':None,
            'channel_order':None,'sampling_chain_evidence_ref':None},
        'display':{'stimulus_program_path':None,'refresh_hz':None,'frequencies_hz':None,'layout_id':None,
            'mapping_id':None,'stimulus_version':None,'frame_timing_evidence_ref':None},
        'session':{'participant_alias':None,'session_group':None,'clock_alignment_evidence_ref':None},
        'calibration':{'task_blocks_ref':None,'probe_blocks_ref':None,'confirm_blocks_ref':None,
            'independent_task_holdout_ref':None,'timing_records_ref':None},
        'real_profile_path':'configs/channels/real_ssvep_uncalibrated.template.json'})
    template('examples/pilot13/independent_blocks.template.json',{
        'document_kind':'unexecuted_collection_design','visibility':'evaluation_private_only','provenance':'not_collected',
        'ready_for_collection':False,'blocks':[
            {'block_id':'task-calibration','mode':'TASK_QUERY','split':'calibration','trial_count':None,'recording_group':None},
            {'block_id':'probe-calibration','mode':'CHANNEL_PROBE','split':'calibration','trial_count':None,'recording_group':None},
            {'block_id':'confirm-calibration','mode':'CONFIRM','split':'calibration','trial_count':None,'recording_group':None},
            {'block_id':'task-holdout','mode':'TASK_QUERY','split':'calibration_holdout','trial_count':None,'recording_group':None},
            {'block_id':'paired-pilot','mode':None,'split':'pilot','trial_count':None,'recording_group':None}],
        'participant_count':None,'duration_choices_s':None,'randomization_ref':None,'real_records':[]})
    template('examples/pilot13/sampling_chain.template.json',{'scope':'unverified_sampling_metadata',
        'raw_segment':None,'processed_segment':None,'transform':None,'raw_file_sha256':None,
        'preprocessing_code_sha256':None,'I01_resolved':False})
    template('examples/pilot13/timing.template.json',{'scope':'not_measured','clock_id':None,'window_start_s':None,'window_end_s':None,
        'required_phases':['reading','switching','acquisition','decoding','network','retry','return_to_task','confirmation','computation'],
        'segments':[],'zero_duration_phases_require_evidence':True,'observed_user_busy_s':None})
    template('examples/pilot13/record_fields.template.json',{'document_kind':'record_schema_outline_not_a_trial',
        'record_id':None,'participant_alias':None,'session_group':None,'recording_group':None,'split':None,'mode':None,
        'source_kind':None,'label_source':None,'label_validity':None,'intended_symbol':None,'probe_prompt':None,
        'compliance_annotation':None,'response_status':None,'observed_symbol':None,'window':None,'source_ref':None,
        'stimulus_context':None,'calibration_condition_ref':None})
    site=json.loads((ROOT/'configs/pilot/step13_site.template.json').read_text(encoding='utf-8-sig'))
    progress=json.loads((ROOT/'execution/progress.json').read_text(encoding='utf-8-sig'))
    profile=json.loads((ROOT/site['real_profile_path']).read_text(encoding='utf-8-sig'))
    engineering=site.get('engineering_checks',{})
    stream_report=None
    if engineering.get('stream_report'):
        stream_report=json.loads((ROOT/engineering['stream_report']).read_text(encoding='utf-8-sig'))
    try:ChannelModel(profile);model_error=None
    except ProtocolError as exc:model_error=str(exc)
    missing=[]
    for section in ['device','display','session','calibration']:
        for key,value in site[section].items():
            if value is None:missing.append(section+'.'+key)
    blockers=[]
    if not progress['research_gates']['G2'].get('research_acceptance_passed',False):blockers.append('G2_research_acceptance_not_passed')
    if missing:blockers.append('site_device_display_calibration_metadata_unverified')
    if model_error:blockers.append('real_channel_profile_unavailable:'+model_error)
    blockers+=['SSVEPAdapterShell_has_no_acquisition_or_stimulus_display',
        'joint_filter_and_authorization_accept_offline_replay_only_in_current_version',
        'I01_device_clock_channel_map_and_stimulus_alignment_unverified','no_real_pilot_measurements_collected_in_this_run',
        'real_BCI_collection_excluded_by_latest_user_instruction']
    report={'date':'2026-10-16','scope':'step13_nonreal_scope_readiness','user_request_recorded':True,
        'latest_user_scope':'all_tests_except_real_BCI','further_real_device_connections':0,
        'ready_for_real_pilot':False,'preparation_completed':True,'blockers':blockers,'missing_fields':missing,
        'real_profile_status':profile['status'],'loader_error':model_error,
        'python':sys.version.split()[0],'platform':sys.platform,
        'optional_packages_in_this_python':{name:importlib.util.find_spec(name) is not None for name in ['brainflow','pylsl','psychopy','pygame','mne','numpy','scipy']},
        'package_presence_is_not_device_readiness':True,'real_eeg_trials_collected':0,'real_timing_measurements':0,
        'prior_to_scope_change_receive_only_connections':0 if stream_report is None else stream_report['recorder_connections_attempted'],
        'hardware_calls_this_nonreal_phase':0,
        'receive_only_recorder_frames':0 if stream_report is None else stream_report['complete_frames'],
        'recorder_stream_received':stream_report is not None and stream_report['complete_frames']>0,
        'recorder_stream_is_not_labeled_pilot_data':True,'engineering_checks':engineering,
        'cloud_calls':0,'physical_execution_allowed':False,'G3_passed':False,
        'source_evidence':['docs/02_issues.txt','execution/06_asset_audit.json','bci_vla/adapters/bci/ssvep.py',
            'bci_vla/models/channel.py','bci_vla/core/joint_belief.py','bci_vla/contracts/records.py']}
    write('execution/13_readiness_audit.json',report)
    write('runs/pilot_virtual_13_preparation/manifest.json',{'scope':'preparation_only_not_pilot_data',
        'real_eeg_trials_collected':0,'real_timing_measurements':0,'participant_count':None,
        'decoder_accuracy':None,'probe_compliance':None,'probe_to_task_transfer':None,'user_time_s':None,
        'missing_reason':'not_collected_site_and_runtime_prerequisites_unmet','G3_passed':False})
    print(json.dumps({'preparation_completed':True,'ready_for_real_pilot':False,'blockers':blockers,'real_trials':0},indent=2))


if __name__=='__main__':main()
