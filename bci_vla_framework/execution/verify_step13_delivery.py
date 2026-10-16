"""Read-only evidence checks; never connects to devices or starts an executor."""
from pathlib import Path
import hashlib
import json
import struct
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.pilot.neuracle_stream import FloatFrameParser
from bci_vla.monitoring.journal import EventJournal
from bci_vla.contracts.session import replay_session_state
from bci_vla.adapters.vla import FakeVLAAdapter
from bci_vla.protocols import thaw

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)
def hashes(p):return all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in read(p)['files'])

run=ROOT/'runs/pilot_virtual_13_nonreal_acceptance_v3'
summary=read(run/'summary.json');flow=read(run/'virtual_flow.json');coverage=read('execution/13_coverage.json')
tests=read('execution/13_test_result.json');p=read('execution/progress.json')
check('new delivery sources and evidence frozen',hashes('execution/13_source_manifest.json'))
check('01-12R manifest payloads unchanged',all(hashes(f'execution/{v}_source_manifest.json') for v in ['09','10','11','12','12R']))
check('full regression passes including new tests',tests['result']=='PASS' and tests['tests_run']==365 and tests['failures']==tests['errors']==tests['skipped']==0)
check('latest scope excludes real BCI and downstream commands',summary['scope']=='non_real_BCI_software_only' and summary['hardware_calls']==summary['cloud_calls']==summary['robot_commands']==summary['real_pilot_trials']==0)
check('all three mock TCP cases pass',len(summary['tcp_cases'])==3 and all(r['passed'] for r in summary['tcp_cases']))
for case in summary['tcp_cases']:
    report=read(case['report_ref']);folder=(ROOT/case['report_ref']).parent
    raw=(folder/'raw_stream.bin').read_bytes()
    check('mock provenance '+case['case'],report['source_kind']=='synthetic_tcp_fixture' and report['port']!=8712 and report['recorder_connections_attempted']==report['application_bytes_sent']==0)
    check('mock bytes hash '+case['case'],hashlib.sha256(raw).hexdigest()==report['raw_sha256'])
    if case['case']=='fragmented':
        parser=FloatFrameParser(12);rows=parser.feed(raw)
        expected=struct.pack('<12000f',*(float(i%97) for i in range(12000)))
        check('mock actual bytes and parsed frames match independent fixture',raw==expected and len(rows)==1000 and not parser.tail and parser.nonfinite==0)
    else:check('failure cases retain zero samples without selection '+case['case'],not raw and report['decoded_selection'] is None)
check('three presentation modes in order',flow['modes']==['TASK_QUERY','CHANNEL_PROBE','CONFIRM'])
check('probe has no contract or execution authority',flow['views'][1]['contract'] is None and all(not v['execution_authority'] for v in flow['views']))
check('one fake execution completes independently',flow['fake_backend_submissions']==1 and flow['final_state']=='SUCCEEDED' and flow['replay_matches_each_checkpoint'])
journal=EventJournal.from_jsonl((run/'session_journal.jsonl').read_text(encoding='utf-8'))
with patch.object(FakeVLAAdapter,'__init__',side_effect=AssertionError('no backend')), \
     patch.object(FakeVLAAdapter,'submit',side_effect=AssertionError('no submit')), \
     patch.object(FakeVLAAdapter,'query',side_effect=AssertionError('no query')), \
     patch.object(FakeVLAAdapter,'stop',side_effect=AssertionError('no stop')):
    replay=replay_session_state(journal)
check('stored journal independently replays all checkpoints without backend',replay==thaw(journal.records[-1]['payload']['state']) and replay['state']=='SUCCEEDED')
check('frame timing distinguishes injected drop',summary['frame_timing']['ideal']['long_intervals']==0 and summary['frame_timing']['injected_drop']['long_intervals']==1 and not summary['actual_display_timing_measured'])
records=read(run/'synthetic_record_summary.json')
check('synthetic records never counted as real calibration',records['records']==5 and records['real_records']==0 and records['scope']=='synthetic_software_check' and not records['real_channel_model_fitted'])
check('unknown label and transport timeout retained separately',records['unknown_intent_labels']==1 and records['status_counts']['TIMEOUT']==1 and records['status_counts']['NO_RESPONSE']==1 and not records['decomposition_C_probe_and_compliance_identified'])
timing=read(run/'synthetic_timing.json')
check('complete and missing timing remain distinguishable',timing['complete']['decomposition_complete'] and timing['missing_phase_case']['user_busy_s'] is None and timing['real_time_measurements']==0)
prior=read('runs/pilot_virtual_13_streamcheck_01/stream_report.json')
check('earlier unlabeled receive-only evidence separately preserved',prior['complete_frames']==3080 and prior['pilot_trials']==0 and prior['application_bytes_sent']==0)
sampling=read('runs/pilot_virtual_13_samplingcheck_01/sampling_report.json')
check('prior raw processing report retains unvalidated real decoder scope',not sampling.get('real_decoder_accuracy_validated',False))
check('scope and progress do not mark whole step13 complete',coverage['nonreal_tests_completed'] and not coverage['step13_whole_completed'] and not coverage['real_pilot_completed'] and p['current_step'] in {'13','14','15','16'} and p['steps'][12]['status']=='进行中')
check('research gates not upgraded',p['research_gates']['G2']['research_acceptance_passed'] is False and p['research_gates']['G3']['status']=='未验证' and not coverage['G3_passed'])
check('real metrics remain missing',all(coverage[k] is None for k in ['decoder_accuracy','probe_compliance','probe_to_task_transfer','user_time_s']))
check('old repair revalidated without overwriting original result',read('execution/12R_integrity_result.json')['result']=='PASS' and read('execution/12R_integrity_recheck_after_step13.json')['result']=='PASS')
check('readiness explicitly records no additional real-device calls',read('execution/13_readiness_audit.json')['further_real_device_connections']==0)
check('static preview PNG present',(run/'ui_preview.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
check('all TXT preserve BOM CRLF',all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n',b'') for q in ROOT.rglob('*.txt')))
value={'date':'2026-10-16','result':'PASS','scope':'step13_nonreal_delivery_only','count':len(checks),'checks':checks,
    'tests_run':tests['tests_run'],'hardware_calls':0,'cloud_calls':0,'robot_commands':0,'real_pilot_trials':0,'G3_passed':False}
destination='execution/13_integrity_recheck_after_step'+p['current_step']+'.json' if p['current_step'] in {'14','15','16'} else 'execution/13_integrity_result.json'
if p['steps'][13].get('nonreal_closed_loop_completed'):
    destination='execution/13_integrity_recheck_after_step14C.json'
if p['steps'][14].get('nonreal_migration_completed'):
    destination='execution/13_integrity_recheck_after_step15C.json'
if p['current_step']=='16':
    destination='execution/13_integrity_recheck_after_step16.json'
(ROOT/destination).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value,ensure_ascii=True,indent=2))
