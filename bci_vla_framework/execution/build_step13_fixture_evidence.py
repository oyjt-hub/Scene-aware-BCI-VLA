"""Exercise the collection-file CLI and time accounting with explicit fixtures."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tests.test_pilot_preparation import record
from bci_vla.models.evidence import summarize_timing,PHASES
folder=ROOT/'runs/pilot_virtual_13_nonreal_acceptance_v3'
source=folder/'synthetic_records.jsonl'
assert not source.exists()
rows=[record(),record(mode='CHANNEL_PROBE',label_source='independent_probe_annotation',probe_prompt='0',
    compliance_annotation='UNKNOWN',label_validity='unknown',intended_symbol=None),
    record(mode='CONFIRM',label_source='independent_confirmation_annotation'),
    record(response_status='NO_RESPONSE',observed_symbol=None),
    record(response_status='TIMEOUT',observed_symbol=None,window=None)]
for i,r in enumerate(rows):
    r['record_id']='synthetic-'+str(i)
    if r['window'] is not None:r['window']['recording_ref']='synthetic-recording-'+str(i)
source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
subprocess.run([sys.executable,'-B',str(ROOT/'execution/inspect_step13_records.py'),'--records',str(source),
    '--output',str(folder/'synthetic_record_summary.json'),'--software-fixture'],check=True)
segments=[{'phase':phase,'clock_id':'synthetic-clock','start_s':i,'end_s':i+1,
    'user_busy':phase not in {'decoding','network','computation'}} for i,phase in enumerate(sorted(PHASES))]
complete=summarize_timing(segments,required_phases=sorted(PHASES),clock_id='synthetic-clock',window_start_s=0,window_end_s=9)
missing=summarize_timing(segments[:-1],required_phases=sorted(PHASES),clock_id='synthetic-clock',window_start_s=0,window_end_s=9)
assert complete['decomposition_complete'] and missing['user_busy_s'] is None
(folder/'synthetic_timing.json').write_text(json.dumps({'scope':'synthetic_accounting_fixture_not_user_measurement',
    'segments':segments,'complete':complete,'missing_phase_case':missing,'real_time_measurements':0},indent=2)+'\n',encoding='utf-8')
