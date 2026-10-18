"""Record scope before development training without promoting old gates."""
import datetime, hashlib, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'execution/v4'
progress_path = ROOT / 'execution/v4_plan_progress.json'
before = OUT / '08_progress_before.json'
if not before.exists():
    before.write_bytes(progress_path.read_bytes())
progress = json.loads(progress_path.read_text(encoding='utf-8-sig'))
progress['current_stage'] = 'V4-08_restricted_development_training_in_progress'
for step in progress['steps']:
    if step['step_id'] == 'V4-08':
        step['status'] = '进行中'
        step['started_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        step['execution_scope'] = 'Restricted explicit public declaration pilot; actual LIBERO images and simulated BCI; training/validation only'
        step['full_step_complete'] = False
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2)+'\n', encoding='utf-8-sig', newline='\r\n')
files = list((ROOT/'bci_vla/training_v4').glob('*.py'))
files += list((ROOT/'tests').glob('test_training_*_v4.py'))
files += [ROOT/'configs/v4/train_step08_pilot.json', ROOT/'configs/v4/train_data_step08.json', OUT/'step08_real_training.py', OUT/'run_step08_real.py']
audit = {'step': 'V4-08', 'scope': 'Development training pilot, not full research acceptance',
    'unit_tests': {'passed': 34, 'suites': ['training_data_v4', 'training_engine_v4', 'training_text_v4']},
    'source_files': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
    'git_status': subprocess.run(['git','status','--short'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8').stdout,
    'failed_attempt1': 'Full B2 prompt exceeded 3072 tokens before any optimizer update; preserved and expanded uniformly to 6144',
    'real_EEG': False, 'physical_robot': False, 'formal_test_accessed': False,
    'remaining_training_scope': ['Unresolved hidden-intent curriculum remains ambiguous', 'Placement observations/targets unavailable', 'Pilot seed0; seeds1/2 and final budget pending measured cost']}
(OUT/'08_prework_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status':'STEP08_IN_PROGRESS', 'unit_tests':34}))
