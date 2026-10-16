"""Freeze the transfer test inputs before accessing second-backend outcomes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'execution/15_pre_run_freeze.json';assert not target.exists()
paths=[]
for directory in ['bci_vla/core','bci_vla/runtime_v2','bci_vla/contracts','bci_vla/models']:
    paths.extend(ROOT.glob(directory+'/*.py'))
paths.extend(ROOT/p for p in ['configs/diagnosis.json','configs/execution/session.json','configs/comparisons/parameters.json',
    'configs/channels/simulation_binary_v1.json','configs/consequences/simulation_canonical_outcomes.json',
    'bci_vla/pilot/libero_gate.py','execution/step14_libero_worker.py'])
plan={'scope':'development_transfer_preparation_not_formal_holdout','created_on':'2026-10-16',
    'core_changes_allowed':False,'core_threshold_tuning_allowed':False,'real_BCI_enabled':False,'physical_robot_enabled':False,
    'backend_A':'existing StarVLA Qwen2.5-VL-GR00T LIBERO4in1/30000',
    'backend_B_checkpoint':'/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT',
    'backend_B_config':'pi05_libero','selection_reason':'existing local compatible simulation baseline, selected before new outcomes',
    'common_suite':'libero_goal','task_ids':[0,1],'seed':7,'init_state_index':0,'max_steps':300,'settling_steps':10,
    'replan_actions':8,'fault_schedule':{'probe_before_step':5,'STOP_before_step':12},
    'expected_episodes':3,'training_scope_equivalence_verified':False,'formal_2x2_transfer_claim_allowed':False,
    'calibration':'no real channel or consequence fitting; retain missing real calibration',
    'source_files':[{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(paths))]}
target.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'sources_frozen':len(paths),'core_changes_allowed':False,'formal_2x2_transfer_claim_allowed':False}))
