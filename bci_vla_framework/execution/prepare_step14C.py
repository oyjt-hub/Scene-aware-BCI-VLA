"""Lock functional cases and independent calibration split before new rollouts."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];path=ROOT/'execution/14C_run_plan.json';assert not path.exists()
assets=json.loads((ROOT/'execution/14_libero_assets.json').read_text())
cases=[dict(id='nominal_cream',goal=0,channel='normal',prior=[.81,.09,.09,.01]),
    dict(id='nominal_wine',goal=1,channel='normal',prior=[.09,.01,.81,.09]),
    dict(id='changed_goal',goal=1,channel='normal',prior=[.81,.09,.09,.01]),
    dict(id='changed_channel',goal=0,channel='flip',prior=[.81,.09,.09,.01]),
    dict(id='joint_alias_conflict',goal=1,channel='flip',prior=[.5,0,0,.5]),
    dict(id='late_confirmation',goal=0,channel='normal',prior=[.9,0,.1,0],late_confirmation=True),
    dict(id='STOP_during_execution',goal=0,channel='normal',prior=[.9,0,.1,0],stop=True),
    dict(id='goal_change_recovery',goal=0,channel='normal',prior=[.9,0,.1,0],goal_change_during_execution=True),
    dict(id='execution_failure_not_channel',goal=0,channel='normal',prior=[.9,0,.1,0],execution_failure=True),
    dict(id='inference_failure_not_channel',goal=0,channel='normal',prior=[.9,0,.1,0],transport_failure=True)]
value={'scope':'nonreal_BCI_full_functional_pipeline_not_strategy_superiority','checkpoint_sha256':assets['files'][0]['sha256'],
    'normalization_sha256':assets['files'][2]['sha256'],'scene':'LIBERO goal6 objects, independent pick tasks (not official task6 success)',
    'task_predicate':'gripper grasp contact AND lift>0.03m, stable3 simulation steps','calibration_init_indices':[0,1],
    'holdout_init_indices':[2],'test_init_indices':[3],'max_execution_steps':160,'Beta_prior':[1,1],
    'cost_scope':{'task_failure':'Beta-smoothed independent simulator endpoint','wrong_start':'command-vs-hypothesis identity',
        'recovery_s':'measured simulator reset wall time','user_time_s':'declared2s synthetic confirmation allowance, not human measurement'},
    'real_BCI_enabled':False,'physical_robot_enabled':False,'cases':cases,
    'old_source_manifest':'execution/15_source_manifest.json',
    'versioned_sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((ROOT/'bci_vla/simulation_v3').glob('*.py'))]}
path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print('14C functional plan frozen before actual model calibration.')
