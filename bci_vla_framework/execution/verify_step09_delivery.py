"""Delivery consistency, not a research or deployment gate."""
from pathlib import Path
import hashlib
import json
import math

root = Path(__file__).resolve().parents[1]
def load(p): return json.loads((root / p).read_text(encoding='utf-8-sig'))
checks = []
def check(name, condition):
    if not condition: raise AssertionError(name)
    checks.append(name)

p = load('execution/progress.json'); t = load('execution/09_test_result.json')
trace = load('execution/09_policy_trace.json'); config = load('configs/diagnosis.json')
check('01-09 remain complete', all(s['status']=='已完成' for s in p['steps'][:9]))
check('progress may advance beyond 09', int(p['current_step'])>=10)
check('G1 conditional G2-G4 unstarted', p['research_gates']['G1']['scope'] == 'conditional_theory_and_protocol_design' and
      all(p['research_gates'][g]['status'] == '未开始' for g in ['G2', 'G3', 'G4']))
check('126 tests pass', t['tests_run'] == 126 and t['result'] == 'PASS' and t['failures'] == t['errors'] == t['skipped'] == 0)
check('simulation-only declaration', config['status'] == 'simulation_only' and config['deployment_allowed'] is False and
      p['steps'][8]['execution_scope'] == 'offline_illustrative_diagnostic_policy')
check('same one/two-step fixture outcome no superiority',
      trace['one_step']['selected_action']['action_id'] == trace['two_step']['selected_action']['action_id'] == 'probe:0:1.0' and
      math.isclose(trace['one_step']['selected_action']['expected_loss'], trace['two_step']['selected_action']['expected_loss']))
check('each root candidate has reasons cost branches', all({'reason', 'expected_loss', 'branches', 'elapsed_s'} <= set(x)
      for x in trace['one_step']['candidates']))
check('no authorization and no hardware/cloud', all(trace[k]['execution_allowed'] is False for k in
      ['one_step', 'two_step', 'no_budget', 'all_explanations_pause', 'stop']) and trace['hardware_calls'] == trace['cloud_calls'] == 0)
check('pause stop no-budget explicit', trace['no_budget']['selected_action']['kind'] == 'PAUSE' and
      trace['all_explanations_pause']['selected_action']['kind'] == 'PAUSE' and trace['stop']['reason'] == 'stop_latched')
next_goal = next(g for g, probability in trace['channel_handoff']['new_goal_prior'].items() if probability == 1)
check('explicit cross-task fresh goal prior', trace['channel_handoff']['new_contracts_required'] is True and
      all(x['probability'] == 0 for x in trace['channel_handoff']['next_joint_prior']['mass'] if x['goal_id'] != next_goal))
check('all 09.1-09.10 mapped', all(f'09.{i}' in (root / 'execution/09_acceptance.txt').read_text(encoding='utf-8-sig') for i in range(1,11)))
check('09 frozen sources match', all(hashlib.sha256(Path(f['path']).read_bytes()).hexdigest() == f['sha256'] for f in load('execution/09_source_manifest.json')['files']))
check('08 frozen sources match', all(hashlib.sha256(Path(f['path']).read_bytes()).hexdigest() == f['sha256'] for f in load('execution/08_source_manifest.json')['files']))
check('TXT BOM CRLF intact', all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n', b'') for q in root.rglob('*.txt')))
v = {'date': '2026-10-16', 'scope': 'offline_diagnostic_policy_delivery', 'result': 'PASS',
     'count': len(checks), 'checks': checks,
     'not_evaluated': ['real calibration and user time', 'strategy superiority', 'within-window change kernel',
                       'multi-task anticipatory value', 'robot execution', 'G2-G4']}
(root / 'execution/09_integrity_recheck_after_step10.json').write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(v, ensure_ascii=True, indent=2))
