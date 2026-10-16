"""Explicit identity remap of illustrative step07 rows; no parameter fitting."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT / p).read_text(encoding='utf-8-sig'))
def write(p, v):
    q = ROOT / p; q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    f = read('configs/belief/simulation_binary_filter.json')
    left, right = [g['goal_id'] for g in f['vocabulary']['goals'] if g['kind'] == 'task']
    table = deepcopy(read('configs/consequences/simulation_outcome_table.json'))
    table['model_id'] = 'simulation-canonical-outcomes-v1'
    table['goal_ids'] = [left, right]
    for row in table['rows']:
        row['goal_id'] = {'old': left, 'new': right}[row['goal_id']]
    write('configs/consequences/simulation_canonical_outcomes.json', table)
    write('configs/diagnosis.json', {
        'schema_version': 'diagnosis/1.0', 'status': 'simulation_only',
        'provenance': 'illustrative_not_fitted', 'deployment_allowed': False,
        'max_depth': 2, 'pause_loss': 1.0, 'exit_loss': 1.0,
        'weights': {'wrong_start': 1.0, 'task_failure': 1.0, 'recovery_s': .02, 'user_time_s': .02},
        'decision_by_goal': {left: 'KEEP_OLD', right: 'ACCEPT_NEW'},
        'context': f['current_context'],
        'actions': {'scope': {'mapping_id': 'binary-v1', 'stimulus_version': 'stimulus-v1',
            'layout_id': f['layout_id'], 'sample_rate_hz': 250},
            'durations_s': [1.0, 2.0], 'confirmation_duration_s': 1.0,
            'setup_s': 1.0, 'time_weight_per_s': .02,
            'goal_symbols': {left: '0', right: '1'}, 'accept_symbol': '0',
            'compliance_id': 'imperfect'},
        'accounting': {'terminal_confirmation_allowance_removed': True,
            'setup_phase': 'display_reading_before_window',
            'acquisition': 'state_constant_within_window',
            'return_and_compute_s': 0.0,
            'time_source': 'illustrative_not_measured',
            'pause_loss_source': 'one_uncompleted_task_failure_no_wrong_start',
            'goal_alias_map': {'old': left, 'new': right},
            'future_tasks': 'not_quantified_without_explicit_future_task_model'}})
    contracts = {}
    for goal in f['vocabulary']['goals'][:2]:
        contracts[goal['goal_id']] = {'contract_id': 'contract-' + goal['goal_id'],
            'task_id': 'fixture-task', 'revision': 1, 'scene_id': 'fixture-scene', 'scene_revision': 1,
            'intent': goal['intent'], 'status': 'proposed',
            'valid_until': '2026-10-15T00:01:00Z', 'confirmation_event_id': None}
    write('examples/diagnosis/state.json', {
        'belief': f['initial_joint'], 'scene_id': 'fixture-scene', 'scene_revision': 1,
        'task_id': 'fixture-task', 'task_revision': 1,
        'now_s': 1792022400.0, 'used_s': 0.0, 'remaining_s': 20.0,
        'scene_valid_until_s': 1792022460.0, 'contracts': contracts})
