"""Collect only selected repair weights, verifying their recorded SHA256."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt', type=int)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[3]
    project = workspace / 'bci_vla_framework'
    evidence = project / 'execution/v4'
    report = json.loads((evidence / f'09_binding_attempt{args.attempt}.json').read_text())
    run = next(r for r in report['runs'] if r['method'] == 'Full' and r['seed'] == 0)
    checkpoint = next(c for c in run['checkpoints'] if c['step'] == run['selected_step'])
    expected_root = report['remote_job'] + '/results/Full_seed0/step' + str(run['selected_step'])
    if run['selected_checkpoint'] != expected_root or checkpoint['path'] != expected_root:
        raise ValueError('Unexpected remote checkpoint path')
    local = project / f'runs/neural_interaction_v4/dev/step09_attempt{args.attempt}/Full_seed0/step{run["selected_step"]}'
    local.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('remote_reader', workspace/'.server-inspect/remote_readonly.py')
    remote = importlib.util.module_from_spec(spec); spec.loader.exec_module(remote)
    client = remote.connect()
    entries = []
    try:
        sftp = client.open_sftp()
        for name in ('connector.safetensors', 'language_lora.safetensors'):
            dest = local / name
            expected = checkpoint['files'][name]
            if not dest.exists():
                partial = local / (name + '.partial')
                sftp.get(expected_root + '/' + name, str(partial))
                if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
                    raise ValueError('Downloaded weight hash mismatch: ' + name)
                partial.rename(dest)
            actual = hashlib.sha256(dest.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError('Existing local weight hash mismatch: ' + name)
            entries.append({'name': name, 'remote_path': expected_root+'/'+name,
                            'local_path': str(dest), 'bytes': dest.stat().st_size,
                            'sha256': actual, 'verified': True})
        sftp.close()
    finally:
        client.close()
    result = {'attempt': args.attempt, 'method': 'Full', 'seed': 0,
              'selected_step': run['selected_step'], 'files': entries,
              'all_verified': all(r['verified'] for r in entries),
              'scope': 'selected development weights; not formal model acceptance'}
    (evidence / f'09_checkpoint_manifest{args.attempt}.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'all_verified': result['all_verified'], 'files': len(entries),
                      'bytes': sum(r['bytes'] for r in entries)}))


if __name__ == '__main__':
    main()
