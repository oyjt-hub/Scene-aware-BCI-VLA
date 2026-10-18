"""Upload only source/schema/config; rebuild data on its original host and train."""
import argparse, hashlib, importlib.util, json, shlex, sys, zipfile
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
PROJECT = WORKSPACE / 'bci_vla_framework'
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt', type=int)
    parser.add_argument('--gpu', type=int, default=7)
    args = parser.parse_args()
    target = OUT / f'08_real_attempt{args.attempt}.json'
    if target.exists():
        raise FileExistsError('Preserve previous training evidence')
    job = f'/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step08_training_20261017_v{args.attempt}'
    capture = '/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step04_libero_data_20261017_v1'
    files = [OUT/'step08_real_training.py', OUT/'build_step08_data.py', OUT/'build_step04_data.py', PROJECT/'bci_vla/__init__.py']
    for package in ('neural_interaction_v4', 'prefix_integration_v4', 'high_level_v4', 'interaction_contract_v4', 'data_v4', 'training_v4'):
        files.extend((PROJECT/'bci_vla'/package).glob('*.py'))
    files.extend(PROJECT/f'configs/v4/{name}.json' for name in ('interfaces', 'interfaces_data_4_1', 'train_step08_pilot', 'train_data_step08'))
    manifest = {'scope': 'source/schema/config ONLY; no local data, images, private labels or credentials',
        'remote_job': job, 'GPU': args.gpu,
        'files': [{'path': p.relative_to(PROJECT).as_posix(), 'sha256': digest(p)} for p in sorted(set(files))]}
    mp = OUT/f'08_code_manifest{args.attempt}.json'
    mp.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    archive = OUT/f'08_code_bundle{args.attempt}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p, p.relative_to(PROJECT).as_posix())
        z.write(mp, 'code_manifest.json')
    spec = importlib.util.spec_from_file_location('remote_reader', WORKSPACE/'.server-inspect/remote_readonly.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = module.connect()
    try:
        sftp = client.open_sftp()
        sftp.mkdir(job)
        sftp.put(str(archive), job+'/code.zip')
        sftp.close()
        env = ('env CUDA_VISIBLE_DEVICES='+str(args.gpu)+' OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
               'PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1 PYTHONPATH='+job+
               ':/home/oyjt/Scene-aware-BCI-VLA-runs/step15C_full_20261016_v1/deps:/home/oyjt/openpi/src ')
        python = '/home/oyjt/openpi/.venv/bin/python'
        build = env+python+' -u execution/v4/build_step08_data.py --capture-root '+shlex.quote(capture)+' --output-root '+shlex.quote(job+'/training_data')
        train = env+python+' -u execution/v4/step08_real_training.py --data '+shlex.quote(job+'/training_data')+' --capture-root '+shlex.quote(capture)+' --output '+shlex.quote(job+'/results')
        command = 'cd '+shlex.quote(job)+' && '+python+' -m zipfile -e code.zip . && '+build+' && timeout --kill-after=15 3600 '+train+' 2>training_stderr.log'
        _, stdout, stderr = client.exec_command(command, timeout=3650)
        with (OUT/f'08_real_attempt{args.attempt}.log').open('w', encoding='utf-8') as log:
            for line in stdout:
                log.write(line); log.flush()
                if line.startswith(('TRAIN_STAGE=', 'TRAINING_RESULT=')):
                    print(line.strip(), flush=True)
        error = stderr.read().decode(errors='replace')
        code = stdout.channel.recv_exit_status()
        sftp = client.open_sftp()
        try:
            sftp.get(job+'/results/training_result.json', str(target))
            result = json.loads(target.read_text())
        except FileNotFoundError:
            result = {'status': 'PROCESS_ERROR_OR_TIMEOUT'}
        result.update(remote_job=job, remote_exit_code=code, stderr=error[-10000:])
        for remote, local in [('training_data/manifest.json',f'08_data_manifest{args.attempt}.json'),
                              ('training_stderr.log',f'08_stderr_attempt{args.attempt}.log')]:
            try:
                sftp.get(job+'/'+remote, str(OUT/local))
            except FileNotFoundError:
                pass
        sftp.close()
        target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({k:result.get(k) for k in ('status','error','remote_exit_code')}), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
