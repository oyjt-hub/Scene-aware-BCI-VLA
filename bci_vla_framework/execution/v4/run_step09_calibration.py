"""Code-only upload; calibration stays on the host of the captured LIBERO data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zipfile

WORKSPACE = Path(__file__).resolve().parents[3]
PROJECT = WORKSPACE / 'bci_vla_framework'
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt', type=int)
    parser.add_argument('--binding-attempt', type=int, required=True)
    args = parser.parse_args()
    target = OUT / f'09_calibration_attempt{args.attempt}.json'
    if target.exists():
        raise FileExistsError('Preserve previous calibration evidence')
    job = f'/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step09_calibration_20261018_v{args.attempt}'
    binding_job = f'/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step09_binding_20261018_v{args.binding_attempt}'
    capture = '/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step04_libero_data_20261017_v1'
    files = [OUT/'step09_calibrate_and_policy.py', OUT/'build_step04_data.py', PROJECT/'bci_vla/__init__.py']
    for package in ('calibration_v4', 'neural_interaction_v4', 'interaction_contract_v4', 'data_v4', 'training_v4', 'evaluation_v4'):
        files.extend((PROJECT/'bci_vla'/package).glob('*.py'))
    files.extend(PROJECT/f'configs/v4/{name}.json' for name in ('interfaces', 'interfaces_data_4_1', 'calibration_and_policy', 'calibration_and_policy_template'))
    manifest = {'scope': 'source/schema/config ONLY; no data, images, labels or credentials',
                'remote_job': job, 'binding_job': binding_job,
                'files': [{'path': p.relative_to(PROJECT).as_posix(), 'sha256': digest(p)} for p in sorted(set(files))]}
    mp = OUT/f'09_calibration_code_manifest{args.attempt}.json'
    mp.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    archive = OUT/f'09_calibration_code_bundle{args.attempt}.zip'
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
        python = '/home/oyjt/openpi/.venv/bin/python'
        env = ('env OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu '
               'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH='+job+':/home/oyjt/openpi/src ')
        run = (env+python+' -u execution/v4/step09_calibrate_and_policy.py --binding-job '+shlex.quote(binding_job)
               +' --capture-root '+shlex.quote(capture)+' --output '+shlex.quote(job+'/results')
               +' --config configs/v4/calibration_and_policy_template.json')
        command = ('cd '+shlex.quote(job)+' && '+python+' -m zipfile -e code.zip . && '
                   'timeout --kill-after=15 2400 '+run+' 2>calibration_stderr.log')
        _, stdout, stderr = client.exec_command(command, timeout=2450)
        with (OUT/f'09_calibration_attempt{args.attempt}.log').open('w', encoding='utf-8') as log:
            for line in stdout:
                log.write(line); log.flush()
                if line.startswith(('CALIBRATION_STAGE=', 'CALIBRATION_RESULT=')):
                    print(line.strip(), flush=True)
        error = stderr.read().decode(errors='replace')
        code = stdout.channel.recv_exit_status()
        sftp = client.open_sftp()
        try:
            sftp.get(job+'/results/calibration_result.json', str(target))
            result = json.loads(target.read_text())
        except FileNotFoundError:
            result = {'status': 'PROCESS_ERROR_OR_TIMEOUT', 'step09_complete': False}
        result.update(remote_job=job, remote_exit_code=code, stderr=error[-10000:])
        for name in ('calibration_stderr.log',):
            try:
                sftp.get(job+'/'+name, str(OUT/f'09_calibration_stderr_attempt{args.attempt}.log'))
            except FileNotFoundError:
                pass
        # Download the complete per-condition evidence, not only aggregate scores.
        collect = ('cd '+shlex.quote(job)+' && '+python+' -m zipfile -c results.zip results')
        _, out2, err2 = client.exec_command(collect, timeout=120)
        out2.read(); collect_error = err2.read().decode(errors='replace')
        if out2.channel.recv_exit_status() != 0:
            raise RuntimeError('Result archive failed: '+collect_error[-1000:])
        local_zip = OUT/f'09_calibration_results{args.attempt}.zip'
        sftp.get(job+'/results.zip', str(local_zip))
        sftp.close()
        result['local_evidence_archive'] = str(local_zip)
        result['local_evidence_archive_sha256'] = digest(local_zip)
        target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({k: result.get(k) for k in ('status', 'step09_complete', 'remote_exit_code')}), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
