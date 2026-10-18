"""Source-only upload for validation-pair diagnostics of selected checkpoints."""
import argparse, hashlib, importlib.util, json, shlex, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; WORKSPACE=ROOT.parent; OUT=ROOT/'execution/v4'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--training-attempt',type=int,default=3)
    args=parser.parse_args()
    target=OUT/f'08_challenges_attempt{args.attempt}.json'
    assert not target.exists(),'Preserve earlier diagnostic results'
    job=f'/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step08_challenges_20261017_v{args.attempt}'
    training=f'/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step08_training_20261017_v{args.training_attempt}'
    files=[OUT/'step08_development_challenges.py',ROOT/'bci_vla/__init__.py']
    for folder in ('training_v4','neural_interaction_v4','prefix_integration_v4','high_level_v4','interaction_contract_v4','data_v4'):
        files.extend((ROOT/'bci_vla'/folder).glob('*.py'))
    files.extend(ROOT/f'configs/v4/{name}.json' for name in ('interfaces','interfaces_data_4_1'))
    manifest={'scope':'code/schema only; validation-only read of server data and selected checkpoints; no training',
        'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(files))]}
    path=OUT/f'08_challenge_code_manifest{args.attempt}.json'
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=OUT/f'08_challenge_code{args.attempt}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p,p.relative_to(ROOT).as_posix())
        z.write(path,'code_manifest.json')
    spec=importlib.util.spec_from_file_location('remote_reader',WORKSPACE/'.server-inspect/remote_readonly.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);client=module.connect()
    try:
        ftp=client.open_sftp();ftp.mkdir(job);ftp.put(str(archive),job+'/code.zip');ftp.close()
        python='/home/oyjt/openpi/.venv/bin/python'
        command='cd '+shlex.quote(job)+' && '+python+' -m zipfile -e code.zip . && timeout --kill-after=10 900 env CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1 PYTHONPATH='+job+':/home/oyjt/Scene-aware-BCI-VLA-runs/step15C_full_20261016_v1/deps:/home/oyjt/openpi/src '+python+' -u execution/v4/step08_development_challenges.py --training-job '+shlex.quote(training)+' --capture-root /home/oyjt/Scene-aware-BCI-VLA-runs/v4_step04_libero_data_20261017_v1 2>stderr.log'
        _,stdout,stderr=client.exec_command(command,timeout=920)
        with (OUT/f'08_challenges_attempt{args.attempt}.log').open('w') as log:
            for line in stdout:
                log.write(line);log.flush()
                if line.startswith(('CHALLENGE_STAGE=','CHALLENGE_RESULT=')):
                    print(line.strip(),flush=True)
        error=stderr.read().decode(errors='replace');code=stdout.channel.recv_exit_status()
        ftp=client.open_sftp()
        try:
            ftp.get(job+'/challenge_result.json',str(target));result=json.loads(target.read_text())
        except FileNotFoundError:
            result={'status':'PROCESS_ERROR_OR_TIMEOUT'}
        try:
            ftp.get(job+'/stderr.log',str(OUT/f'08_challenges_stderr{args.attempt}.log'))
        except FileNotFoundError:
            pass
        ftp.close();result.update(remote_job=job,remote_exit_code=code,stderr=error[-8000:])
        target.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result.get(k) for k in ('status','error','remote_exit_code')}),flush=True)
    finally:
        client.close()


if __name__=='__main__':
    main()
