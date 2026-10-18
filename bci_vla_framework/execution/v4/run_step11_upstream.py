"""Source-only isolated remote developer probe; never installs or dispatches."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zipfile

WORKSPACE=Path(__file__).resolve().parents[3];ROOT=WORKSPACE/'bci_vla_framework';OUT=ROOT/'execution/v4'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt',type=int)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--execute-prepared',action='store_true');args=parser.parse_args()
    assert args.attempt>0
    target=OUT/f'11_upstream_probe_attempt{args.attempt}.json';assert not target.exists()
    job=f'/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step11_upstream_probe_20261018_v{args.attempt}'
    files=[OUT/'step11_upstream_probe.py',ROOT/'bci_vla/__init__.py',ROOT/'configs/v4/upstream_subtask_source.json']
    for package in ('neural_interaction_v4','prefix_integration_v4','high_level_v4','interaction_contract_v4','data_v4','training_v4'):
        files.extend((ROOT/'bci_vla'/package).glob('*.py'))
    files.extend(ROOT/'configs/v4'/name for name in ('interfaces.json','interfaces_data_4_1.json'))
    files.extend(ROOT/'third_party/openpi_with_subtask'/name for name in ('src/openpi/models/pi0.py','src/openpi/models/tokenizer.py','LICENSE','LICENSE_GEMMA.txt'))
    manifest={'remote_job':job,'scope':'source/schema only; no local data or credentials uploaded',
              'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(set(files))]}
    mp=OUT/f'11_code_manifest{args.attempt}.json'
    archive=OUT/f'11_code_bundle{args.attempt}.zip'
    if args.prepare_only:
        mp.write_text(json.dumps(manifest,indent=2)+'\n')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(files)):z.write(p,p.relative_to(ROOT).as_posix())
            z.write(mp,'code_manifest.json')
    else:
        assert manifest==json.loads(mp.read_text()),'Prepared source changed; review a fresh bundle first'
        with zipfile.ZipFile(archive) as z:
            assert json.loads(z.read('code_manifest.json'))==manifest
            assert all(hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'] for r in manifest['files'])
    if args.prepare_only:
        print(json.dumps({'prepared_only':True,'files':len(manifest['files']),'archive_bytes':archive.stat().st_size,
                          'archive_sha256':sha(archive),'destination':job,'manifest':str(mp)},indent=2))
        return
    spec=importlib.util.spec_from_file_location('reader',WORKSPACE/'.server-inspect/remote_readonly.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);client=module.connect()
    try:
        _,stdout,stderr=client.exec_command('nvidia-smi --id=7 --query-gpu=memory.free,memory.total --format=csv,noheader,nounits',timeout=20)
        inventory=stdout.read().decode();error=stderr.read().decode();assert stdout.channel.recv_exit_status()==0,error
        print('GPU7_MB='+inventory.strip(),flush=True);assert int(inventory.split(',')[0])>27000
        ftp=client.open_sftp();ftp.mkdir(job);ftp.put(str(archive),job+'/code.zip');ftp.close()
        python='/home/oyjt/openpi/.venv/bin/python'
        env='env CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1 PYTHONPATH='+job+':/home/oyjt/Scene-aware-BCI-VLA-runs/step15C_full_20261016_v1/deps:/home/oyjt/openpi/src '
        command='cd '+shlex.quote(job)+' && '+python+' -m zipfile -e code.zip . && timeout --kill-after=10 600 '+env+python+' -u execution/v4/step11_upstream_probe.py --output '+shlex.quote(job+'/results')+' 2>stderr.log'
        _,stdout,stderr=client.exec_command(command,timeout=630)
        with (OUT/f'11_upstream_probe_attempt{args.attempt}.log').open('w',encoding='utf-8') as log:
            for line in stdout:
                log.write(line);log.flush()
                if line.startswith('STEP11_STAGE='):print(line.strip(),flush=True)
        error=stderr.read().decode(errors='replace');code=stdout.channel.recv_exit_status()
        ftp=client.open_sftp()
        try:ftp.get(job+'/results/result.json',str(target))
        except FileNotFoundError:target.write_text(json.dumps({'status':'PROCESS_FAILURE','stderr':error[-4000:]}))
        ftp.get(job+'/stderr.log',str(OUT/f'11_upstream_probe_stderr{args.attempt}.log'));ftp.close()
        result=json.loads(target.read_text());result.update(remote_job=job,exit_code=code)
        target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'summary':result.get('summary'),'error':result.get('error')}),flush=True)
    finally:client.close()

if __name__=='__main__':main()
