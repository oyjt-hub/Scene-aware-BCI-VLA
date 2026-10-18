"""Upload code/schema only. Existing public LIBERO fixtures remain on server."""
import importlib.util,json,zipfile,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];PROJECT=ROOT/'bci_vla_framework';OUT=Path(__file__).resolve().parent
def main():
 attempt=int(sys.argv[1]) if len(sys.argv)>1 else 1;target=OUT/f'07_real_attempt{attempt}.json';assert not target.exists()
 job=f'/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step07_prefix_20261017_v{attempt}'
 files=[OUT/'step07_real_integration.py',PROJECT/'bci_vla/__init__.py']
 for folder in ['neural_interaction_v4','prefix_integration_v4','high_level_v4','interaction_contract_v4','data_v4']:files.extend((PROJECT/'bci_vla'/folder).glob('*.py'))
 files+=[PROJECT/'configs/v4/interfaces.json',PROJECT/'configs/v4/interfaces_data_4_1.json']
 manifest={'scope':'code/schema only; no local data/labels','files':[{'path':p.relative_to(PROJECT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(files))]}
 mp=OUT/f'07_real_code_manifest{attempt}.json';mp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');archive=OUT/'07_code_bundle.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(set(files)):z.write(p,p.relative_to(PROJECT).as_posix())
  z.write(mp,'code_manifest.json')
 spec=importlib.util.spec_from_file_location('remote_reader',ROOT/'.server-inspect/remote_readonly.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);c=mod.connect()
 try:
  ftp=c.open_sftp();ftp.mkdir(job);ftp.put(str(archive),job+'/code.zip');ftp.close()
  cmd='cd '+job+' && /home/oyjt/openpi/.venv/bin/python -m zipfile -e code.zip . && timeout --kill-after=10 600 env CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1 PYTHONPATH='+job+':/home/oyjt/Scene-aware-BCI-VLA-runs/step15C_full_20261016_v1/deps:/home/oyjt/openpi/src /home/oyjt/openpi/.venv/bin/python -u execution/v4/step07_real_integration.py'
  _,stdout,stderr=c.exec_command(cmd,timeout=620)
  with (OUT/f'07_real_attempt{attempt}.log').open('w',encoding='utf-8') as log:
   for line in stdout:
    log.write(line);log.flush()
    if line.startswith(('CHECK=','INTEGRATION_RESULT=')):print(line.strip(),flush=True)
  err=stderr.read().decode(errors='replace');code=stdout.channel.recv_exit_status();ftp=c.open_sftp()
  try:ftp.get(job+'/integration_result.json',str(target));r=json.loads(target.read_text(encoding='utf-8'))
  except FileNotFoundError:r={'status':'PROCESS_ERROR_OR_TIMEOUT'}
  ftp.close();r.update(remote_job=job,remote_exit_code=code,stderr=err[-12000:]);target.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:r.get(k) for k in ['status','checks_passed','error']}))
 finally:c.close()
if __name__=='__main__':main()
