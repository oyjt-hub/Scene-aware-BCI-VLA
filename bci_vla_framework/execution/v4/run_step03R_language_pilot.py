"""Isolated pilot sends two code files only; data are generated from server assets."""
import importlib.util,json,shlex,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];PROJECT=ROOT/'bci_vla_framework';OUT=Path(__file__).resolve().parent
def main():
 attempt=int(sys.argv[1]) if len(sys.argv)>1 else 1
 job=f'/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step03R_language_20261017_v{attempt}'
 target=OUT/f'03R_language_pilot_v{attempt}.json';assert not target.exists(),'Preserve pilot evidence'
 files=[(OUT/'step03R_language_pilot.py','pilot.py'),(PROJECT/'bci_vla/high_level_v4/lora.py','high_level_lora.py')]
 plan={'scope':'user-requested Step03 high-level gap remediation; bounded LoRA pilot','remote_job':job,'steps':240,'wall_timeout_s':900,'GPU':7,'rank':16,'alpha':32,'targets':['q_proj','v_proj','o_proj'],'learning_rate':.001,'source':'existing server LIBERO train/validation captures only; labels generated from explicit selections and declared symbolic state','no_local_project_data_upload':True,'no_old_private_labels_read':True,'after_grasp_examples':'symbolic text-only; no matching real grasp images, not expert demos','formal_test_sealed':True,'base_weights_frozen':True,'action_calls':False,'acceptance':'pilot is not automatic gate pass; independent contrast checks and scope audit required','files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'remote_name':name} for p,name in files]}
 (OUT/f'03R_language_pilot_plan_v{attempt}.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
 spec=importlib.util.spec_from_file_location('remote_reader',ROOT/'.server-inspect/remote_readonly.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);client=mod.connect()
 try:
  ftp=client.open_sftp();ftp.mkdir(job)
  for path,name in files:ftp.put(str(path),job+'/'+name)
  ftp.close()
  cmd='cd '+shlex.quote(job)+' && timeout --kill-after=10 900 env CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/oyjt/Scene-aware-BCI-VLA-runs/step15C_full_20261016_v1/deps:/home/oyjt/openpi/src /home/oyjt/openpi/.venv/bin/python -u pilot.py'
  _,stdout,stderr=client.exec_command(cmd,timeout=920);lines=[]
  with (OUT/f'03R_language_pilot_v{attempt}.log').open('w',encoding='utf-8') as log:
   for line in stdout:
    lines.append(line);log.write(line);log.flush()
    if line.startswith('PILOT_STAGE='):print(line.strip(),flush=True)
  err=stderr.read().decode(errors='replace');exitcode=stdout.channel.recv_exit_status();ftp=client.open_sftp()
  try:ftp.get(job+'/pilot_result.json',str(target));value=json.loads(target.read_text(encoding='utf-8'))
  except FileNotFoundError:value={'status':'TIMEOUT_OR_PROCESS_ERROR','error':'No final result; inspect partial log'}
  for name in ['before.json','step120.json','step240.json','progress.json']:
   try:ftp.get(job+'/'+name,str(OUT/f'03R_language_v{attempt}_{name}'))
   except FileNotFoundError:pass
  ftp.close();value.update(remote_job=job,remote_exit_code=exitcode,stderr=err[-10000:]);target.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:value.get(k) for k in ['status','optimizer_steps','error','off_regression_max_abs','language_loss_last20']}),flush=True)
 finally:client.close()
if __name__=='__main__':main()
