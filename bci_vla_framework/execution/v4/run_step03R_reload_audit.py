"""Send audit code only to the existing pilot directory; copy results/adapter in."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
JOB='/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step03R_language_20261017_v1'
def main():
 state=sys.argv[1:]==['--state-precision'];prefix='03R_state_precision_audit' if state else '03R_reload_audit'
 if sys.argv[1:] not in [[],['--state-precision']]:raise SystemExit('Unsupported mode')
 target=OUT/(prefix+'.json');assert not target.exists()
 spec=importlib.util.spec_from_file_location('remote_reader',ROOT/'.server-inspect/remote_readonly.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);c=mod.connect()
 try:
  remote_script='state_cache_check.py' if state else 'reload_check.py';ftp=c.open_sftp();ftp.put(str(OUT/'step03R_reload_audit.py'),JOB+'/'+remote_script);ftp.close()
  mode=' V4_AUDIT_MODE=state_precision' if state else ''
  cmd='cd '+JOB+' && timeout --kill-after=10 600 env CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1'+mode+' PYTHONPATH=/home/oyjt/Scene-aware-BCI-VLA-runs/step15C_full_20261016_v1/deps:/home/oyjt/openpi/src /home/oyjt/openpi/.venv/bin/python -u '+remote_script
  _,stdout,stderr=c.exec_command(cmd,timeout=620)
  with (OUT/(prefix+'.log')).open('w',encoding='utf-8') as log:
   for line in stdout:
    log.write(line);log.flush()
    if line.startswith(('AUDIT_STAGE=','AUDIT_RESULT=')):print(line.strip(),flush=True)
  err=stderr.read().decode(errors='replace');code=stdout.channel.recv_exit_status();ftp=c.open_sftp();ftp.get(JOB+('/state_precision_audit.json' if state else '/reload_audit.json'),str(target))
  artifacts=ROOT/'bci_vla_framework/artifacts/v4/high_level_lora';artifacts.mkdir(parents=True,exist_ok=True)
  if not (artifacts/'step240.safetensors').exists():ftp.get(JOB+'/high_level_lora_step240.safetensors',str(artifacts/'step240.safetensors'))
  ftp.close()
  r=json.loads(target.read_text(encoding='utf-8'));r.update(remote_exit_code=code,stderr=err[-6000:]);target.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps({'saved':str(target),'local_adapter':str(artifacts/'step240.safetensors')}))
 finally:c.close()
if __name__=='__main__':main()
