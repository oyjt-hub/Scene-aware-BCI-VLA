"""Send diagnostic code only; data/weights already on the authorized server."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
def main():
 sft_only=sys.argv[1:]==['--sft-only']
 if sys.argv[1:] not in [[],['--sft-only']]:raise SystemExit('Unsupported mode')
 prefix='03R_sft_native_probe' if sft_only else '03R_prompt_probe'
 target=OUT/(prefix+'.json');assert not target.exists(),'Preserve diagnosis'
 spec=importlib.util.spec_from_file_location('remote_reader',ROOT/'.server-inspect/remote_readonly.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);client=mod.connect()
 try:
  code=(OUT/'step03R_prompt_probe.py').read_text(encoding='utf-8')
  selector=' V4_03R_CHECKPOINT=libero_sft' if sft_only else ''
  cmd="cd /home/oyjt/openpi && timeout --kill-after=10 600 env CUDA_VISIBLE_DEVICES=7 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1"+selector+" PYTHONPATH=/home/oyjt/Scene-aware-BCI-VLA-runs/step15C_full_20261016_v1/deps:/home/oyjt/openpi/src .venv/bin/python -u - <<'PROBE'\n"+code+"\nPROBE\n"
  _,stdout,stderr=client.exec_command(cmd,timeout=620);lines=[]
  with (OUT/(prefix+'.log')).open('w',encoding='utf-8') as log:
   for line in stdout:
    lines.append(line);log.write(line);log.flush()
    if line.startswith('STAGE='):
     row=json.loads(line[len('STAGE='):]);print(json.dumps({k:row[k] for k in ['name','tag','case','generated','seconds'] if k in row}),flush=True)
  raw=''.join(lines);err=stderr.read().decode(errors='replace');exitcode=stdout.channel.recv_exit_status();marker='V4_03R_RESULT='
  result=json.loads(raw.rsplit(marker,1)[1]) if marker in raw else {'status':'FAILED','error':'No result marker'};result.update(remote_exit_code=exitcode,stderr=err[-6000:]);target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
  print(json.dumps({'status':result['status'],'error':result.get('error'),'cases':[{'checkpoint':s['checkpoint'],'case':s['case'],'generated':s['generated'],'exact_match':s['exact_match']} for s in result.get('stages',[]) if s['name']=='prompt_case']}))
 finally:client.close()
if __name__=='__main__':main()
