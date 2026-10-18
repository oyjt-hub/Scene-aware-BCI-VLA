"""Read-only existing server assets and runtime for user-requested gap repair."""
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
CODE=r'''
from pathlib import Path
import os,json,subprocess,importlib.util
roots=[Path('/home/oyjt/workspace/checkpoints'),Path('/home/oyjt/workspace/RLinf-RECAP/models'),Path('/home/oyjt/.cache/openpi')]
rows=[]
for root in roots:
 if not root.exists():continue
 for current,dirs,files in os.walk(root):
  depth=len(Path(current).relative_to(root).parts)
  dirs[:]=[d for d in dirs if d not in ['.git','.venv','venv','node_modules','datasets','logs','wandb','data','params','d','ocdbt.process_0'] and depth<3]
  for name in files:
   if name.endswith('.safetensors') or name in ['paligemma_tokenizer.model','tokenizer.model']:
    p=Path(current)/name;rows.append({'path':str(p),'bytes':p.stat().st_size})
  if len(rows)>200:break
print(json.dumps({'existing_weights_and_tokenizers':rows,'gpu':subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout,'peft_installed':importlib.util.find_spec('peft') is not None,'overlay_exists':Path('/home/oyjt/Scene-aware-BCI-VLA-runs/step15C_full_20261016_v1/deps/transformers').is_dir()}))
'''
def main():
 path=OUT/'03R_asset_inspection.json';assert not path.exists(),'Preserve evidence'
 spec=importlib.util.spec_from_file_location('remote_reader',ROOT/'.server-inspect/remote_readonly.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);client=mod.connect()
 try:
  _,stdout,stderr=client.exec_command("timeout 45 /home/oyjt/openpi/.venv/bin/python - <<'READONLY'\n"+CODE+"\nREADONLY\n",timeout=55)
  raw=stdout.read().decode();err=stderr.read().decode();code=stdout.channel.recv_exit_status();value=json.loads(raw) if raw.strip() else {'status':'inspection_timeout_or_error'};value.update(exit_code=code,stderr=err)
  path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value))
 finally:client.close()
if __name__=='__main__':main()
