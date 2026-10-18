"""Read only two existing checkpoint headers and neighboring public configs."""
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
CODE=r'''
import json,struct
from pathlib import Path
paths=['/home/oyjt/workspace/RLinf-RECAP/models/openpi_pytorch/pi05_base/model.safetensors','/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors']
rows=[]
for name in paths:
 p=Path(name)
 with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
 keys=[k for k in h if k!='__metadata__'];picked=[k for k in keys if any(s in k for s in ['embed_tokens.weight','lm_head.weight','layers.0.self_attn.q_proj.weight','layers.0.self_attn.k_proj.weight','layers.0.mlp.gate_proj.weight','action_out_proj.weight'])]
 configs={}
 for c in p.parent.glob('*.json'):
  if c.name in ['config.json','model_config.json','tokenizer_config.json']:configs[c.name]=json.loads(c.read_text())
 rows.append({'path':name,'bytes':p.stat().st_size,'metadata':h.get('__metadata__'),'num_tensors':len(keys),'selected_tensors':{k:{x:h[k][x] for x in ['dtype','shape']} for k in picked},'configs':configs,'siblings':[q.name for q in p.parent.iterdir()][:15]})
print(json.dumps(rows))
'''
def main():
 target=OUT/'03R_weight_headers.json';assert not target.exists()
 spec=importlib.util.spec_from_file_location('remote_reader',ROOT/'.server-inspect/remote_readonly.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);client=mod.connect()
 try:
  _,stdout,stderr=client.exec_command("timeout 30 /home/oyjt/openpi/.venv/bin/python - <<'HEADER'\n"+CODE+"\nHEADER\n",timeout=40)
  raw=stdout.read().decode();err=stderr.read().decode();code=stdout.channel.recv_exit_status();value={'checkpoints':json.loads(raw),'exit_code':code,'stderr':err};target.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value))
 finally:client.close()
if __name__=='__main__':main()
