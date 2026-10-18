"""Inspect existing LIBERO tasks/scene lineage without running policies."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
PAYLOAD=r'''
import os
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
from pathlib import Path
import hashlib,json
from libero.libero import benchmark,get_libero_path
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
tasks=[]
for name in ['libero_spatial','libero_object','libero_goal','libero_90','libero_10']:
 suite=benchmark.get_benchmark_dict()[name]()
 for i in range(suite.n_tasks):
  task=suite.get_task(i);p=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
  init=Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
  tasks.append(dict(suite=name,index=i,name=task.name,language=task.language,bddl=str(p),bddl_sha256=digest(p),bddl_text=p.read_text(),init_states=str(init),init_states_exists=init.exists(),init_states_sha256=digest(init) if init.exists() else None))
print('V4_CATALOG_JSON='+json.dumps(dict(scope='existing LIBERO task catalogue only; no model, rendering or actions',tasks=tasks)))
'''
def main():
 datasets=sys.argv[1:]==['--datasets']
 assert sys.argv[1:] in ([],['--datasets'])
 target=OUT/('04_dataset_paths.json' if datasets else '04_libero_catalog.json')
 assert not target.exists(),'Preserve existing catalogue'
 spec=importlib.util.spec_from_file_location('private_remote_reader',ROOT/'.server-inspect/remote_readonly.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);client=mod.connect()
 try:
  payload=PAYLOAD
  if datasets:
   payload="""from pathlib import Path
import json
from libero.libero import get_libero_path
roots=[Path(get_libero_path('datasets')),Path('/home/oyjt/workspace/datasets'),Path('/home/oyjt/LIBERO/libero/datasets'),Path('/home/oyjt/datasets')]
rows=[]
for root in roots:
 row={'root':str(root),'exists':root.exists(),'hdf5_files':[]}
 if root.is_dir():
  row['children']=sorted(p.name for p in root.iterdir())[:50]
  for pattern in ['*.hdf5','*/*.hdf5','*/*/*.hdf5']:
   row['hdf5_files'].extend(str(p) for p in list(root.glob(pattern))[:50])
 rows.append(row)
print('V4_CATALOG_JSON='+json.dumps({'scope':'bounded existing dataset path search, not exhaustive','roots':rows}))
"""
  command="cd /home/oyjt/LIBERO && timeout --kill-after=5 60 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/oyjt/LIBERO /home/oyjt/workspace/miniconda3/envs/libero/bin/python -u - <<'CATALOG'\n"+payload+"\nCATALOG\n"
  _,stdout,stderr=client.exec_command(command,timeout=70);raw=stdout.read().decode();error=stderr.read().decode();code=stdout.channel.recv_exit_status()
  if code:raise RuntimeError(error[-2500:])
  value=json.loads(raw.rsplit('V4_CATALOG_JSON=',1)[1]);target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  print(json.dumps(dict(saved=str(target),tasks=len(value.get('tasks',[])),roots=value.get('roots',[]))))
 finally:client.close()
if __name__=='__main__':main()
