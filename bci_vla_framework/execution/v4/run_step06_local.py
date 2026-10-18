"""Run final CPU verification locally, without transferring project data."""
import os,sys,json,hashlib,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'execution/v4'
runtime=ROOT.parent/'.runtime/step06_cpu'
if not (runtime/'torch').is_dir():raise SystemExit('Install official torch CPU in workspace .runtime/step06_cpu first')
sys.path[:0]=[str(runtime),str(ROOT)]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';sys.dont_write_bytecode=True
files=list((ROOT/'bci_vla/neural_interaction_v4').glob('*.py'))+list((OUT/'06_fixtures').glob('*.json'))
files+=[ROOT/'tests/test_neural_interaction_v4.py',OUT/'step06_cpu_check.py',Path(__file__).resolve()]
for folder in ['bci_vla/interaction_contract_v4','bci_vla/data_v4']:files+=list((ROOT/folder).glob('*.py'))
files+=[ROOT/'configs/v4/interfaces.json',ROOT/'configs/v4/interfaces_data_4_1.json']+list((ROOT/'examples/v4').glob('*.json'))
manifest={'location':'local Windows CPU; no upload','files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(files))]}
(OUT/'06_cpu_input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
runpy.run_path(str(OUT/'step06_cpu_check.py'),run_name='__main__')
