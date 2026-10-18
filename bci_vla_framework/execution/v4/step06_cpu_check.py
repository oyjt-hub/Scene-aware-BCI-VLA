"""Isolated server entry: CPU only, random weights, no optimizer or VLA import."""
import os,sys,json,hashlib,unittest,time,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(2);torch.manual_seed(17)
from bci_vla.neural_interaction_v4.model import matched_models,count_parameters
from bci_vla.neural_interaction_v4.tensorizer import collate_public
from bci_vla.neural_interaction_v4.export import inspect_prediction

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';assert not torch.cuda.is_initialized()
 mp=ROOT/'execution/v4/06_cpu_input_manifest.json'
 if not mp.exists():mp=ROOT/'execution/v4/06_upload_manifest.json'
 manifest=json.loads(mp.read_text(encoding='utf-8'))
 for row in manifest['files']:assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
 started=time.time();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_neural_interaction_v4.py')
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 models=matched_models();audit={}
 for name,model in models.items():
  groups={}
  for key,p in model.named_parameters():groups[key.split('.')[0]]=groups.get(key.split('.')[0],0)+p.numel()
  audit[name]={'parameters':count_parameters(model),'component_parameters':groups,'device':str(next(model.parameters()).device),'mlp_width':getattr(model,'width',None),'trainable':True,'trained':False,'prefix_tokens':8,'aux_labels':'same masked intent/channel/interaction losses; no language loss yet'}
 rows=json.loads((ROOT/'execution/v4/06_fixtures/public.json').read_text());x,meta=collate_public([rows[0]])
 with torch.no_grad():sample=inspect_prediction(models['Full'].eval()(x),meta)
 report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'failures':[(str(t),s) for t,s in result.failures],'errors':[(str(t),s) for t,s in result.errors],'elapsed_s':time.time()-started,'python':platform.python_version(),'torch':torch.__version__,'platform':platform.system(),'execution_location':'local_windows' if platform.system()=='Windows' else 'isolated_project_server','cuda_initialized':torch.cuda.is_initialized(),'device':'cpu','models':audit,'sample_prediction':sample,'input_files_verified':len(manifest['files']),'optimizer_steps':0,'weights_saved':False,'VLA_loaded':False,'simulation_dynamics_advanced':False,'real_LIBERO_capture_inputs':True,'real_EEG':False,'physical_robot':False,'formal_test_accessed':False,'training_run':False}
 (ROOT/'execution/v4/06_cpu_result.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['status','tests_run','elapsed_s','torch','cuda_initialized']}));return 0 if result.wasSuccessful() else 1
if __name__=='__main__':sys.exit(main())
