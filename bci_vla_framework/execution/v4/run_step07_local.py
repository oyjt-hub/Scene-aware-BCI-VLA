"""Local prework tests; no server/data transfer/checkpoint load/training."""
import os,sys,json,time,unittest,platform,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'execution/v4'
sys.path[:0]=[str(ROOT.parent/'.runtime/step06_cpu'),str(ROOT)];sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
import torch
torch.set_num_threads(2)
suite=unittest.TestSuite()
for name in ['test_prefix_integration_v4.py','test_neural_interaction_v4.py']:
 suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=name))
start=time.time();result=unittest.TextTestRunner(verbosity=2).run(suite)
report={'status':'PASS_PREWORK_ONLY' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'elapsed_s':time.time()-start,'failures':[(str(t),s) for t,s in result.failures],'errors':[(str(t),s) for t,s in result.errors],'torch':torch.__version__,'python':platform.python_version(),'device':'cpu','cuda_initialized':torch.cuda.is_initialized(),'backend':'tiny random attention reference with audited Pi05 two-stream API, NOT real Gemma/checkpoint','actual_LIBERO_pixels_and_public_features':True,'private_labels_uploaded':False,'checkpoint_loaded':False,'high_level_capability_passed':False,'step07_acceptance_passed':False,'optimizer_steps':0,'training_run':False,'LoRA_used':False,'action_expert_called':False,'formal_test_accessed':False}
(OUT/'07_local_result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['status','tests_run','elapsed_s','checkpoint_loaded','step07_acceptance_passed']}));sys.exit(0 if result.wasSuccessful() else 1)
