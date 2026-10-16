"""Isolated second-VLA simulation test; no modification of existing environments."""
import json,os,socket,subprocess,time,shutil
from pathlib import Path
base=Path(__file__).resolve().parent;port=16815
checkpoint='/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT'
check=socket.socket();check.bind(('127.0.0.1',port));check.close()
deps=base/'deps';deps.mkdir(exist_ok=True)
shutil.copytree('/home/oyjt/openpi/.venv/lib/python3.11/site-packages/transformers',deps/'transformers',
    ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copytree('/home/oyjt/openpi/src/openpi/models_pytorch/transformers_replace',deps/'transformers',dirs_exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',JAX_PLATFORMS='cpu',
    TORCHDYNAMO_DISABLE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONPATH=str(base)+':'+str(deps)+':/home/oyjt/openpi/src:/home/oyjt/openpi/packages/openpi-client/src:/home/oyjt/LIBERO')
server_code='''import torch
from openpi.training import config
from openpi.policies.policy_config import create_trained_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
torch.manual_seed(7)
policy=create_trained_policy(config.get_config('pi05_libero'),CHECKPOINT,pytorch_device='cuda:0')
WebsocketPolicyServer(policy,host='127.0.0.1',port=16815,metadata={'scope':'step15_pi05_LIBERO_simulation','checkpoint':CHECKPOINT}).serve_forever()
'''.replace('CHECKPOINT',repr(checkpoint))
report={'scope':'owned_pi05_simulation_processes','started_at_unix':time.time(),'port':port,'server_python':'openpi/.venv',
    'transformers_overlay':'task-local4.53.2_with_repository_patches','original_environment_modified':False,'torch_compile_disabled_for_smoke':True}
server=None
try:
    with (base/'server.log').open('w') as log:
        server=subprocess.Popen(['/home/oyjt/openpi/.venv/bin/python','-u','-c',server_code],env=env,stdout=log,stderr=subprocess.STDOUT)
        report['server_pid']=server.pid;deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError('owned_model_server_exited:'+str(server.returncode))
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.5):break
            except OSError:time.sleep(2)
        else:raise TimeoutError('model_start_timeout')
        with (base/'worker.log').open('w') as log:
            result=subprocess.run(['/home/oyjt/workspace/miniconda3/envs/libero/bin/python','-u',str(base/'step15_libero_worker.py'),
                '--output',str(base/'results'),'--port',str(port),'--checkpoint-root',checkpoint],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=660)
            report['worker_exit_code']=result.returncode
except Exception as exc:report['error']=type(exc).__name__+':'+str(exc)
finally:
    if server:
        server.terminate()
        try:server.wait(15)
        except subprocess.TimeoutExpired:server.kill();server.wait(5)
        report['owned_server_stopped']=server.poll() is not None
    report['finished_at_unix']=time.time();(base/'orchestration.json').write_text(json.dumps(report,indent=2))
