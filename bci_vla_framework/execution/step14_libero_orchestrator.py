"""Own-process-only simulator orchestration, bounded runtime and cleanup."""
import json,os,socket,subprocess,time,sys
from pathlib import Path
base=Path(__file__).resolve().parent
checkpoint='/home/oyjt/workspace/Qwen2.5-VL-GR00T-LIBERO-4in1'
port=16814
probe=socket.socket();probe.bind(('127.0.0.1',port));probe.close()
(base/'deps').mkdir(exist_ok=True)
for target in Path('/home/oyjt/openpi/.venv/lib/python3.11/site-packages').glob('diffusers*'):
    link=base/'deps'/target.name
    if not link.exists():link.symlink_to(target,target_is_directory=target.is_dir())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DEBUG='',
    PYTHONPATH=str(base)+':'+str(base/'deps')+':/home/oyjt/starVLA:/home/oyjt/LIBERO',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
server_code='''import torch
from starVLA.model.framework.base_framework import baseframework
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
vla=baseframework.from_pretrained(CHECKPOINT).to(torch.bfloat16).to('cuda').eval()
WebsocketPolicyServer(vla,host='127.0.0.1',port=16814,idle_timeout=180,metadata={'scope':'LIBERO_step14_simulation','checkpoint':CHECKPOINT}).serve_forever()
'''.replace('CHECKPOINT',repr(checkpoint+'/checkpoints/steps_30000_pytorch_model.pt'))
report={'scope':'owned_simulation_processes_only','started_at_unix':time.time(),'server_port':port,'GPU_physical_index':0}
server=None
try:
    with (base/'server.log').open('w') as log:
        server=subprocess.Popen(['/home/oyjt/workspace/miniconda3/envs/starVLA/bin/python','-u','-c',server_code],env=env,stdout=log,stderr=subprocess.STDOUT)
        report['server_pid']=server.pid
        deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError('owned_model_server_exited:'+str(server.returncode))
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.5):break
            except OSError:time.sleep(2)
        else:raise TimeoutError('owned_model_server_startup_timeout')
        with (base/'worker.log').open('w') as workerlog:
            worker=subprocess.run(['/home/oyjt/workspace/miniconda3/envs/libero/bin/python','-u',str(base/'step14_libero_worker.py'),
                '--output',str(base/'results'),'--port',str(port),'--checkpoint-root',checkpoint],env=env,stdout=workerlog,stderr=subprocess.STDOUT,timeout=660)
            report['worker_exit_code']=worker.returncode
except Exception as exc:report['error']=type(exc).__name__+':'+str(exc)
finally:
    if server is not None:
        server.terminate()
        try:server.wait(15)
        except subprocess.TimeoutExpired:server.kill();server.wait(5)
        report['owned_server_stopped']=server.poll() is not None
    report['finished_at_unix']=time.time()
    (base/'orchestration.json').write_text(json.dumps(report,indent=2))
