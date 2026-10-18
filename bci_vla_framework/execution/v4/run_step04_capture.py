"""Own isolated LIBERO data job, then copy actual capture artifacts locally."""
import importlib.util,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
JOB='/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step04_libero_data_20261017_v1'
def main():
 target=HERE/'04_capture_job.json';assert not target.exists(),'Preserve capture evidence'
 spec=importlib.util.spec_from_file_location('remote_reader',ROOT/'.server-inspect/remote_readonly.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);client=mod.connect()
 try:
  ftp=client.open_sftp();ftp.mkdir(JOB)
  for source,name in [(HERE/'step04_libero_capture.py','capture.py'),(HERE/'04_capture_plan.json','capture_plan.json')]:ftp.put(str(source),JOB+'/'+name)
  ftp.close()
  command='cd '+shlex.quote(JOB)+' && timeout --kill-after=10 600 env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/oyjt/LIBERO /home/oyjt/workspace/miniconda3/envs/libero/bin/python -u capture.py'
  _,stdout,stderr=client.exec_command(command,timeout=620);raw=stdout.read().decode(errors='replace');err=stderr.read().decode(errors='replace');code=stdout.channel.recv_exit_status()
  ftp=client.open_sftp();local=HERE.parents[1]/'data/v4/captures';local.mkdir(parents=True,exist_ok=True);(local/'images').mkdir(exist_ok=True)
  ftp.get(JOB+'/captures/capture_manifest.json',str(local/'capture_manifest.json'))
  value=json.loads((local/'capture_manifest.json').read_text())
  for episode in value['episodes']:
   for frame in episode['frames']:
    name=Path(frame['image_path']).name;ftp.get(JOB+'/captures/images/'+name,str(local/'images'/name))
  ftp.close();record={'remote_job':JOB,'remote_exit_code':code,'stdout':raw[-4000:],'stderr':err[-4000:],'status':value['status'],'episodes':len(value['episodes']),'test_rendered':False}
  target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))
 finally:client.close()
if __name__=='__main__':main()
