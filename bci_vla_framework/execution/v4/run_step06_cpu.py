"""Deploy a hash-bound isolated CPU bundle; preserve each attempt's evidence."""
import importlib.util,json,shlex,hashlib,zipfile,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];PROJECT=ROOT/'bci_vla_framework';HERE=Path(__file__).resolve().parent
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,required=True);args=parser.parse_args()
 job=f'/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step06_cpu_20261017_v{args.attempt}'
 result_path=HERE/f'06_cpu_attempt{args.attempt}.json';assert not result_path.exists(),'Preserve attempt evidence'
 files=[]
 for folder in ['bci_vla/neural_interaction_v4','bci_vla/interaction_contract_v4','bci_vla/data_v4']:
  files.extend(p for p in (PROJECT/folder).glob('*.py'))
 files.extend([PROJECT/'bci_vla/__init__.py',PROJECT/'tests/test_neural_interaction_v4.py',HERE/'step06_cpu_check.py'])
 files.extend(HERE.joinpath('06_fixtures').glob('*.json'))
 files.extend(PROJECT.joinpath('examples/v4').glob('*.json'))
 files.extend([PROJECT/'configs/v4/interfaces.json',PROJECT/'configs/v4/interfaces_data_4_1.json'])
 manifest={'files':[{'path':p.relative_to(PROJECT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(files))]}
 mp=HERE/'06_upload_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 archive=HERE/'06_cpu_bundle.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(set(files)|{mp}):z.write(p,p.relative_to(PROJECT).as_posix())
 spec=importlib.util.spec_from_file_location('remote_reader',ROOT/'.server-inspect/remote_readonly.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);client=mod.connect()
 try:
  ftp=client.open_sftp();ftp.mkdir(job);ftp.put(str(archive),job+'/bundle.zip');ftp.close()
  command='cd '+shlex.quote(job)+' && /home/oyjt/openpi/.venv/bin/python -m zipfile -e bundle.zip . && timeout --kill-after=5 180 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/oyjt/openpi/.venv/bin/python -u execution/v4/step06_cpu_check.py > cpu.log 2>&1'
  _,stdout,stderr=client.exec_command(command,timeout=200);out=stdout.read().decode(errors='replace');err=stderr.read().decode(errors='replace');code=stdout.channel.recv_exit_status()
  ftp=client.open_sftp();ftp.get(job+'/cpu.log',str(HERE/f'06_cpu_attempt{args.attempt}.log'))
  try:ftp.get(job+'/execution/v4/06_cpu_result.json',str(result_path))
  except FileNotFoundError:result_path.write_text(json.dumps({'status':'RUNNER_ERROR','exit_code':code,'stderr':err,'stdout':out}))
  ftp.close();r=json.loads(result_path.read_text());r.update(remote_job=job,remote_exit_code=code,bundle_sha256=hashlib.sha256(archive.read_bytes()).hexdigest());result_path.write_text(json.dumps(r,indent=2)+'\n')
  print(json.dumps({'status':r['status'],'remote_exit_code':code,'tests_run':r.get('tests_run'),'failures':r.get('failures'),'errors':r.get('errors'),'log':str(HERE/f'06_cpu_attempt{args.attempt}.log')}))
 finally:client.close()
if __name__=='__main__':main()
