"""Preserve earlier evidence before prerequisite-gated integration development."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'execution/v4'
def main():
 target=OUT/'07_before_manifest.json';assert not target.exists(),'Preserve prior evidence'
 rows={}
 for name in ['06_before_manifest.json','06_source_manifest.json']:
  for row in json.loads((OUT/name).read_text(encoding='utf-8'))['files']:rows[row['path']]=row
 for p in [OUT/'06_source_manifest.json',OUT/'06_before_manifest.json']:
  rows[str(p)]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 target.write_text(json.dumps({'files':list(rows.values())},indent=2)+'\n',encoding='utf-8')
 pp=ROOT/'execution/v4_plan_progress.json';old=json.loads(pp.read_text(encoding='utf-8'))
 (OUT/'07_progress_before.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 old['steps'][6].update(status='进行中（高层能力依赖未满足）',completion_scope='仅开展本地接入工程预备；不得宣布真实π0.5接入验收通过')
 old.update(current_stage='V4-07_local_integration_preparation_with_unmet_V4-03_gate',authorization_this_turn='Execute V4-07; implement and locally verify adapter prework, preserve failed high-level gate; no checkpoint training, private-data transfer, EEG, robot, formal test or VLA action.')
 pp.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('V4-07 preparation started; high-level dependency remains unmet')
if __name__=='__main__':main()
