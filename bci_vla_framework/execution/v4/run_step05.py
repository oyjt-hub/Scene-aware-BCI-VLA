"""Run B0 and applicable actual legacy-filter reference on LIBERO development."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from bci_vla.data_v4.channel import digest
from bci_vla.evaluation_v4.policies import RecentChoice,LegacyReference
from bci_vla.evaluation_v4.runner import run_episode,text_serialization
DATA=ROOT/'data/v4';OUT=ROOT/'execution/v4/05_runs'
def main():
 if (OUT/'summary.json').exists():
  assert sys.argv[1:]==['--refresh-unfrozen'] and not (ROOT/'execution/v4/05_acceptance.json').exists(),'Preserve frozen runs'
 OUT.mkdir(exist_ok=True)
 captures=json.loads((DATA/'captures/capture_manifest.json').read_text())['episodes']
 worlds=[json.loads(s) for s in (DATA/'private_worlds.jsonl').read_text(encoding='utf-8').splitlines()]
 records=[];unavailable=[]
 for cap in captures:
  if cap['partition']!='validation':continue
  for variant in ['stable','goal_change','channel_change','both_change']:
   world=next(w for w in worlds if w['base_episode_id']==cap['base_episode_id'] and w['variant']==variant)
   for kind in [RecentChoice,LegacyReference]:
    name=kind.name;run_id=digest([cap['base_episode_id'],variant,name])[:24]
    if kind is LegacyReference and sum(o['object_valid'] for o in cap['frames'][0]['objects'])!=2:
     unavailable.append({'method':name,'base_episode_id':cap['base_episode_id'],'variant':variant,'reason':'legacy filter binary scope; do not discard candidates to force applicability'});continue
    result=run_episode(cap,world,kind(),DATA)
    result['private_condition_for_scoring_only']=variant
    (OUT/(run_id+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    records.append({'run_id':run_id,'method':name,'base_episode_id':cap['base_episode_id'],'variant':variant,'file':str(OUT/(run_id+'.json')),'interactions':sum(r['kind']=='INTERACTION' for r in result['journal']),'submissions':len(result['submissions']),'VLA_called':False})
    if not (OUT/'B2_text_serialization_example.json').exists():
     (OUT/'B2_text_serialization_example.json').write_text(json.dumps(text_serialization(result['journal'][0]['input']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 summary={'status':'PASS','scope':'development interaction plumbing only, not method superiority or robot task success','platform':'actual LIBERO frozen capture replay; current questions generate fresh noisy responses','records':records,'unavailable':unavailable,'methods':{m:{'runs':sum(r['method']==m for r in records),'submissions':sum(r['submissions'] for r in records if r['method']==m)} for m in [RecentChoice.name,LegacyReference.name]},'neural_baselines_run':False,'VLA_called':False,'physical_dispatch':False,'formal_test_read':False,'real_time_physics_advanced':False}
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k not in ['records','unavailable']},ensure_ascii=False))
if __name__=='__main__':main()
