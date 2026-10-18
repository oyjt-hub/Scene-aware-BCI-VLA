"""Freeze scene-disjoint LIBERO development splits before rendering/noise."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def parse(s):
 tokens=iter(re.findall(r'\(|\)|[^\s()]+',s))
 def part():
  out=[]
  for t in tokens:
   if t==')':return out
   out.append(part() if t=='(' else t)
  return out
 assert next(tokens)=='('
 return part()
def typed(section):
 values=section[1:];result={};i=0;pending=[]
 while i<len(values):
  if values[i]=='-':
   for name in pending:result[name]=values[i+1]
   pending=[];i+=2
  else:pending.append(values[i]);i+=1
 return result
def main():
 out=ROOT/'data/v4';out.mkdir(parents=True,exist_ok=True)
 assert not (out/'split_manifest.json').exists(),'Preserve frozen split'
 cat=json.loads((ROOT/'execution/v4/04_libero_catalog.json').read_text(encoding='utf-8'))
 grouped={}
 for row in cat['tasks']:
  if row['suite']!='libero_90':continue
  tree=parse(row['bddl_text']);sections={v[0]:v for v in tree if isinstance(v,list) and v}
  objects=typed(sections[':objects']);fixtures=typed(sections.get(':fixtures',[':fixtures']))
  goal=sections[':goal'][1]
  terms=goal[1:] if goal[0]=='And' else [goal]
  if len(terms)!=1 or len(terms[0])!=3 or terms[0][0] not in ['On','In']:continue
  relation,src,dst=terms[0]
  if src not in objects:continue
  destination=next((v for v in sorted(list(objects)+list(fixtures),key=len,reverse=True) if dst==v or dst.startswith(v+'_')),None)
  if destination is None or destination==src:continue
  family=re.match(r'(.+?_SCENE\d+)_',row['name']).group(1)
  geometry=digest([sections.get(k) for k in [':domain',':regions',':fixtures',':objects']])
  item={k:row[k] for k in ['suite','index','name','language','bddl','bddl_sha256','init_states','init_states_sha256']}
  item.update(scene_family=family,geometry_sha256=geometry,objects=objects,fixtures=fixtures,benchmark_goal={'relation':relation,'source':src,'destination':destination,'destination_region':dst})
  grouped.setdefault(family,[]).append(item)
 # Each family is one physical scene lineage. Also merge matching geometry across families.
 parents={k:k for k in grouped}
 def find(k):
  while parents[k]!=k:k=parents[k]
  return k
 geometries={};init_hashes={}
 for family,items in grouped.items():
  for item in items:
   for key,table in [(item['geometry_sha256'],geometries),(item['init_states_sha256'],init_hashes)]:
    if key in table:parents[find(family)]=find(table[key])
    else:table[key]=family
 groups={}
 for family,items in grouped.items():groups.setdefault(find(family),[]).extend(items)
 keys=sorted(groups,key=lambda k:digest(['v4-libero-scene-split-1',k]))
 assert len(keys)>=14,('Need 14 independent scene groups',len(keys))
 allocation=['train']*6+['validation']*2+['calibration_fit']*2+['calibration_audit']*2+['test_sealed']*2
 chosen=[]
 for group,split in zip(keys[:14],allocation):
  options=sorted(groups[group],key=lambda t:(not any(list(t['objects'].values()).count(c)>1 for c in t['objects'].values()),t['index']))
  item=options[0];item.update(group_id=digest(['scene',group])[:20],partition=split,init_indices=[0,1,2,3])
  chosen.append(item)
 manifest={'version':'libero-data-v4.1-dev1','scope':'48 actual LIBERO development base episodes, 8 sealed planned test episodes; no formal evaluation','seed_namespace':'v4-libero-scene-split-1','assignment_before_augmentation':True,'scene_grouping':'union scene-family OR canonical BDDL geometry OR identical init-state file hash','object_instance_scope':'scene_family::object_id; asset classes may recur across groups; no unseen-asset claim','partitions':{s:[t for t in chosen if t['partition']==s] for s in sorted(set(allocation))},'test_policy':'Reserve task/init IDs only. No test rendering, noise generation, predictions, calibration or evaluation until V4-13 freeze.','total_eligible_scene_groups':len(groups),'eligible_tasks':sum(len(v) for v in groups.values()),'formal_test_unsealed':False}
 (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 plan={'version':manifest['version'],'episodes':[dict(t,base_episode_id=digest([t['group_id'],i])[:24],init_index=i) for t in chosen if t['partition']!='test_sealed' for i in t['init_indices']],'test_rendering_allowed':False,'source':'actual LIBERO simulator; privileged state/masks explicitly tagged','states_per_episode':2,'simulation_settle_steps':10,'probe_motion_steps':2}
 (ROOT/'execution/v4/04_capture_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'scene_groups':len(groups),'development_episodes':len(plan['episodes']),'sealed_test_episodes':8,'same_class_development_groups':sum(any(list(t['objects'].values()).count(c)>1 for c in t['objects'].values()) for t in chosen if t['partition']!='test_sealed')}))
if __name__=='__main__':main()
