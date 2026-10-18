"""Actual LIBERO rendering/state capture, not policy or task-success evaluation."""
import os
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_camera_transform_matrix,project_points_from_world_to_camera
ROOT=Path(__file__).resolve().parent
def digest(raw):return hashlib.sha256(raw).hexdigest()
def main():
 plan=json.loads((ROOT/'capture_plan.json').read_text());out=ROOT/'captures';(out/'images').mkdir(parents=True,exist_ok=True)
 report={'scope':'actual LIBERO environment, reset/settle and two neutral simulation steps; no policy/model/EEG/physical robot','episodes':[],'errors':[],'test_rendered':False}
 suite=benchmark.get_benchmark_dict()['libero_90']();env=None;current=None
 try:
  for item in plan['episodes']:
   assert item['partition']!='test_sealed'
   if current!=item['index']:
    if env is not None:env.close()
    task=suite.get_task(item['index']);path=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    assert digest(path.read_bytes())==item['bddl_sha256']
    env=OffScreenRenderEnv(bddl_file_name=str(path),camera_heights=256,camera_widths=256)
    states=suite.get_task_init_states(item['index']);current=item['index']
   env.seed(1234+item['init_index']);env.reset();obs=env.set_init_state(states[item['init_index']])
   state_hash=digest(np.asarray(states[item['init_index']]).tobytes())
   for _ in range(plan['simulation_settle_steps']):obs,_,_,_=env.step([0.]*6+[-1.])
   frames=[]
   for frame_index in range(2):
    if frame_index:
     for _ in range(plan['probe_motion_steps']):obs,_,_,_=env.step([0.]*6+[-1.])
    image=np.ascontiguousarray(obs['agentview_image'][::-1])
    file=out/'images'/(item['base_episode_id']+'_'+str(frame_index)+'.png');Image.fromarray(image).save(file)
    sim=env.env.sim;transform=get_camera_transform_matrix(sim,'agentview',256,256);objects=[];predicates={}
    for oid,kind in sorted({**item['fixtures'],**item['objects']}.items()):
     obj=env.env.objects_dict.get(oid)
     if obj is None:continue
     corners=[]
     for geom in obj.contact_geoms:
      gid=sim.model.geom_name2id(geom);center=sim.data.geom_xpos[gid];radius=float(np.linalg.norm(sim.model.geom_size[gid]))
      for x in [-1,1]:
       for y in [-1,1]:
        for z in [-1,1]:corners.append(center+radius*np.asarray([x,y,z]))
     if not corners:continue
     pixels=project_points_from_world_to_camera(np.asarray(corners),transform,256,256)
     low=pixels.min(axis=0);high=pixels.max(axis=0)
     box=np.clip([low[1]/256,low[0]/256,(high[1]+1)/256,(high[0]+1)/256],0,1).tolist()
     valid=box[2]>box[0] and box[3]>box[1]
     feature=None
     if valid:
      x1,y1,x2,y2=[int(round(v*255)) for v in box];crop=image[y1:y2+1,x1:x2+1].astype(np.float64)/255
      if crop.size:feature=np.concatenate([crop.mean(axis=(0,1)),crop.std(axis=(0,1))]).tolist()
     objects.append({'instance_id':oid,'alias':'OBJ_'+str(len(objects)),'label':kind.replace('_',' '),'bbox_xyxy':box if valid else None,'object_valid':bool(valid),'roi_feature':feature,'roi_valid':feature is not None,'semantic_feature':None,'semantic_valid':False})
     if oid in item['objects']:
      bid=env.env.obj_body_id[oid]
      predicates[oid]={'position':sim.data.body_xpos[bid].tolist(),'grasp_contact':bool(env.env._check_grasp(env.env.robots[0].gripper,obj.contact_geoms))}
    frames.append({'frame_index':frame_index,'image_path':'captures/images/'+file.name,'image_sha256':digest(file.read_bytes()),'rgb_sha256':digest(image.tobytes()),'sim_state_sha256':digest(np.asarray(sim.get_state().flatten()).tobytes()),'objects':objects,'predicates':predicates,'source':'privileged_simulator_projection','roi_source':'RGB mean/std from projected bounding crop; not pretrained features','visibility':'in-frame projected geometry only, occlusion not verified'})
   assert 1<=len(frames[0]['objects'])<=16
   report['episodes'].append({'base_episode_id':item['base_episode_id'],'group_id':item['group_id'],'partition':item['partition'],'suite':item['suite'],'task_index':item['index'],'init_index':item['init_index'],'initial_state_sha256':state_hash,'frames':frames,'benchmark_goal_success':bool(env.env._check_success()),'original_bddl_sha256':item['bddl_sha256']})
   (out/'capture_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
  report['status']='PASS'
 except Exception as exc:
  report.update(status='FAIL',error=str(exc),traceback=traceback.format_exc())
 finally:
  if env is not None:env.close()
  (out/'capture_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print('V4_CAPTURE_RESULT='+json.dumps({'status':report.get('status'),'episodes':len(report['episodes']),'error':report.get('error')}),flush=True)
if __name__=='__main__':main()
