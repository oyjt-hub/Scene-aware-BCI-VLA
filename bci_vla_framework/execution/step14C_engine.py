"""Real LIBERO renderer/state predicates and real StarVLA inference."""
import os
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
from pathlib import Path
from collections import deque
import hashlib,json,time,itertools
import numpy as np
import cv2,imageio
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_camera_transform_matrix,project_points_from_world_to_camera
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from bci_vla.pilot.libero_actions import decode_starvla_chunk
from bci_vla.simulation_v3.factory import OBJECTS

class LiberoEngine:
    def __init__(self,folder,port,checkpoint_root,checkpoint):
        self.folder=Path(folder);self.folder.mkdir(exist_ok=True,parents=True);self.client=WebsocketClientPolicy('127.0.0.1',port)
        self.stats=json.loads((Path(checkpoint_root)/'dataset_statistics.json').read_text())['franka']['action']
        self.checkpoint=checkpoint;self.queue=deque();self.cap=160;self.finished=False;self.events=[];self.frames=[]
        self.suite=benchmark.get_benchmark_dict()['libero_goal']();task=self.suite.get_task(6)
        self.bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        self.initial_states=self.suite.get_task_init_states(6)
        self.env=OffScreenRenderEnv(bddl_file_name=str(self.bddl),camera_heights=256,camera_widths=256);self.env.seed(7)
        self.instruction_to_object={'Pick up the '+label+'.':obj for obj,label in OBJECTS}
        self.fail_next=False;self.label='not-reset';self.steps=0;self.instruction=None

    def log(self,kind,**data):
        record=dict(kind=kind,label=self.label,wall_monotonic_s=time.monotonic(),**data);self.events.append(record)
        with (self.folder/'simulator_events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')

    def clear_queue(self):
        count=len(self.queue);self.queue.clear()
        if count:self.log('queue_cleared',discarded=count)

    def reset(self,index,label,cap=160):
        self.clear_queue();self.label=label;start=time.perf_counter();self.env.reset();self.obs=self.env.set_init_state(self.initial_states[index])
        for _ in range(10):self.obs,_,_,_=self.env.step([0.]*6+[-1.])
        self.steps=0;self.finished=False;self.cap=cap;self.frames=[];self.instruction=None;self.held_streak=0
        self.initial_z={name:self.position(name)[2] for name,_ in OBJECTS}
        self.log('reset',init_state_index=index,initial_state_sha256=hashlib.sha256(np.asarray(self.initial_states[index]).tobytes()).hexdigest(),
            bddl_sha256=hashlib.sha256(self.bddl.read_bytes()).hexdigest())
        return time.perf_counter()-start

    def position(self,name):return self.env.env.sim.data.body_xpos[self.env.env.obj_body_id[name]].copy()

    def predicate_details(self):
        result={}
        for name,_ in OBJECTS:
            obj=self.env.env.objects_dict[name]
            grasp=bool(self.env.env._check_grasp(self.env.env.robots[0].gripper,obj.contact_geoms))
            lift=float(self.position(name)[2]-self.initial_z[name])
            result[name]=dict(grasp_contact=grasp,lift_m=lift,held=grasp and lift>.03)
        return result

    def held(self):return {name:detail['held'] for name,detail in self.predicate_details().items()}

    def observation_digests(self):
        images=[hashlib.sha256(np.ascontiguousarray(self.obs[key]).tobytes()).hexdigest() for key in ['agentview_image','robot0_eye_in_hand_image']]
        state=hashlib.sha256(self.env.env.sim.get_state().flatten().tobytes()).hexdigest()
        return images,state

    def observation(self,context,at):
        images,state=self.observation_digests()
        return dict(interface_version='libero-model/3.0',scene_id=context['scene_id'],scene_revision=context['scene_revision'],captured_at=at,
            image_sha256=images,sim_state_sha256=state,scene_source='privileged_simulator_state')

    def scene(self,session_id,revision=1):
        from bci_vla.benchmark.offline import stamp
        transform=get_camera_transform_matrix(self.env.env.sim,'agentview',256,256);objects=[]
        for name,label in OBJECTS:
            # Conservative extents from actual simulator contact geometries.
            corners=[];sim=self.env.env.sim
            for geom in self.env.env.objects_dict[name].contact_geoms:
                gid=sim.model.geom_name2id(geom);center=sim.data.geom_xpos[gid];radius=float(np.linalg.norm(sim.model.geom_size[gid]))
                corners.extend(center+np.array(signs)*radius for signs in itertools.product([-1,1],repeat=3))
            pixels=project_points_from_world_to_camera(np.asarray(corners),transform,256,256)
            low=pixels.min(axis=0);high=pixels.max(axis=0)
            box=[float(low[1]/256),float(low[0]/256),float((high[1]+1)/256),float((high[0]+1)/256)]
            objects.append(dict(instance_id=name,track_token='libero-body:'+name,label=label,box_xyxy=box,visible=True,
                affordances=['pickable'],attributes=[],detection_confidence=None))
        return dict(schema_version='grounded-scene/1.0',scene_id='libero-goal6-init-scene',revision=revision,
            observed_at=dict(host_id='simulator-state',at=stamp(0),offset_ms=0,uncertainty_ms=0),
            coordinate_frame=dict(name='image_xy',unit='normalized',width_px=256,height_px=256),
            identity=dict(session_id=session_id,scope='session',tracker_ref='privileged-LIBERO-body-identity'),objects=objects,relations=[])

    def advance(self,instruction,count):
        if self.fail_next:self.fail_next=False;raise ConnectionError('injected_policy_transport_failure')
        if instruction!=self.instruction:self.clear_queue();self.instruction=instruction
        target=self.instruction_to_object[instruction];applied=0
        for _ in range(count):
            if self.finished:break
            img=np.ascontiguousarray(self.obs['agentview_image'][::-1,::-1]);wrist=np.ascontiguousarray(self.obs['robot0_eye_in_hand_image'][::-1,::-1])
            if not self.queue:
                payload={'examples':[dict(image=[cv2.resize(img,(224,224)),cv2.resize(wrist,(224,224))],lang=instruction)],
                    'do_sample':False,'use_ddim':True,'num_ddim_steps':10}
                start=time.perf_counter();raw=np.asarray(self.client.predict_action(payload)['data']['normalized_actions'],dtype=float)
                actions=decode_starvla_chunk(raw,self.stats);self.queue.extend(actions.tolist())
                self.log('real_model_chunk',instruction=instruction,shape=list(raw.shape),latency_s=time.perf_counter()-start,
                    actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),image_sha256=self.observation_digests()[0])
            action=self.queue.popleft();self.obs,_,_,_=self.env.step(action);self.steps+=1;applied+=1
            details=self.predicate_details();self.held_streak=self.held_streak+1 if details[target]['held'] else 0
            self.finished=self.held_streak>=3 or self.steps>=self.cap
            self.log('actual_simulation_step',step=self.steps,instruction=instruction,action=action,predicates=details)
            if self.steps%2==0:self.frames.append(np.ascontiguousarray(self.obs['agentview_image'][::-1,::-1]))
        return applied

    def save_video(self,label):
        if self.frames:
            imageio.mimwrite(self.folder/(label+'.mp4'),self.frames,fps=10)
            imageio.imwrite(self.folder/(label+'_last.png'),self.frames[-1])

    def close(self):self.clear_queue();self.env.close();self.client.close()
