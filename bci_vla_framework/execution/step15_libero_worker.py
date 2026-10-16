"""Headless LIBERO + real StarVLA weights; scripted interaction, simulation only."""
import os
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import cv2
import imageio
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi_client import image_tools
from bci_vla.pilot.libero_gate import LiberoSimulationGate
from bci_vla.adapters.vla.libero_pi05 import controller_actions
from bci_vla.protocols import ProtocolError

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--port',type=int,required=True)
p.add_argument('--checkpoint-root',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(exist_ok=True)
stats_path=Path(a.checkpoint_root)/'assets/physical-intelligence/libero/norm_stats.json'
client=WebsocketClientPolicy('127.0.0.1',a.port)
suite=benchmark.get_benchmark_dict()['libero_goal']()
events=[];episodes=[];start=time.perf_counter()
def event(kind,**data):
    events.append(dict(kind=kind,elapsed_s=time.perf_counter()-start,**data))
    with (out/'events.jsonl').open('a') as f:f.write(json.dumps(events[-1])+'\n')
def confirm(g,c):g.set_mode('CONFIRM');g.confirm(c,accepted=True,input_source='scripted_simulation_fixture')
def blocked(fn):
    try:fn()
    except ProtocolError:return True
    return False

try:
    for label,task_id,cap in [('nominal_task0',0,300),('probe_stop_task0',0,20),('nominal_task1',1,300)]:
        task=suite.get_task(task_id);bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        init=suite.get_task_init_states(task_id)[0]
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
        env.seed(7);env.reset();obs=env.set_init_state(init)
        g=LiberoSimulationGate();c=g.bind(suite='libero_goal',task_id=task_id,language=task.language,scene_revision=1)
        checks={'preconfirm_blocked':blocked(lambda:g.enqueue([[0.]*7],c))}
        confirm(g,c);frames=[];success=False;fault_stop=False;error=None;inferences=0
        event('episode_start',episode=label,task_id=task_id,language=task.language,bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
            input_source='scripted_simulation_fixture',context=list(c),server_metadata=client.get_server_metadata())
        try:
            for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
            for step in range(cap):
                if label=='probe_stop_task0' and step==5:
                    queued=len(g.queue);g.set_mode('CHANNEL_PROBE')
                    checks['probe_flush']=len(g.queue)==0 and g.confirmed is None
                    checks['probe_blocks']=blocked(lambda:g.pop(c))
                    event('probe_pause',queued_discarded=queued,snapshot=g.snapshot());confirm(g,c)
                if label=='probe_stop_task0' and step==12:
                    g.stop();checks['stop_blocks']=blocked(lambda:g.pop(c));checks['stop_flush']=len(g.queue)==0
                    event('stop_latched',snapshot=g.snapshot());fault_stop=True;break
                img=np.ascontiguousarray(obs['agentview_image'][::-1,::-1])
                wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]);frames.append(img)
                if not g.queue:
                    quat=np.asarray(obs['robot0_eef_quat'],dtype=float).copy()
                    quat[3]=np.clip(quat[3],-1,1);den=np.sqrt(1-quat[3]*quat[3])
                    axis=np.zeros(3) if np.isclose(den,0) else quat[:3]*2*np.arccos(quat[3])/den
                    payload={'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(img,224,224)),
                        'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist,224,224)),
                        'observation/state':np.concatenate([obs['robot0_eef_pos'],axis,obs['robot0_gripper_qpos']]),
                        'prompt':task.language}
                    t=time.perf_counter();response=client.infer(payload)
                    raw=np.asarray(response['actions'],dtype=float)[None,...]
                    assert raw.ndim==3 and raw.shape[0]==1 and raw.shape[2]==7 and np.isfinite(raw).all()
                    lower,upper=env.env.action_spec
                    actions,conversion=controller_actions(raw,lower=lower,upper=upper,replan_steps=8)
                    g.enqueue(actions.tolist(),c);inferences+=1
                    event('model_chunk',episode=label,step=step,shape=list(raw.shape),latency_s=time.perf_counter()-t,
                        image_shapes=[list(img.shape),list(wrist.shape)],raw_clipped_values=int(np.sum(np.abs(raw)>1)),
                        actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),controller_conversion=conversion)
                action=g.pop(c);obs,reward,done,info=env.step(action)
                success=bool(env.check_success())
                event('simulation_step',episode=label,step=step,action=action,done=bool(done),task_success=success)
                if success:break
            if label=='nominal_task0':
                old=c;new=g.bind(suite='libero_goal',task_id=1,language=suite.get_task(1).language,scene_revision=2)
                checks['changed_task_flush']=not g.queue and g.confirmed is None
                checks['old_context_blocked']=blocked(lambda:g.pop(old))
                checks['new_context_requires_confirmation']=blocked(lambda:g.enqueue([[0.]*7],new))
                event('task_change_no_old_action',snapshot=g.snapshot(),next_scene_requires_reset=True)
        except Exception as exc:
            error=type(exc).__name__+':'+str(exc);event('episode_error',episode=label,error=error)
        finally:
            endpoint=bool(env.check_success());g.invalidate();event('terminal_queue_cleared',episode=label,snapshot=g.snapshot());env.close()
        video=label+'.mp4'
        if frames:
            imageio.mimwrite(out/video,frames,fps=20)
            imageio.imwrite(out/(label+'_first.png'),frames[0]);imageio.imwrite(out/(label+'_last.png'),frames[-1])
        episodes.append({'episode':label,'suite':'libero_goal','task_id':task_id,'language':task.language,'seed':7,'init_state_index':0,
            'task_success':endpoint,'termination':'injected_stop' if fault_stop else 'error' if error else 'task_success' if success else 'step_limit',
            'error':error,'applied_actions':g.applied_actions,'inference_calls':inferences,'checks':checks,'gate_final':g.snapshot(),
            'video':video if frames else None})
        (out/'episodes.json').write_text(json.dumps(episodes,indent=2))
    result={'scope':'pi05_LIBERO_simulation_scripted_interaction_not_real_BCI','episodes':episodes,
        'all_software_guards_passed':all(all(e['checks'].values()) for e in episodes),
        'all_episodes_without_exception':all(e['error'] is None for e in episodes),
        'nominal_task_successes':sum(e['task_success'] for e in episodes if e['episode'].startswith('nominal')),
        'nominal_episodes':2,'formal_success_rate_claim':False,'physical_robot_commands':0,'real_BCI_trials':0,
        'checkpoint':str(Path(a.checkpoint_root)/'model.safetensors'),
        'stats_sha256':hashlib.sha256(stats_path.read_bytes()).hexdigest(),'elapsed_s':time.perf_counter()-start,
        'original_framework_ontology_integrated':False,'G3_passed':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:client._ws.close()
