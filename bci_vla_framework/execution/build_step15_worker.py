"""Derive a second-backend worker while preserving the step14 frozen worker."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'execution/step14_libero_worker.py').read_text(encoding='utf-8')
def replace(old,new):
    global source
    assert source.count(old)==1,old
    source=source.replace(old,new)
replace('from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy',
    'from openpi_client.websocket_client_policy import WebsocketClientPolicy\nfrom openpi_client import image_tools')
replace('from bci_vla.pilot.libero_actions import decode_starvla_chunk','from bci_vla.adapters.vla.libero_pi05 import controller_actions')
replace("stats_path=Path(a.checkpoint_root)/'dataset_statistics.json';stats=json.loads(stats_path.read_text())['franka']['action']",
    "stats_path=Path(a.checkpoint_root)/'assets/physical-intelligence/libero/norm_stats.json'")
replace("                    payload={'examples':[{'image':[cv2.resize(img,(224,224)),cv2.resize(wrist,(224,224))],'lang':task.language}],\n                        'do_sample':False,'use_ddim':True,'num_ddim_steps':10}",
'''                    quat=np.asarray(obs['robot0_eef_quat'],dtype=float).copy()
                    quat[3]=np.clip(quat[3],-1,1);den=np.sqrt(1-quat[3]*quat[3])
                    axis=np.zeros(3) if np.isclose(den,0) else quat[:3]*2*np.arccos(quat[3])/den
                    payload={'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(img,224,224)),
                        'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist,224,224)),
                        'observation/state':np.concatenate([obs['robot0_eef_pos'],axis,obs['robot0_gripper_qpos']]),
                        'prompt':task.language}''')
replace('response=client.predict_action(payload)','response=client.infer(payload)')
replace("raw=np.asarray(response['data']['normalized_actions'],dtype=float)","raw=np.asarray(response['actions'],dtype=float)[None,...]")
replace('actions=decode_starvla_chunk(raw,stats)',
    "lower,upper=env.env.action_spec\n                    actions,conversion=controller_actions(raw,lower=lower,upper=upper,replan_steps=8)")
replace("'actions_sha256'=", "'actions_sha256'=") if False else None
replace("actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest())", "actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),controller_conversion=conversion)")
replace("'checkpoint':str(Path(a.checkpoint_root)/'checkpoints/steps_30000_pytorch_model.pt')", "'checkpoint':str(Path(a.checkpoint_root)/'model.safetensors')")
replace("'scope':'real_VLA_LIBERO_simulation_scripted_interaction_not_real_BCI'", "'scope':'pi05_LIBERO_simulation_scripted_interaction_not_real_BCI'")
replace('finally:client.close()','finally:client._ws.close()')
replace('endpoint=bool(env.check_success());env.close()',
    "endpoint=bool(env.check_success());g.invalidate();event('terminal_queue_cleared',episode=label,snapshot=g.snapshot());env.close()")
(ROOT/'execution/step15_libero_worker.py').write_text(source,encoding='utf-8')
print('Second-backend worker created; step14 source unchanged.')
