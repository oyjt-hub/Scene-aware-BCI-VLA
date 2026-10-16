"""Freeze a full-framework backend/codec migration run without editing 14C."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1];E=R/'execution';N=R/'bci_vla/simulation_transfer_v4'
assert not N.exists();N.mkdir()
(N/'__init__.py').write_text('"""Simulation-only backend and synthetic transport migration."""\n')
for name in ['factory','interaction','recovery']:
 t=(R/f'bci_vla/simulation_v3/{name}.py').read_text()
 for shared in ['backend','session','records','outcomes']:
  t=t.replace('from .'+shared+' import','from ..simulation_v3.'+shared+' import')
 if name=='factory':
  t=t.replace('class SimulationController(RuntimeController):', '''TRANSPORT='direct'
EXECUTOR_ID='Pi05-LIBERO-v4'

class SimulationController(RuntimeController):
    def roundtrip(self,event):
        from ..adapters.bci.packet_replay import PacketReplayTransport
        codec=PacketReplayTransport(known_target_probe_supported=True)
        raw=codec.encode(event);received=[]
        for offset in range(0,len(raw),17):received.extend(codec.feed(raw[offset:offset+17]))
        assert len(received)==1 and received[0]==event
        self.packet_events=getattr(self,'packet_events',0)+1
        return received[0]

    def begin(self,action):
        from ..protocols import Event
        q,start,end=super().begin(action)
        if self.transport=='packet':q=Event.parse(self.roundtrip(q.to_dict()))
        return q,start,end''')
  t=t.replace('result=self.session.ingest(event,',"if self.transport=='packet':event=self.roundtrip(event)\n        result=self.session.ingest(event,")
  t=t.replace("executor_id='StarVLA-LIBERO-v3'","executor_id=EXECUTOR_ID")
  t=t.replace("backend_id='StarVLA-LIBERO-v3'","backend_id=outcomes['executor_id']")
  t=t.replace('controller.now=now','controller.now=now\n    controller.transport=TRANSPORT')
 if name=='interaction':t=t.replace("'executor_id':'StarVLA-LIBERO-v3'","'executor_id':session.config['backend_id']")
 (N/(name+'.py')).write_text(t,encoding='utf-8')
engine=(E/'step14C_engine.py').read_text().replace('Real StarVLA','Real pi05')
engine=engine.replace('from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy','from openpi_client.websocket_client_policy import WebsocketClientPolicy\nfrom openpi_client import image_tools')
engine=engine.replace('from bci_vla.pilot.libero_actions import decode_starvla_chunk','from bci_vla.adapters.vla.libero_pi05 import controller_actions')
engine=engine.replace("self.stats=json.loads((Path(checkpoint_root)/'dataset_statistics.json').read_text())['franka']['action']","self.stats_path=Path(checkpoint_root)/'assets/physical-intelligence/libero/norm_stats.json'")
start=engine.index("                payload={'examples'");end=engine.index("                self.log('real_model_chunk'",start)
engine=engine[:start]+'''                quat=np.asarray(self.obs['robot0_eef_quat'],dtype=float).copy()
                quat[3]=np.clip(quat[3],-1,1);den=np.sqrt(1-quat[3]*quat[3])
                axis=np.zeros(3) if np.isclose(den,0) else quat[:3]*2*np.arccos(quat[3])/den
                payload={'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(img,224,224)),
                    'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist,224,224)),
                    'observation/state':np.concatenate([self.obs['robot0_eef_pos'],axis,self.obs['robot0_gripper_qpos']]),'prompt':instruction}
                start=time.perf_counter();raw=np.asarray(self.client.infer(payload)['actions'],dtype=float)[None,...]
                lower,upper=self.env.env.action_spec
                actions,conversion=controller_actions(raw,lower=lower,upper=upper,replan_steps=8);self.queue.extend(actions.tolist())
''' +engine[end:]
engine=engine.replace("image_sha256=self.observation_digests()[0])","image_sha256=self.observation_digests()[0],controller_conversion=conversion)")
engine=engine.replace('self.client.close()','self.client._ws.close()')
(E/'step15C_pi05_engine.py').write_text(engine,encoding='utf-8')
worker=(E/'step14C_worker.py').read_text().replace('from step14C_engine import LiberoEngine',"import os\nfrom step15C_pi05_engine import LiberoEngine\nfrom bci_vla.simulation_transfer_v4 import factory as migration_factory")
worker=worker.replace('bci_vla.simulation_v3.factory','bci_vla.simulation_transfer_v4.factory').replace('bci_vla.simulation_v3.interaction','bci_vla.simulation_transfer_v4.interaction').replace('bci_vla.simulation_v3.recovery','bci_vla.simulation_transfer_v4.recovery')
worker=worker.replace("'14C_run_plan.json'","'15C_run_plan.json'")
worker=worker.replace("image_pipeline='RGB uint8 2views 256->224 rotation180',action_pipeline='franka stats + original LIBERO gripper conversion'","image_pipeline='RGB uint8 2views 256->224 rotation180 resize_with_pad',action_pipeline='pi05 already unnormalized + explicit controller clipping'")
worker=worker.replace("identifier=case['id'];engine.reset", "migration_factory.TRANSPORT=case['transport']\n        identifier=case['id'];engine.reset")
worker=worker.replace("case['id'] in {'changed_goal','changed_channel','joint_alias_conflict'}","case['base_case'] in {'changed_goal','changed_channel','joint_alias_conflict'}")
worker=worker.replace("row=dict(id=identifier,", "row=dict(id=identifier,transport=case['transport'],base_case=case['base_case'],packet_events=sum(getattr(getattr(x,'_controller',None),'packet_events',0) for x in []),")
# Capture all controllers indirectly at creation, including repair sessions.
worker=worker.replace("records=[];episodes=[];started=time.perf_counter()", "records=[];episodes=[];started=time.perf_counter()\ncontrollers=[]\noriginal_runtime=migration_factory.make_runtime\ndef tracked_runtime(*args,**kwargs):\n    session,controller=original_runtime(*args,**kwargs);controllers.append(controller);return session,controller\nmigration_factory.make_runtime=tracked_runtime\nmake_runtime=tracked_runtime\nimport bci_vla.simulation_transfer_v4.recovery as recovery_module\nrecovery_module.make_runtime=tracked_runtime")
worker=worker.replace("migration_factory.TRANSPORT=case['transport']", "controllers.clear()\n        migration_factory.TRANSPORT=case['transport']")
worker=worker.replace("packet_events=sum(getattr(getattr(x,'_controller',None),'packet_events',0) for x in [])", "packet_events=sum(getattr(c,'packet_events',0) for c in controllers)")
(E/'step15C_worker.py').write_text(worker,encoding='utf-8')
plan=json.loads((E/'14C_run_plan_v3.json').read_text());binding=json.loads((R/'configs/transfer/step15_binding.json').read_text())['backend_B']
for k in ['reuse_calibration_json','reused_calibration_sha256','source_fingerprints']:plan.pop(k,None)
plan.update(scope='full_framework_pi05_two_synthetic_transports_no_real_BCI',checkpoint_sha256=binding['weight_sha256'],normalization_sha256=binding['norm_stats_sha256'])
cases=plan['cases'];plan['cases']=[dict(c,id=transport+'-'+c['id'],base_case=c['id'],transport=transport) for transport in ['direct','packet'] for c in cases]
plan['transfer_claims']=dict(real_second_BCI=False,formal_2x2=False,training_scope_equivalence=False,core_threshold_tuning=False,heldout_codec_backend_combination='packet-pi05; engineering functional only; no parameter fitting on this cell')
(E/'15C_run_plan_v1.json').write_text(json.dumps(plan,indent=2)+'\n')
o=(E/'step15_pi05_orchestrator.py').read_text().replace('16815','16817').replace('step15_libero_worker.py','step15C_worker.py').replace('timeout=660','timeout=1100')
(E/'step15C_orchestrator.py').write_text(o,encoding='utf-8')
print('Prepared isolated full pi05 migration with 20 functional cases and six independent trajectories.')
