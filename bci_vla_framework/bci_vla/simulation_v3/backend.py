"""Actual model/simulator adapter. No physical transport or device API."""
from copy import deepcopy
from ..protocols import freeze,thaw,fields,require,text,timestamp
from ..contracts.records import digest

class SimulationSpec:
    def __init__(self,data):
        require(data['scope']=='LIBERO_simulation' and data['physical_execution_allowed'] is False,'physical_backend_forbidden')
        require(data['interface_version']=='libero-model/3.0' and len(data['checkpoint_sha256'])==64,'checkpoint_bound_spec_required')
        self.data=freeze(data)

    def validate_observation(self,obs):
        fields(obs,{'interface_version','scene_id','scene_revision','captured_at','image_sha256','sim_state_sha256','scene_source'})
        require(obs['interface_version']==self.data['interface_version'] and obs['scene_source']=='privileged_simulator_state','observation_interface_mismatch')
        timestamp(obs['captured_at']);require(len(obs['image_sha256'])==2 and all(len(x)==64 for x in obs['image_sha256']) and len(obs['sim_state_sha256'])==64,'observation_digest_required')
        return deepcopy(obs)

class SimulationVLAAdapter:
    def __init__(self,spec,engine):
        self.spec=SimulationSpec(spec);self.engine=engine;self._submissions={};self.active=None

    def validate_observation(self,obs):
        self.spec.validate_observation(obs)
        require(self.engine.observation_digests()==(obs['image_sha256'],obs['sim_state_sha256']),'observation_not_current_simulator_frame')
        return deepcopy(obs)

    def submit(self,submission_id,contract_id,revision,instruction,observation):
        payload=dict(contract_id=contract_id,revision=revision,instruction=instruction,observation=observation)
        fingerprint=digest(payload)
        if submission_id in self._submissions:
            require(self._submissions[submission_id]['fingerprint']==fingerprint,'submission_id_payload_conflict')
            return self.query(submission_id)
        require(self.active is None,'unresolved_previous_simulation_submission')
        require(instruction in self.spec.data['instruction_to_object'],'instruction_outside_declared_task_registry')
        self.validate_observation(observation);self.engine.clear_queue()
        result=dict(submission_id=submission_id,contract_id=contract_id,revision=revision,instruction=instruction,
            status='ACCEPTED',execution_status='NOT_STARTED',stop_requested=False,last_feedback_id=None,actual_execution=False,
            physical_execution=False,execution_scope='LIBERO_simulation',reason='simulator_transport_ack_not_execution',duplicate=False)
        self._submissions[submission_id]={'fingerprint':fingerprint,'result':result,'steps':0};self.active=submission_id
        return deepcopy(result)

    def query(self,submission_id):
        require(submission_id in self._submissions,'unknown_simulation_submission');return deepcopy(self._submissions[submission_id]['result'])

    def advance(self,steps=8):
        require(self.active is not None,'no_authorized_simulation_submission')
        row=self._submissions[self.active];result=row['result']
        require(not result['stop_requested'] and result['execution_status'] not in {'COMPLETED','STOPPED','FAILED'},'simulation_action_after_terminal_or_stop')
        try:applied=self.engine.advance(result['instruction'],steps)
        except Exception:
            self.engine.clear_queue();result.update(execution_status='UNKNOWN',last_feedback_id=self.active+'-inference-unknown')
            return deepcopy(result)
        row['steps']+=applied;result['actual_execution']=applied>0 or result['actual_execution']
        if result['execution_status']!='STARTED':
            result['execution_status']='STARTED'
            result['last_feedback_id']=self.active+'-sim-started'
        return deepcopy(result)

    def finish(self):
        require(self.active is not None and self.engine.finished,'simulator_completion_required')
        row=self._submissions[self.active];result=row['result'];self.engine.clear_queue()
        result.update(execution_status='COMPLETED',last_feedback_id=self.active+'-simulator-completed')
        self.active=None;return deepcopy(result)

    def stop(self,submission_id):
        row=self._submissions[submission_id];result=row['result'];self.engine.clear_queue()
        result.update(stop_requested=True,execution_status='STOPPED',last_feedback_id=submission_id+'-simulator-stopped')
        self.active=None;return deepcopy(result)
