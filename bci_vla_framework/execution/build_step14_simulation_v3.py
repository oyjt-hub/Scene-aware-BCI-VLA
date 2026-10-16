"""Versioned simulation lifecycle, preserving every frozen source and its algebra."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];dest=ROOT/'bci_vla/simulation_v3';dest.mkdir(exist_ok=True)
(dest/'__init__.py').write_text('"""Actual VLA in LIBERO, replay EEG only, never physical robot execution."""\n',encoding='utf-8')
changes=[]
for name in ['records','session','replay']:
    original=ROOT/'bci_vla/contracts'/f'{name}.py';text=original.read_text(encoding='utf-8')
    source_hash=hashlib.sha256(original.read_bytes()).hexdigest()
    text=text.replace('offline_fake_only','simulation_model_only')
    text=text.replace("'simulation_fixture_annotation'","'simulator_state_annotation'")
    text=text.replace("trajectory.data['provenance']=='simulation_fixture'","trajectory.data['provenance']=='independent_annotation'")
    text=text.replace("v['provenance']=='simulation_fixture' and v['executor_id']==c['backend_id'] and v['checkpoint_ref'] is None",
        "v['provenance']=='independent_annotation' and v['executor_id']==c['backend_id'] and v['checkpoint_ref']==self.backend.spec.data['checkpoint_sha256']")
    text=text.replace('v["provenance"] == "simulation_fixture" and v["executor_id"] == c["backend_id"] and\n                v["checkpoint_ref"] is None',
        'v["provenance"] == "independent_annotation" and v["executor_id"] == c["backend_id"] and\n                v["checkpoint_ref"] == self.backend_spec["checkpoint_sha256"]')
    if name=='session':
        text=text.replace('from ..adapters.vla import FakeVLAAdapter, DryRunVLAAdapter','from .backend import SimulationVLAAdapter')
        text=text.replace("type(backend) in {FakeVLAAdapter,DryRunVLAAdapter}","type(backend) is SimulationVLAAdapter")
        text=text.replace("require(reply.get('actual_execution') is False,'fake_feedback_required')",
            "require(reply.get('physical_execution') is False and reply.get('execution_scope')=='LIBERO_simulation','simulation_feedback_required')")
    elif name=='replay':
        text=text.replace('from ..models.consequence import OutcomeTable, ObservedTrajectory, evaluate_trajectory',
            'from ..models.consequence import ObservedTrajectory, evaluate_trajectory\nfrom .outcomes import OutcomeTable\nfrom .backend import SimulationSpec')
        text=text.replace('require(self.backend_spec["scope"] == "offline_fixture" and\n                self.backend_spec["deployment_allowed"] is False and\n                self.backend_spec["interface_version"] == "fake-vla/1.0", "live_interface_unverified")','SimulationSpec(self.backend_spec)')
        text=text.replace('require(event["actual_execution"] is False, "offline_feedback_only")',
            'require(event["physical_execution"] is False and event["execution_scope"] == "LIBERO_simulation", "simulation_feedback_only")')
        start=text.index('        """Validate the declared fixture interface locally; create no adapter."""')
        end=text.index('\n    def submission_intent',start)
        text=text[:start]+'''        SimulationSpec(self.backend_spec).validate_observation(observation)
        _equal((observation['scene_id'],observation['scene_revision']),
               (self.context['scene_id'],self.context['scene_revision']),'observation_scene_mismatch')
        age=(now-timestamp(observation['captured_at'])).total_seconds()
        require(0<=age<=self.config['max_observation_age_s'],'observation_stale_or_future')
'''+text[end:]
    (dest/f'{name}.py').write_text(text,encoding='utf-8')
    changes.append({'source':str(original),'source_sha256':source_hash,'versioned_copy':str(dest/f'{name}.py')})
source=(ROOT/'bci_vla/models/consequence.py').read_text(encoding='utf-8')
header='import math\nfrom ..protocols import fields,freeze,thaw,json_object,require,text,number\nfrom ..models.consequence import COMPONENTS\n'
body=source[source.index('class OutcomeTable:'):]
body=body.replace('v["provenance"]=="illustrative_not_fitted"','v["provenance"]=="empirical_simulator_calibration"')
(dest/'outcomes.py').write_text(header+body,encoding='utf-8')
(ROOT/'execution/14C_version_delta.json').write_text(json.dumps({'scope':'versioned_simulation_only_session',
    'unchanged':['joint filter','protocol schemas','confirmation and authorization algebra','runtime_v2 Bellman solver','threshold0.8 and age2s'],
    'changes':['real simulator backend spec accepted','simulation scope instead of fake','bound checkpoint and independent simulator-state endpoint','empirical simulator outcome provenance'],
    'copies':changes},indent=2)+'\n',encoding='utf-8')
print('Versioned simulation lifecycle generated; old files unchanged.')
