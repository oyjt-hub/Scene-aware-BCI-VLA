"""Step11 mechanism tests. Stubs are explicit; no learned capability claims."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from bci_vla.high_level_v4.authorization import HighLevelSession,AuthorizationError,parse_task

ROOT=Path(__file__).resolve().parents[1]

def public():
    path=ROOT/'execution/v4/09_binding_data_v2/validation/public_input.jsonl'
    return json.loads(path.read_text(encoding='utf-8').splitlines()[0])

class StubBackend:
    def __init__(self):
        self.model_binding={'base':'test-only','connector':'test-only','language_lora':'test-only'}
        self.calls=[];self.resets=0;self.text='Pick up OBJ_0.';self.eos=True;self.pointer_wrong=False
    def reset_generation(self):self.resets+=1
    def generate(self,p,mode,gripper,*,confirmed_task=None):
        self.calls.append((deepcopy(p),mode,gripper,deepcopy(confirmed_task)))
        ids=[o['instance_id'] for o in p['scene']['objects']]
        probs={key:.01 for key in ids};probs[ids[1 if self.pointer_wrong else 0]]=.98
        return {'raw_text':self.text,'stopped_at_eos':self.eos,'model_binding':deepcopy(self.model_binding),
                'candidate_probabilities':probs,'outside_probability':.01}

class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.p=public();self.now=self.p['as_of_s'];self.b=StubBackend();self.s=HighLevelSession(self.b)
    def main_confirm(self):
        main=self.s.propose_main(self.p,now=self.now)
        self.s.confirm(self.p,proposal_sha256=main['sha256'],receipt_id='main',accepted=True,now=self.now+.1)
        return main
    def sub_confirm(self):
        main=self.main_confirm()
        sub=self.s.propose_subtask(self.p,now=self.now+.2,gripper_state='empty')
        self.s.confirm(self.p,proposal_sha256=sub['sha256'],receipt_id='sub',accepted=True,now=self.now+.3)
        return main,sub
    def test_closed_loop_same_backend_separate_modes(self):
        self.sub_confirm();result=self.s.consume_virtual(self.p,now=self.now+.4)
        self.assertEqual([c[1] for c in self.b.calls],['main_task','next_subtask'])
        self.assertEqual(self.b.calls[-1][3]['confirmation_receipt_id'],'main')
        self.assertFalse(result['execution_authorized']);self.assertFalse(result['VLA_called'])
        self.assertIsNone(result['joint_action']);self.assertTrue(result['simulation_authorized'])
    def test_parent_does_not_authorize_subtask(self):
        self.main_confirm()
        with self.assertRaisesRegex(AuthorizationError,'per_subtask'):self.s.consume_virtual(self.p,now=self.now+.2)
    def test_subtask_proposal_does_not_authorize_itself(self):
        self.main_confirm();self.s.propose_subtask(self.p,now=self.now+.2,gripper_state='empty')
        with self.assertRaises(AuthorizationError):self.s.consume_virtual(self.p,now=self.now+.3)
    def test_single_use(self):
        self.sub_confirm();self.s.consume_virtual(self.p,now=self.now+.4)
        with self.assertRaises(AuthorizationError):self.s.consume_virtual(self.p,now=self.now+.5)
    def test_old_confirmation(self):
        main=self.s.propose_main(self.p,now=self.now);self.s.propose_main(self.p,now=self.now+.1)
        with self.assertRaisesRegex(AuthorizationError,'stale_confirmation'):
            self.s.confirm(self.p,proposal_sha256=main['sha256'],receipt_id='old',accepted=True,now=self.now+.2)
        self.assertIsNone(self.s.main);self.assertFalse(self.s.queue)
    def test_receipt_replay(self):
        self.main_confirm();sub=self.s.propose_subtask(self.p,now=self.now+.2,gripper_state='empty')
        with self.assertRaisesRegex(AuthorizationError,'replayed_receipt'):
            self.s.confirm(self.p,proposal_sha256=sub['sha256'],receipt_id='main',accepted=True,now=self.now+.3)
    def test_user_rejection(self):
        main=self.s.propose_main(self.p,now=self.now)
        with self.assertRaisesRegex(AuthorizationError,'rejected_by_user'):
            self.s.confirm(self.p,proposal_sha256=main['sha256'],receipt_id='no',accepted=False,now=self.now+.1)
    def test_missing_destination(self):
        self.b.text='Put OBJ_0 in .'
        with self.assertRaisesRegex(AuthorizationError,'missing_slot'):self.s.propose_main(self.p,now=self.now)
        self.assertTrue(any(e['event']=='VLM_RAW' for e in self.s.journal))
    def test_hallucinated_object(self):
        self.b.text='Pick up OBJ_99.'
        with self.assertRaisesRegex(AuthorizationError,'hallucinated'):self.s.propose_main(self.p,now=self.now)
    def test_unauthorized_code(self):
        self.b.text='robot.move_joints([1,2,3])'
        with self.assertRaises(AuthorizationError):self.s.propose_main(self.p,now=self.now)
    def test_pointer_disagreement_not_repaired(self):
        self.b.pointer_wrong=True
        with self.assertRaisesRegex(AuthorizationError,'pointer_text_disagreement'):self.s.propose_main(self.p,now=self.now)
    def test_truncated_generation(self):
        self.b.eos=False
        with self.assertRaisesRegex(AuthorizationError,'truncated'):self.s.propose_main(self.p,now=self.now)
    def test_expiry_clears_queue(self):
        self.sub_confirm()
        with self.assertRaises(AuthorizationError):self.s.consume_virtual(self.p,now=self.now+11)
        self.assertFalse(self.s.queue);self.assertIsNone(self.s.main)
    def test_stop_and_cancel_clear_queue(self):
        for control in ('STOP','CANCEL'):
            self.setUp();self.sub_confirm();calls=len(self.b.calls);resets=self.b.resets;self.s.control(control)
            self.assertFalse(self.s.queue);self.assertGreater(self.b.resets,resets)
            with self.assertRaises(AuthorizationError):self.s.consume_virtual(self.p,now=self.now+.4)
            self.assertEqual(len(self.b.calls),calls)
    def test_mapping_scene_target_changes_clear_queue(self):
        for mutate in ('mapping','scene','evidence'):
            self.setUp();self.sub_confirm();changed=deepcopy(self.p)
            if mutate=='mapping':changed['context']['mapping_revision']+=1
            elif mutate=='scene':changed['context']['scene_revision']+=1
            else:changed['history'][0]['evidence']['selected_option_id']='o1'
            with self.assertRaises(ValueError):self.s.consume_virtual(changed,now=self.now+.4)
            self.assertFalse(self.s.queue)
    def test_model_change_revokes(self):
        self.sub_confirm();self.b.model_binding['connector']='changed'
        with self.assertRaisesRegex(AuthorizationError,'model_changed'):self.s.consume_virtual(self.p,now=self.now+.4)
        self.assertFalse(self.s.queue)
    def test_clock_reverse_revokes(self):
        self.sub_confirm()
        with self.assertRaisesRegex(AuthorizationError,'clock_reversed'):self.s.consume_virtual(self.p,now=self.now)
        self.assertFalse(self.s.queue)
    def test_unconfirmed_parent_no_model_call(self):
        with self.assertRaises(AuthorizationError):self.s.propose_subtask(self.p,now=self.now,gripper_state='empty')
        self.assertFalse(self.b.calls)
    def test_conflicts_rejected_before_model(self):
        supers=[json.loads(l) for l in (ROOT/'execution/v4/09_binding_data_v2/validation/supervision.jsonl').read_text().splitlines()]
        pubs=[json.loads(l) for l in (ROOT/'execution/v4/09_binding_data_v2/validation/public_input.jsonl').read_text().splitlines()]
        count=0
        for p,label in zip(pubs,supers):
            if label['family']!='conflicting_declarations':continue
            s=HighLevelSession(self.b)
            with self.assertRaises(AuthorizationError):s.propose_main(p,now=p['as_of_s'])
            count+=1
        self.assertEqual(count,8);self.assertFalse(self.b.calls)
    def test_caller_cannot_mutate_bound_proposal(self):
        proposal=self.s.propose_main(self.p,now=self.now);proposal['source_id']='forged'
        self.assertNotEqual(self.s.pending['source_id'],'forged')
    def test_upstream_alias_spelling_is_bijective(self):
        self.assertEqual(parse_task('pick up obj 0',self.p),parse_task('Pick up OBJ_0.',self.p))
        with self.assertRaises(AuthorizationError):parse_task('pick up obj 99',self.p)
    def test_only_verified_subtask_phase(self):
        self.main_confirm()
        with self.assertRaisesRegex(AuthorizationError,'unverified_subtask_phase'):
            self.s.propose_subtask(self.p,now=self.now+.2,gripper_state='holding_selected')

if __name__=='__main__':unittest.main()
