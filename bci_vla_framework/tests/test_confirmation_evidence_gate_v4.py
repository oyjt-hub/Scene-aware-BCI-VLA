from copy import deepcopy
import unittest
from bci_vla.calibration_v4.evidence_gate import confirmation_evidence_gate


class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.context = {'mapping_revision':1}
        self.config = {'confirmation_gate':{'mode':'consistent_recheck','consistent_selections':2,
                                           'correct_probes':2,'selection_max_age_s':10.}}
        self.public = {'context':self.context,'as_of_s':14.,'history':[],
                       'public_state':{'diagnostics':{'layout_id':'probe-2','receipts':[]}}}

    def event(self, role, end, choice='a', sequence=1, symbol='s0'):
        options = [{'option_id':'o0','kind':'OBJECT' if role!='PROBE' else 'DIAGNOSTIC',
                    'reference':'a','symbol':symbol},
                   {'option_id':'o1','kind':'OBJECT' if role!='PROBE' else 'DIAGNOSTIC',
                    'reference':'b','symbol':'s1' if symbol=='s0' else 's0'}]
        return {'event_id':str(sequence),'sequence':sequence,'ended_at_s':end,'valid':True,'role':role,
                'question':{'role':role,'context':deepcopy(self.context),'layout_id':'probe-2',
                            'options':options,'diagnostic_option_id':'o0' if role=='PROBE' else None},
                'evidence':{'selected_option_id':'o0' if choice=='a' else 'o1'}}

    def ready(self):
        self.public['history']=[self.event('TASK',7.25,sequence=1),self.event('INTENT_QUERY',9.25,sequence=2)]
        self.public['public_state']['diagnostics']['receipts']=[
            self.event('PROBE',11.25,sequence=3),self.event('PROBE',13.25,sequence=4,symbol='s1')]

    def gate(self):
        return confirmation_evidence_gate(self.public,'a',self.config)

    def test_complete_recent_certificate_allows_only_normal_confirmation_path(self):
        self.ready(); self.assertIsNone(self.gate())

    def test_three_check_contract_does_not_accept_a_two_check_certificate(self):
        self.ready();self.config['confirmation_gate']['correct_probes']=3
        self.assertEqual(self.gate()['action'],'PROBE')
        self.public['as_of_s']=16.
        self.public['public_state']['diagnostics']['receipts'].append(self.event('PROBE',15.25,sequence=5))
        self.assertIsNone(self.gate())

    def test_high_model_confidence_cannot_replace_two_matching_selections(self):
        self.ready();self.public['history'][0]['evidence']['selected_option_id']='o1'
        self.assertEqual(self.gate()['action'],'INTENT_QUERY')
        self.public['history']=self.public['history'][-1:]
        self.assertEqual(self.gate()['action'],'INTENT_QUERY')

    def test_startup_probes_do_not_certify_a_later_task(self):
        self.ready()
        for e in self.public['public_state']['diagnostics']['receipts']: e['ended_at_s']-=8
        self.assertEqual(self.gate()['action'],'PROBE')

    def test_wrong_and_no_response_diagnostics_pause_instead_of_cherry_picking_success(self):
        for role,choice in [('PROBE','b'),('NO_RESPONSE','a')]:
            self.ready();e=self.public['public_state']['diagnostics']['receipts'][0]
            e['role']=role;e['evidence']['selected_option_id']='o1' if choice=='b' else None
            self.assertEqual(self.gate()['action'],'WAIT')

    def test_stale_mapping_or_same_symbol_cannot_create_certificate(self):
        self.ready(); self.public['history'][0]['question']['context']={'mapping_revision':0}
        self.assertEqual(self.gate()['action'],'INTENT_QUERY')
        self.ready();self.public['as_of_s']=30.
        self.assertEqual(self.gate()['action'],'INTENT_QUERY')
        self.ready();self.public['as_of_s']=14.
        self.public['public_state']['diagnostics']['receipts'][1]=self.event('PROBE',13.25,sequence=4)
        self.assertEqual(self.gate()['action'],'PROBE')

    def test_duplicate_receipts_are_rejected(self):
        self.ready();r=self.public['public_state']['diagnostics']['receipts'];r[1]=deepcopy(r[0])
        with self.assertRaisesRegex(ValueError,'duplicate'):self.gate()

    def test_hidden_or_future_annotations_do_not_change_decision(self):
        self.ready();before=self.gate()
        self.public['audit_only']={'future_goal':'b','injected_success':.01}
        self.assertEqual(self.gate(),before)


if __name__=='__main__':unittest.main()
