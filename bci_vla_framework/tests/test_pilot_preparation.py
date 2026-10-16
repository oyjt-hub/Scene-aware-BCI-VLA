"""Pilot metadata checks; all data here are explicit software fixtures."""
from copy import deepcopy
import unittest
from bci_vla.protocols import ProtocolError
from bci_vla.pilot.validation import validate_sampling_chain,validate_record,summarize_records
from bci_vla.models.evidence import summarize_timing


def window(rate=250,count=1000):
    return {'sample_rate_hz':rate,'sample_start':0,'sample_end':count,'start_s':0,'end_s':4,
        'clock_id':'fixture-reference','channels':['O1','Oz','O2'],'recording_ref':'fixture-recording'}
def transform():
    return {'operation':'resample','input_rate_hz':1000,'output_rate_hz':250,
        'anti_alias_evidence_ref':'fixture-filter-trace','trace_ref':'fixture-preprocess-trace'}
def record(**updates):
    v={'record_id':'fixture-1','participant_alias':'fixture-user','session_group':'fixture-session','recording_group':'fixture-recording',
        'split':'calibration','mode':'TASK_QUERY','source_kind':'synthetic_fixture','label_source':'independent_task_card',
        'label_validity':'known','intended_symbol':'0','probe_prompt':None,'compliance_annotation':None,
        'response_status':'OBSERVATION','observed_symbol':'0','window':window(),'source_ref':'synthetic-software-fixture',
        'stimulus_context':{'layout_id':'fixture','mapping_id':'fixture','stimulus_version':'fixture','decoder_version':'fixture'},
        'calibration_condition_ref':'independent-fixture-condition'}
    v.update(updates);return v


class SamplingTests(unittest.TestCase):
    def test_1000_to250_consistent_metadata_does_not_claim_raw_pipeline_verified(self):
        r=validate_sampling_chain(window(1000,4000),window(),transform())
        self.assertTrue(r['declared_transform_consistent']);self.assertFalse(r['raw_signal_verified'])
        self.assertEqual(r['processed_samples'],1000)

    def test_double_decimation_interpreted_at250_is_rejected(self):
        with self.assertRaisesRegex(ProtocolError,'sample_count_time_rate_mismatch'):
            validate_sampling_chain(window(1000,4000),window(250,250),transform())

    def test_rates_cannot_be_changed_without_transform_evidence(self):
        t=transform();t['output_rate_hz']=1000
        with self.assertRaises(ProtocolError):validate_sampling_chain(window(1000,4000),window(),t)
        t=transform();t['anti_alias_evidence_ref']=None
        with self.assertRaises(ProtocolError):validate_sampling_chain(window(1000,4000),window(),t)

    def test_unaligned_clocks_and_channel_order_are_rejected(self):
        for change in [{'clock_id':'other-host'},{'channels':['Oz','O1','O2']}]:
            b=window();b.update(change)
            with self.assertRaises(ProtocolError):validate_sampling_chain(window(1000,4000),b,transform())

    def test_identity_must_preserve_sample_count_and_rate(self):
        t={'operation':'identity','input_rate_hz':250,'output_rate_hz':250,'anti_alias_evidence_ref':None,'trace_ref':'fixture'}
        self.assertTrue(validate_sampling_chain(window(),window(),t)['declared_transform_consistent'])
        t['operation']='downsample_guess'
        with self.assertRaises(ProtocolError):validate_sampling_chain(window(),window(),t)


class PilotRecordTests(unittest.TestCase):
    def test_fixture_rejected_as_real_pilot_by_default(self):
        with self.assertRaisesRegex(ProtocolError,'synthetic_not_real_pilot'):validate_record(record())
        self.assertEqual(summarize_records([record()],allow_synthetic=True)['scope'],'synthetic_software_check')

    def test_independent_labels_required_not_algorithm_self_labels(self):
        with self.assertRaises(ProtocolError):validate_record(record(label_source='algorithm_prediction'),allow_synthetic=True)

    def test_probe_effective_counts_do_not_identify_compliance_and_decoder_separately(self):
        row=record(mode='CHANNEL_PROBE',label_source='independent_probe_annotation',probe_prompt='0',
            compliance_annotation='UNKNOWN',intended_symbol=None,label_validity='unknown')
        result=summarize_records([row],allow_synthetic=True)
        self.assertEqual(result['conditional_intended_counts'],[])
        self.assertEqual(result['effective_probe_counts'][0]['n'],1)
        self.assertFalse(result['decomposition_C_probe_and_compliance_identified'])

    def test_probe_annotation_cannot_be_replaced_by_prompt(self):
        row=record(mode='CHANNEL_PROBE',label_source='independent_probe_annotation',probe_prompt='0',compliance_annotation='1')
        with self.assertRaisesRegex(ProtocolError,'probe_intended_must_match_independent_annotation'):validate_record(row,allow_synthetic=True)

    def test_no_response_is_not_transport_timeout(self):
        a=record(response_status='NO_RESPONSE',observed_symbol=None)
        b=record(record_id='fixture-2',response_status='TIMEOUT',observed_symbol=None,recording_group='second')
        b['window']['recording_ref']='second';r=summarize_records([a,b],allow_synthetic=True)
        self.assertEqual(r['records'],2);self.assertEqual(r['conditional_intended_counts'][0]['counts']['NO_RESPONSE'],1)
        self.assertEqual(r['status_counts']['TIMEOUT'],1)

    def test_formal_test_cannot_be_used_as_calibration(self):
        with self.assertRaises(ProtocolError):validate_record(record(split='formal_test'),allow_synthetic=True)

    def test_timeout_without_captured_samples_is_retained_in_denominator(self):
        r=summarize_records([record(response_status='TIMEOUT',observed_symbol=None,window=None)],allow_synthetic=True)
        self.assertEqual(r['records'],1);self.assertEqual(r['status_counts']['TIMEOUT'],1)
        self.assertEqual(r['records_without_sampling_window'],1);self.assertEqual(r['conditional_intended_counts'],[])

    def test_observation_and_no_response_need_a_measured_window(self):
        for status,symbol in [('OBSERVATION','0'),('NO_RESPONSE',None)]:
            with self.assertRaises(ProtocolError):validate_record(record(response_status=status,observed_symbol=symbol,window=None),allow_synthetic=True)

    def test_recording_group_cannot_cross_calibration_and_holdout(self):
        a=record();b=record(record_id='fixture-2',split='calibration_holdout');b['window']['recording_ref']='different-file'
        with self.assertRaisesRegex(ProtocolError,'recording_group_split_leakage'):summarize_records([a,b],allow_synthetic=True)

    def test_overlap_and_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ProtocolError,'duplicate_pilot_record'):summarize_records([record(),record()],allow_synthetic=True)
        b=record(record_id='fixture-2')
        with self.assertRaisesRegex(ProtocolError,'overlapping_recording_samples'):summarize_records([record(),b],allow_synthetic=True)

    def test_distinct_durations_stay_separate(self):
        a=record();b=record(record_id='fixture-2',recording_group='second')
        b['window'].update(sample_end=500,end_s=2,recording_ref='second')
        self.assertEqual(len(summarize_records([a,b],allow_synthetic=True)['conditional_intended_counts']),2)

    def test_distinct_menus_and_decoders_do_not_pool_counts(self):
        a=record();b=record(record_id='fixture-2',recording_group='second')
        b['window']['recording_ref']='second';b['stimulus_context']['mapping_id']='another-menu'
        self.assertEqual(len(summarize_records([a,b],allow_synthetic=True)['conditional_intended_counts']),2)

    def test_empty_real_data_keeps_absent_metrics_unfitted(self):
        r=summarize_records([])
        self.assertEqual(r['scope'],'not_collected');self.assertEqual(r['real_records'],0)
        self.assertFalse(r['real_channel_model_fitted']);self.assertFalse(r['transfer_validated']);self.assertFalse(r['G3_passed'])

    def test_missing_timing_phases_not_silently_zero(self):
        r=summarize_timing([{'phase':'acquisition','clock_id':'fixture','start_s':1,'end_s':2,'user_busy':True}],
            required_phases=['reading','acquisition'],clock_id='fixture',window_start_s=0,window_end_s=3)
        self.assertIsNone(r['user_busy_s']);self.assertEqual(r['missing_phases'],['reading']);self.assertEqual(r['total_elapsed_s'],3)


class ReceiverAndSignalTests(unittest.TestCase):
    def test_fragmented_frames_are_not_lost_or_reordered(self):
        import struct
        from bci_vla.pilot.neuracle_stream import FloatFrameParser
        p=FloatFrameParser(2);data=struct.pack('<4f',1.,2.,3.,4.)
        self.assertEqual(p.feed(data[:5]),[])
        self.assertEqual(p.feed(data[5:13]),[(1.,2.)])
        self.assertEqual(p.feed(data[13:]),[(3.,4.)]);self.assertEqual(p.frames,2);self.assertEqual(p.tail,b'')

    def test_nonfinite_stream_values_are_reported_not_treated_as_valid_signal(self):
        import struct
        from bci_vla.pilot.neuracle_stream import FloatFrameParser
        p=FloatFrameParser(1);p.feed(struct.pack('<f',float('nan')));self.assertEqual(p.nonfinite,1)

    def test_4000_raw_points_become1000_not250(self):
        import numpy as np
        from bci_vla.pilot.signal import prepare_decoder_window
        x=np.sin(2*np.pi*12*np.arange(4000)/1000)[None,:]
        y,meta=prepare_decoder_window(x,source_rate_hz=1000)
        self.assertEqual(y.shape,(1,1000));self.assertEqual(meta['source_duration_s'],meta['decoder_duration_s'])
        self.assertFalse(meta['real_decoder_accuracy_validated'])

    def test_slice_indices_use_source_timebase_before_resampling(self):
        import numpy as np
        from bci_vla.pilot.signal import prepare_decoder_window
        x=np.zeros((2,4000));y,m=prepare_decoder_window(x,source_rate_hz=1000,start_s=.5,duration_s=2)
        self.assertEqual(m['source_start_sample'],500);self.assertEqual(m['source_end_sample'],2500)
        self.assertEqual(y.shape,(2,500))

    def test_already250Hz_is_not_decimated_again(self):
        import numpy as np
        from bci_vla.pilot.signal import prepare_decoder_window
        y,m=prepare_decoder_window(np.zeros((1,1000)),source_rate_hz=250)
        self.assertEqual(y.shape[1],1000);self.assertEqual(m['decoder_duration_s'],4)

    def test_signal_window_outside_data_or_nonfinite_is_rejected(self):
        import numpy as np
        from bci_vla.pilot.signal import prepare_decoder_window
        with self.assertRaises(ProtocolError):prepare_decoder_window(np.zeros((1,1000)),source_rate_hz=1000,duration_s=4)
        x=np.zeros((1,1000));x[0,3]=float('inf')
        with self.assertRaises(ProtocolError):prepare_decoder_window(x,source_rate_hz=1000)


if __name__=='__main__':unittest.main()
