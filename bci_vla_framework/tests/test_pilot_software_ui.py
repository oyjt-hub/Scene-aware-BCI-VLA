import unittest
from copy import deepcopy
from bci_vla.benchmark.offline import make_session,stamp
from bci_vla.protocols import ProtocolError
from bci_vla.pilot.software_ui import presentation,luminance,frame_timing
from bci_vla.pilot.software_flow import run_software_flow


class SoftwarePresentationTests(unittest.TestCase):
    def setUp(self):
        self.s=make_session('ui-check')
        self.q=self.s.begin_interaction('TASK_QUERY',now=stamp(0),monotonic_now=0).to_dict()

    def test_task_view_is_bound_to_request_without_authority(self):
        view=presentation(self.q,self.s.context,now=stamp(1))
        self.assertEqual(view['request_id'],self.q['event_id'])
        self.assertEqual(view['mode'],'TASK_QUERY');self.assertFalse(view['execution_authority'])

    def test_stale_context_rejected_before_display(self):
        context=dict(self.s.context);context['mapping_id']='stale'
        with self.assertRaises(ProtocolError):presentation(self.q,context,now=stamp(1))

    def test_expired_and_future_requests_rejected(self):
        for t in (-1,10,11):
            with self.subTest(t=t),self.assertRaises(ProtocolError):presentation(self.q,self.s.context,now=stamp(t))

    def test_probe_view_is_not_a_contract(self):
        s=make_session('probe-ui')
        q=s.begin_interaction('CHANNEL_PROBE',now=stamp(0),monotonic_now=0,prompt_symbol='0',presentation_ack=True,
            presented_at={'host_id':'ui','at':stamp(0),'offset_ms':0,'uncertainty_ms':0})
        view=presentation(q,s.context,now=stamp(0))
        self.assertIsNone(view['contract']);self.assertIsNotNone(view['probe_prompt'])

    def test_luminance_bounds_and_period(self):
        values=[luminance(i,frequency_hz=10,refresh_hz=60) for i in range(120)]
        self.assertTrue(all(0<=v<=255 for v in values));self.assertEqual(values[:6],values[6:12])

    def test_invalid_frequencies_rejected(self):
        for f in (0,-1,30,60,float('nan')):
            with self.subTest(f=f),self.assertRaises(ProtocolError):luminance(0,frequency_hz=f,refresh_hz=60)

    def test_drop_injection_detected_and_bad_clock_rejected(self):
        self.assertEqual(frame_timing([0,1/60,3/60],refresh_hz=60)['long_intervals'],1)
        for times in ([0,0],[1,0],[0,float('nan')]):
            with self.assertRaises(ProtocolError):frame_timing(times,refresh_hz=60)

    def test_full_three_mode_fake_execution_replays_without_backend(self):
        result,journal=run_software_flow()
        self.assertEqual(result['modes'],['TASK_QUERY','CHANNEL_PROBE','CONFIRM'])
        self.assertEqual(result['final_state'],'SUCCEEDED');self.assertEqual(result['fake_backend_submissions'],1)
        self.assertTrue(result['replay_matches_each_checkpoint']);self.assertFalse(result['G3_passed'])
        self.assertGreater(len(journal),0)


class SyntheticSignalTests(unittest.TestCase):
    def test_resampling_keeps_fixture_frequency_in_correct_timebase(self):
        import numpy as np
        from bci_vla.pilot.signal import prepare_decoder_window
        t=np.arange(4000)/1000
        y,meta=prepare_decoder_window(np.sin(2*np.pi*12*t)[None,:],source_rate_hz=1000)
        # Exclude edge transients; 12 Hz remains 12 Hz at the declared decoder rate.
        middle=y[0,250:750]
        peak=np.fft.rfftfreq(len(middle),1/250)[np.argmax(abs(np.fft.rfft(middle)))]
        self.assertEqual(peak,12);self.assertEqual(meta['decoder_samples'],1000)

    def test_resampling_rejects_high_frequency_alias_in_fixture(self):
        import numpy as np
        from bci_vla.pilot.signal import prepare_decoder_window
        t=np.arange(4000)/1000
        x=np.stack([np.sin(2*np.pi*12*t),np.sin(2*np.pi*180*t)])
        y,_=prepare_decoder_window(x,source_rate_hz=1000)
        # 180 Hz would alias to 70 Hz after a naive 4x stride. Measure that
        # component directly, excluding the unrelated slow filter transient.
        spectrum=abs(np.fft.rfft(y[:,250:750],axis=1))
        frequencies=np.fft.rfftfreq(500,1/250)
        alias=spectrum[1,np.argmin(abs(frequencies-70))]
        retained=spectrum[0,np.argmin(abs(frequencies-12))]
        self.assertLess(alias,.01*retained)
