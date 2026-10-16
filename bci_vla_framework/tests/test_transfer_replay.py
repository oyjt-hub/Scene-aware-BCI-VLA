import json,unittest
from copy import deepcopy
from bci_vla.adapters.bci.packet_replay import PacketReplayTransport
from bci_vla.pilot.transfer_replay import core_transfer_trace
from bci_vla.benchmark.offline import make_session,stamp,response
from bci_vla.protocols import ProtocolError

class TransferReplayTests(unittest.TestCase):
    def event(self,probe=False):
        s=make_session('transport-check')
        kwargs={} if not probe else dict(prompt_symbol='0',presentation_ack=True,
            presented_at={'host_id':'ui','at':stamp(0),'offset_ms':0,'uncertainty_ms':0})
        q=s.begin_interaction('CHANNEL_PROBE' if probe else 'TASK_QUERY',now=stamp(0),monotonic_now=0,**kwargs)
        return response(q,'0',start=1,duration=1,index=1)

    def test_core_checkpoints_and_journal_payloads_same_across_transport(self):
        direct=core_transfer_trace(packetized=False);packet=core_transfer_trace(packetized=True)
        self.assertEqual(direct['checkpoints'],packet['checkpoints'])
        semantic=lambda text:[{k:r[k] for k in ['kind','at','payload']} for r in map(json.loads,text.splitlines())]
        self.assertEqual(semantic(direct['journal']),semantic(packet['journal']))

    def test_fragmentation_and_concatenation_preserve_event(self):
        t=PacketReplayTransport(known_target_probe_supported=True);e=self.event();raw=t.encode(e)
        self.assertEqual(t.feed(raw[:7]),[]);self.assertEqual(t.feed(raw[7:]+raw),[e,e])

    def test_capability_cannot_assume_motor_imagery_known_target_probe(self):
        t=PacketReplayTransport(known_target_probe_supported=False)
        with self.assertRaisesRegex(ProtocolError,'known_target_probe_unsupported'):t.encode(self.event(True))

    def test_live_data_cannot_be_relabelled_by_transport(self):
        t=PacketReplayTransport(known_target_probe_supported=True);e=self.event();e['payload']['sample_window']['source_kind']='live_bci'
        with self.assertRaises(ProtocolError):t.encode(e)

    def test_wrong_provenance_and_malformed_packets_rejected(self):
        for data in [b'not-json\n',b'{"schema_version":"live"}\n',b'\xff\n']:
            t=PacketReplayTransport(known_target_probe_supported=True)
            with self.assertRaises(ProtocolError):t.feed(data)

    def test_buffer_limit_prevents_unterminated_packet_growth(self):
        t=PacketReplayTransport(known_target_probe_supported=True)
        with self.assertRaises(ProtocolError):t.feed(b'x'*262145)
        self.assertEqual(len(t.buffer),0)
