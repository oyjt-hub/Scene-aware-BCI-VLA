"""Bounded alternate replay transport. Not a second physiological BCI decoder."""
import json
from ...protocols import Event,fields,require

class PacketReplayTransport:
    def __init__(self,*,known_target_probe_supported):
        require(type(known_target_probe_supported) is bool,'explicit_probe_capability_required')
        self.supports_probe=known_target_probe_supported;self.buffer=bytearray()

    def encode(self,event):
        event=Event.parse(event).to_dict();self._check(event)
        return (json.dumps({'schema_version':'packet-replay/1.0','provenance':'synthetic_codec_fixture','event':event},ensure_ascii=False)+'\n').encode('utf-8')

    def _check(self,event):
        require(event['mode']!='CHANNEL_PROBE' or self.supports_probe,'known_target_probe_unsupported')
        if event['kind']=='RESPONSE':
            window=event['payload']['sample_window']
            require(window is None or window['source_kind']=='offline_replay','live_evidence_not_replay_transport')

    def feed(self,data):
        require(type(data) is bytes,'byte_packet_required')
        require(len(self.buffer)+len(data)<=262144,'packet_buffer_limit')
        self.buffer.extend(data);events=[]
        while b'\n' in self.buffer:
            line,_,tail=self.buffer.partition(b'\n');self.buffer=bytearray(tail)
            try:packet=json.loads(line.decode('utf-8'))
            except (ValueError,UnicodeError):
                self.buffer.clear();require(False,'malformed_replay_packet')
            fields(packet,{'schema_version','provenance','event'})
            require(packet['schema_version']=='packet-replay/1.0' and packet['provenance']=='synthetic_codec_fixture','unsupported_packet_provenance')
            event=Event.parse(packet['event']).to_dict();self._check(event);events.append(event)
        return events
