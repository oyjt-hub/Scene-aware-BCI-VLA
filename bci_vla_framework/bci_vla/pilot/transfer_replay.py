"""Exercise unchanged core through two replay encodings; no physiological claim."""
from ..benchmark.offline import make_session,stamp,response,read,observation
from ..adapters.bci.packet_replay import PacketReplayTransport
from ..benchmark.runner import replay_without_backend

def core_transfer_trace(*,packetized):
    s=make_session('step15-replay-equivalence',prior=(1,0,0,0),fixed_transition=True)
    codec=PacketReplayTransport(known_target_probe_supported=True) if packetized else None
    checkpoints=[]
    for mode,begin,start,end,index in [('TASK_QUERY',0,1,2,1),('CHANNEL_PROBE',2.1,3,4,2),('CONFIRM',4.1,5,6,3)]:
        if mode=='CONFIRM':s.propose(read('examples/diagnosis/state.json')['contracts'][s.policy.goals[0]],now=stamp(begin))
        kwargs={} if mode!='CHANNEL_PROBE' else dict(prompt_symbol='0',presentation_ack=True,
            presented_at={'host_id':'ui','at':stamp(begin),'offset_ms':0,'uncertainty_ms':0})
        q=s.begin_interaction(mode,now=stamp(begin),monotonic_now=begin,**kwargs)
        event=response(q,'0',start=start,duration=1,index=index)
        if codec:
            raw=codec.encode(event);decoded=[]
            for offset in range(0,len(raw),17):decoded.extend(codec.feed(raw[offset:offset+17]))
            assert len(decoded)==1;event=decoded[0]
        result=s.ingest(event,now=stamp(end),monotonic_now=end);assert result.accepted
        checkpoints.append(s.snapshot())
    s.authorize(now=stamp(6.01));checkpoints.append(s.snapshot())
    s.dispatch('step15-fixture',observation(6.01),now=stamp(6.01));checkpoints.append(s.snapshot())
    assert replay_without_backend(s)
    return {'scope':'synthetic_transport_equivalence_not_second_real_BCI','packetized':packetized,
        'checkpoints':checkpoints,'journal':s.journal.to_jsonl(),'replay_matches':True,'physical_execution_allowed':False}
