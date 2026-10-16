"""Offline presentation model. No display loop, device or stimulus I/O."""
import math
from ..protocols import Event, require

CONTEXT=('session_id','task_id','task_revision','scene_id','scene_revision','mapping_id','stimulus_version','channel_profile_id')
BANNERS={'TASK_QUERY':'任务选择：选择目标','CHANNEL_PROBE':'通道探测：按提示选择，不下发任务','CONFIRM':'任务确认：核对后接受或拒绝'}


def presentation(request,context,*,now):
    event=Event.parse(request.to_dict() if isinstance(request,Event) else request)
    q=event.to_dict()
    require(q['kind']=='REQUEST' and q['mode'] in BANNERS,'presentation_requires_interaction_request')
    require(all(q[k]==context[k] for k in CONTEXT),'stale_presentation_context')
    from datetime import datetime
    t=datetime.fromisoformat(now.replace('Z','+00:00'))
    require(datetime.fromisoformat(q['sent_at'].replace('Z','+00:00'))<=t<datetime.fromisoformat(q['expires_at'].replace('Z','+00:00')),'expired_or_future_presentation')
    return {'scope':'software_fixture_only','mode':q['mode'],'banner':BANNERS[q['mode']],
        'request_id':q['event_id'],'context':{k:q[k] for k in CONTEXT},'options':q['payload']['options'],
        'probe_prompt':q['payload']['probe_prompt'],'contract':q['payload']['contract'],
        'execution_authority':False,'stimulus_presented':False}


def luminance(frame,*,frequency_hz,refresh_hz):
    require(type(frame) is int and frame>=0,'invalid_frame_index')
    require(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (frequency_hz,refresh_hz)),'invalid_frequency')
    require(0<frequency_hz<refresh_hz/2,'frequency_outside_simulated_nyquist')
    phase=round((frequency_hz*frame/refresh_hz)%1,12)
    return round(round(127.5*(1+math.sin(2*math.pi*phase)),9))


def frame_timing(timestamps,*,refresh_hz):
    require(isinstance(refresh_hz,(int,float)) and math.isfinite(refresh_hz) and refresh_hz>0,'invalid_refresh')
    require(len(timestamps)>=2 and all(isinstance(t,(int,float)) and math.isfinite(t) for t in timestamps),'invalid_frame_times')
    gaps=[b-a for a,b in zip(timestamps,timestamps[1:])]
    require(all(g>0 for g in gaps),'nonmonotonic_frame_times')
    period=1/refresh_hz
    return {'scope':'supplied_timestamp_software_check_not_display_measurement','intervals':len(gaps),
        'long_intervals':sum(g>1.5*period for g in gaps),'max_jitter_s':max(abs(g-period) for g in gaps),
        'actual_monitor_verified':False}
