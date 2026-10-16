"""Bounded known-target replay mapping repair; no hidden goal enters inference."""
from ..protocols import thaw,require
from ..benchmark.offline import stamp
from .factory import GOALS,make_runtime
from .backend import SimulationVLAAdapter
from .interaction import interact,resolve

def joint_vector(session,*,swap_modes=False):
    cells={(r['goal_id'],r['mode_id']):r['probability'] for r in session.filter.snapshot()['belief']['mass']}
    modes=['flip','normal'] if swap_modes else ['normal','flip']
    return tuple(cells.get((g,z),0) for g in GOALS for z in modes)

def repair_replay_mapping(old,controller,engine,spec,outcomes,identifier,*,emit_goal,emit_channel):
    # emit_* are used only by the external scripted evidence generator below.
    # Decisions and new priors use public evidence, never those hidden labels.
    now=controller.now+.1
    if not old.stop_latched:old.request_stop(now=stamp(now))
    revision=old.context['scene_revision']+1
    backend=SimulationVLAAdapter(spec,engine)
    diagnostic,c=make_runtime(identifier+'-mapping-check',engine.scene(identifier+'-mapping-check',revision),backend,outcomes,
        prior=joint_vector(old),now=now+.1,task_revision=old.context['task_revision']+1)
    observations=[]
    for prompt in ['0','1','0','1']:
        item=interact(c,dict(kind='CHANNEL_PROBE',prompt=prompt,setup_s=1.,duration_s=1.,elapsed_s=2.),
            private_goal_index=emit_goal,private_channel_mode=emit_channel)
        observations.append(dict(prompt=prompt,raw_symbol=item['observed_symbol'],request_id=item['request_id']))
    flips=[o['prompt']!=o['raw_symbol'] for o in observations]
    require(len(set(flips))==1,'inconsistent_probe_mapping_pause')
    inversion=flips[0];new_revision=revision+1
    backend=SimulationVLAAdapter(spec,engine)
    session,new=make_runtime(identifier+'-repaired',engine.scene(identifier+'-repaired',new_revision),backend,outcomes,
        prior=joint_vector(diagnostic,swap_modes=inversion),now=c.now+.1,task_revision=old.context['task_revision']+2,
        input_inversion=inversion,profile_revision=new_revision)
    initial_goals=session.filter.snapshot()['goal_marginals']
    modes=resolve(new,emit_goal,emit_channel)
    evidence=dict(scope='known_target_synthetic_replay_mapping_repair_not_real_EEG_calibration',observations=observations,
        inversion=inversion,new_context=session.context,prior_from_joint_posterior_not_private_goal=True,
        goal_marginals_before=diagnostic.filter.snapshot()['goal_marginals'],initial_repaired_goal_marginals=initial_goals,
        goal_marginals_after_queries=session.filter.snapshot()['goal_marginals'])
    return session,new,backend,[diagnostic,session],modes,evidence
