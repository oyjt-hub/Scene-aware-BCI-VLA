"""Public-evidence contract before a noisy confirmation can be proposed.

This is an explicit rule gate, not a learned policy or an execution authority.
It never reads a latent goal, injected channel rate, or a future answer.
"""
import math


def confirmation_evidence_gate(public, source_id, config):
    gate = config.get('confirmation_gate', {'mode': 'none'})
    if gate['mode'] == 'none':
        return None
    if gate['mode'] != 'consistent_recheck':
        raise ValueError('unknown confirmation evidence gate')
    required = gate['consistent_selections']
    checks = gate['correct_probes']
    age = gate['selection_max_age_s']
    if (type(required) is not int or required < 2 or type(checks) is not int or checks < 2
            or type(age) not in (int, float) or not math.isfinite(age) or age <= 0):
        raise ValueError('at least two selections and two diagnostic checks required')
    def ask(action, reason):
        return {'action': action, 'reason': reason, 'execution_authorized': False,
                'requires_user_confirmation': True, 'evidence_gate': 'consistent-recheck/0.1'}
    ctx = public['context']; now = public['as_of_s']
    events = []
    for event in public['history']:
        q = event.get('question')
        if (not event['valid'] or event['role'] not in ('TASK', 'INTENT_QUERY')
                or q is None or q['context'] != ctx or now-event['ended_at_s'] > age):
            continue
        selected = event['evidence']['selected_option_id']
        option = next((o for o in q['options'] if o['option_id'] == selected), None)
        if option is not None and option['kind'] == 'OBJECT':
            events.append((event, option['reference']))
    recent = events[-required:]
    if len(recent) < required or any(oid != source_id for _, oid in recent):
        return ask('INTENT_QUERY', 'FRESH_SELECTION_CONSISTENCY_REQUIRED')
    # Checks must occur AFTER the latest object selection; startup probes do
    # not certify a later proposed task. Every intervening check must succeed.
    after = recent[-1][0]['ended_at_s']
    diagnostics = public['public_state']['diagnostics']
    probes = []
    for event in diagnostics['receipts']:
        q = event.get('question')
        if (event['valid'] and q and q['role'] == 'PROBE' and q['context'] == ctx
                and q['layout_id'] == diagnostics['layout_id'] and event['ended_at_s'] > after):
            probes.append(event)
    probes.sort(key=lambda e: (e['ended_at_s'], e['sequence']))
    if len({e['event_id'] for e in probes}) != len(probes):
        raise ValueError('duplicate diagnostic receipts cannot certify a proposal')
    for event in probes:
        q = event['question']
        if event['role'] == 'NO_RESPONSE' or event['evidence']['selected_option_id'] != q['diagnostic_option_id']:
            return ask('WAIT', 'POST_SELECTION_CHANNEL_CHECK_FAILED')
    if len(probes) < checks:
        return ask('PROBE', 'POST_SELECTION_CHANNEL_CERTIFICATE_REQUIRED')
    symbols = {next(o['symbol'] for o in e['question']['options']
                    if o['option_id'] == e['question']['diagnostic_option_id']) for e in probes}
    if len(symbols) < 2:
        return ask('PROBE', 'POST_SELECTION_SYMBOL_COVERAGE_REQUIRED')
    return None
