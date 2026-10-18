"""Restricted declared-protocol high-level loop. Virtual semantic dispatch ONLY.

No actuator API is accepted. Confirmation is a separate trusted UI boundary;
model text can never confirm a task, create a receipt, or grant joint authority.
"""
from copy import deepcopy
import math
import re
from threading import RLock
from uuid import uuid4

from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.data_v4.contracts import validate_public

_MODEL_OWNER = RLock()


class AuthorizationError(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise AuthorizationError(reason)


def parse_task(text, public):
    """Exact bounded grammar: no fuzzy alias correction, JSON or code execution."""
    require(isinstance(text, str) and len(text) <= 256, 'invalid_text')
    objects = [o for o in public['scene']['objects'] if o['object_valid']]
    aliases = {o['alias']: o['instance_id'] for o in objects}
    require(len(aliases) == len(objects), 'duplicate_alias')
    # Upstream tokenizer lowercases and replaces underscores. Both spellings
    # are explicit public aliases; never substitute a different candidate ID.
    normalized = re.sub(r'\bobj (\d+)\b', lambda m: 'OBJ_'+m[1], text)
    normalized = re.sub(r'^pick up ', 'Pick up ', normalized)
    if not normalized.endswith('.'):
        normalized += '.'
    match = re.fullmatch(r'Pick up (OBJ_\d+)\.', normalized)
    if match:
        source = match[1]; destination = None; action = 'pick'
    else:
        match = re.fullmatch(r'Put (OBJ_\d+) in (OBJ_\d+)\.', normalized)
        require(match is not None, 'unparseable_or_missing_slot')
        source, destination, action = match[1], match[2], 'place'
    require(source in aliases and (destination is None or destination in aliases), 'hallucinated_object')
    require(source != destination, 'same_source_destination')
    return {'action': action, 'source_id': aliases[source],
            'destination_id': None if destination is None else aliases[destination]}


def declared_scope(public):
    """Only explicit consistent public hard declarations, never private labels.

    This is NOT the noisy Step09 policy and must not enable its VLA flag.
    """
    tasks = {t['task_id']: t for t in public['task_options']}
    latest = public['history'][-1] if public['history'] else None
    require(latest is not None and latest['role'] != 'NO_RESPONSE', 'latest_no_response')
    declarations = {}
    for event in public['history']:
        role = event['role']; q = event['question']; ev = event['evidence']
        if role not in {'TASK', 'INTENT_QUERY'} or not event['valid']:
            continue
        if q['context'] != public['context'] or not q['presentation_ack']:
            continue
        if not q['issued_at_s'] <= event['started_at_s'] <= event['ended_at_s'] < q['expires_at_s']:
            continue
        if ev['score_kind'] != 'HARD_LABEL':
            continue
        if public['as_of_s'] - event['ended_at_s'] > 10:
            continue
        chosen = next((o for o in q['options'] if o['option_id'] == ev['selected_option_id']), None)
        if chosen is None:
            continue
        if chosen['kind'] == 'OBJECT':
            declarations[role] = {'source_id': chosen['reference']}
        elif chosen['kind'] == 'TASK' and chosen['reference'] in tasks:
            declarations[role] = {k: tasks[chosen['reference']][k] for k in ('action', 'source_id', 'destination_id')}
    require(set(declarations) == {'TASK', 'INTENT_QUERY'}, 'explicit_declarations_required')
    a, b = declarations['TASK'], declarations['INTENT_QUERY']
    require(a['source_id'] == b['source_id'], 'conflicting_declarations')
    require('action' in b, 'explicit_action_required')
    return b


class HighLevelSession:
    """One session owns one serialized backend; outputs never leave virtual scope.

    backend.generate(public, mode, gripper_state) -> raw_text, stopped_at_eos,
    candidate_probabilities {instance_id: p}, outside_probability, model_binding.
    Backend must expose model_binding and reset_generation(). It is shared for
    both modes. Backend implementations must serialize shared-model ownership.
    """
    def __init__(self, backend, *, ttl_s=10.):
        require(math.isfinite(ttl_s) and 0 < ttl_s <= 60, 'invalid_ttl')
        self.backend = backend; self.ttl_s = ttl_s; self.lock = RLock()
        self.binding = deepcopy(backend.model_binding)
        self.journal = []; self.used_receipts = set(); self.queue = []
        self.context = None; self.public_digest = None; self.main = None
        self.pending = None; self.confirmed = None; self.halted = False
        self.last_time = -math.inf; self.phase = 'IDLE'

    def _revoke(self, reason):
        self.queue.clear(); self.pending = None; self.main = None; self.confirmed = None
        self.backend.reset_generation(); self.phase = 'PAUSED'
        self.journal.append({'event': 'REVOKED', 'reason': reason})

    def control(self, kind):
        with self.lock:
            require(kind in {'STOP', 'CANCEL'}, 'unknown_control')
            self._revoke(kind); self.halted = True

    def _current(self, public, now):
        require(isinstance(now, (int, float)) and not isinstance(now, bool) and math.isfinite(now), 'invalid_clock')
        if now < self.last_time:
            self._revoke('clock_reversed'); raise AuthorizationError('clock_reversed')
        self.last_time = now
        require(not self.halted, 'stop_or_cancel_latched')
        try:
            validate_public(public)
            require(not public['public_state']['stop_latched'], 'public_stop')
            require(self.backend.model_binding == self.binding, 'model_changed')
            require(now >= public['as_of_s'] and now-public['as_of_s'] <= self.ttl_s, 'stale_observation')
            digest = canonical_digest(public)
            if self.public_digest is not None:
                require(self.public_digest == digest, 'public_evidence_or_scope_changed')
            if self.pending:
                require(now < self.pending['expires_at_s'], 'expired')
            if self.main:
                require(now < self.main['expires_at_s'], 'parent_expired')
        except Exception:
            self._revoke('context_or_lease_invalid'); raise
        self.context = deepcopy(public['context']); self.public_digest = digest

    def _generation(self, public, mode, gripper):
        self.backend.reset_generation()
        with _MODEL_OWNER:
            result = self.backend.generate(deepcopy(public), mode, gripper,
                                           confirmed_task=deepcopy(self.main) if mode == 'next_subtask' else None)
        self.journal.append({'event': 'VLM_RAW', 'mode': mode, 'result': deepcopy(result)})
        require(result.get('model_binding') == self.binding, 'generation_model_mismatch')
        require(result.get('stopped_at_eos') is True, 'truncated_generation')
        task = parse_task(result.get('raw_text'), public)
        probs = result['candidate_probabilities']
        require(probs and all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values()), 'invalid_candidate_distribution')
        outside = result['outside_probability']
        require(math.isfinite(outside) and 0 <= outside <= .2, 'outside_or_invalid')
        require(abs(sum(probs.values())+outside-1) < 1e-4, 'unnormalized_distribution')
        require(task['source_id'] == max(probs, key=probs.get) and probs[task['source_id']] >= .6,
                'pointer_text_disagreement_or_uncertain')
        return task

    def _proposal(self, task, kind, now, parent_id=None):
        value = dict(task, kind=kind, suggestion_id=uuid4().hex,
                     parent_id=parent_id, context=deepcopy(self.context),
                     public_sha256=self.public_digest, model_binding=deepcopy(self.binding),
                     expires_at_s=min(now+self.ttl_s, self.main['expires_at_s']) if self.main else now+self.ttl_s,
                     execution_authorized=False, physical_actions=False,
                     authorization_scope='one_virtual_semantic_subtask' if kind == 'SUBTASK' else 'planning_only')
        value['sha256'] = canonical_digest(value)
        self.pending = value; self.phase = 'AWAITING_'+kind+'_CONFIRMATION'
        return deepcopy(value)

    def propose_main(self, public, *, now):
        with self.lock:
            self._revoke('new_main_proposal'); self.public_digest = None
            try:
                self._current(public, now)
                scope = declared_scope(public)
                task = self._generation(public, 'main_task', None)
                require(task == scope, 'task_outside_public_declaration')
                return self._proposal(task, 'MAIN', now)
            except Exception:
                self._revoke('main_rejected'); raise

    def confirm(self, public, *, proposal_sha256, receipt_id, accepted, now):
        """Trusted SIMULATED UI callback. Never pass language-model output here."""
        with self.lock:
            try:
                self._current(public, now)
                require(isinstance(receipt_id, str) and 0 < len(receipt_id) <= 128, 'invalid_receipt')
                require(receipt_id not in self.used_receipts, 'replayed_receipt')
                require(self.pending is not None and self.pending['sha256'] == proposal_sha256, 'wrong_or_stale_confirmation')
                require(type(accepted) is bool and accepted, 'rejected_by_user')
                self.used_receipts.add(receipt_id)
                if self.pending['kind'] == 'MAIN':
                    self.main = deepcopy(self.pending)
                    self.main['confirmation_receipt_id'] = receipt_id
                    self.main['confirmed_at_s'] = now
                    self.phase = 'MAIN_CONFIRMED'
                else:
                    require(self.main and self.pending['parent_id'] == self.main['suggestion_id'], 'wrong_parent')
                    self.queue.append(deepcopy(self.pending)); self.phase = 'VIRTUAL_SUBTASK_AUTHORIZED'
                self.journal.append({'event': 'SIMULATED_UI_CONFIRMATION', 'receipt_id': receipt_id,
                                     'proposal_sha256': proposal_sha256, 'kind': self.pending['kind']})
                self.pending = None
            except Exception:
                self._revoke('confirmation_rejected'); raise

    def propose_subtask(self, public, *, now, gripper_state):
        with self.lock:
            try:
                self._current(public, now)
                require(self.main is not None and self.phase == 'MAIN_CONFIRMED', 'main_confirmation_required')
                require(gripper_state == 'empty' and self.main['action'] == 'pick', 'unverified_subtask_phase')
                task = self._generation(public, 'next_subtask', gripper_state)
                require(task == {k: self.main[k] for k in ('action', 'source_id', 'destination_id')}, 'subtask_exceeds_parent')
                return self._proposal(task, 'SUBTASK', now, self.main['suggestion_id'])
            except Exception:
                self._revoke('subtask_rejected'); raise

    def consume_virtual(self, public, *, now):
        with self.lock:
            try:
                self._current(public, now)
                require(len(self.queue) == 1, 'per_subtask_confirmation_required')
                task = self.queue.pop()
                require(now < task['expires_at_s'], 'subtask_expired')
                self.phase = 'AWAITING_OBSERVED_ENDPOINT'
                result = {'semantic_request': task, 'simulation_authorized': True,
                          'execution_authorized': False, 'VLA_called': False,
                          'joint_action': None, 'physical_actions': False}
                self.journal.append({'event': 'VIRTUAL_DISPATCH', 'result': deepcopy(result)})
                return result
            except Exception:
                self._revoke('dispatch_rejected'); raise
