"""Public-only validation challenges for the restricted declaration pilot.

These are development diagnostics, not additional training data, sealed tests,
real BCI observations or action rollouts. No filesystem I/O occurs here.
"""
from copy import deepcopy

import torch

from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.neural_interaction_v4.tensorizer import collate_public
from bci_vla.training_v4.data import (
    PROTOCOL, declared_teacher, language_prompt, make_example, validate_example,
)

SCHEMA_VERSION = 'step08-development-challenges/0.1'
FAMILIES = ('declared_pick', 'next_pick_symbolic')


def _public_target(row):
    """Targets are re-derived from explicit public declarations, not hidden truth."""
    teaching = declared_teacher(row['public'])
    if teaching['task'] is None or teaching['task']['action'] != 'pick':
        raise ValueError('challenge requires a resolved public pick declaration')
    return teaching['text'], teaching['task']['source_id']


def _tensors_equal(first, second):
    a, _ = collate_public([first['public']])
    b, _ = collate_public([second['public']])
    return all(torch.equal(a[key], b[key]) for key in a)


def _pair(kind, source, first, second, *, alias_mapping=None):
    first_target, first_source = _public_target(first)
    second_target, second_source = _public_target(second)
    same_prompt = first['supervision']['language']['prompt'] == second['supervision']['language']['prompt']
    same_tensors = _tensors_equal(first, second)
    same_source = first_source == second_source
    same_context = first['public']['context'] == second['public']['context']
    same_image = (first['public']['scene']['image_ref'] == second['public']['scene']['image_ref']
                  and first['supervision']['source']['image_sha256'] == second['supervision']['source']['image_sha256'])
    if first_target == second_target or not same_context or not same_image:
        raise ValueError('challenge must change answer while preserving scene/image')
    if kind == 'paired_evidence' and (not same_prompt or same_tensors or same_source):
        raise ValueError('paired evidence must change public choice, not native prompt')
    if kind == 'joint_alias' and (same_prompt or not same_tensors or not same_source):
        raise ValueError('joint alias must preserve physical choice and connector tensors')
    pair = {
        'challenge_id': canonical_digest([SCHEMA_VERSION, kind, source['public']['sample_id'],
                                          first['public']['sample_id'], second['public']['sample_id']])[:32],
        'kind': kind, 'family': source['supervision']['family'],
        'source_sample_id': source['public']['sample_id'],
        'first': first, 'second': second,
        'expected': {'first_target': first_target, 'second_target': second_target,
                     'first_source_id': first_source, 'second_source_id': second_source,
                     'answer_must_change': True},
        'invariants': {'native_prompt_equal': same_prompt, 'connector_tensors_equal': same_tensors,
                       'physical_source_equal': same_source, 'scene_context_equal': same_context,
                       'image_equal': same_image},
    }
    if alias_mapping is not None:
        pair['alias_mapping'] = alias_mapping
    return pair


def _joint_alias(row):
    """Rename aliases jointly in the public dictionary and public-only target.

    Physical instance references, candidate row order, numeric features, event
    evidence, task contracts and pixels remain identical. Aliases never enter
    the connector, so only the VLM dictionary/answer binding should change.
    """
    result = deepcopy(row)
    public = result['public']
    aliases = [obj['alias'] for obj in public['scene']['objects']]
    if len(aliases) < 2:
        raise ValueError('at least two candidate aliases required')
    mapping = dict(zip(aliases, aliases[1:] + aliases[:1]))
    for obj in public['scene']['objects']:
        obj['alias'] = mapping[obj['alias']]
    public['sample_id'] = canonical_digest([SCHEMA_VERSION, row['public']['sample_id'], mapping])[:32]
    digest = canonical_digest(public)
    teaching = declared_teacher(public)
    for private in result['private_variants']:
        private['sample_id'] = public['sample_id']
        private['public_input_sha256'] = digest
        private['labels']['main_task']['text'] = teaching['text']
        private['labels']['interaction']['teacher_input_sha256'] = digest
    result['private_labels'] = result['private_variants'][0]
    supervision = result['supervision']
    supervision['sample_id'] = public['sample_id']
    supervision['public_input_sha256'] = digest
    lang = supervision['language']
    lang['target'] = teaching['text']
    lang['prompt'] = language_prompt(public, supervision['protocol'], lang['kind'], lang['gripper_state'])
    validate_example(result)
    return result, mapping


def build_development_challenges(validation_rows, *, max_source_rows=None):
    """Return paired evidence and joint alias records with complete runnable rows.

    Pair records contain `first` and `second` examples in the usual Step08 row
    interface. The same first row is reused across both challenge types, so a
    runner can cache generation by public sample ID. Each eligible source gives
    three unique examples: original publicly declared source, another publicly
    declared source under the *same* native prompt, and jointly rebound aliases
    for the first example. Model output is never used to choose the challenge.

    Default source families are declared_pick and next_pick_symbolic only.
    Every supplied row must belong to validation: no train/calibration/test
    selection or hidden-goal-conditioned target generation is allowed.
    """
    if max_source_rows is not None and (type(max_source_rows) is not int or max_source_rows < 1):
        raise ValueError('positive source limit required')
    eligible = []
    for row in validation_rows:
        if row['supervision']['split'] != 'validation':
            raise ValueError('development challenges accept validation only; other partitions are sealed')
        validate_example(row)
        if row['supervision']['family'] in FAMILIES:
            if row['supervision']['protocol'] != PROTOCOL:
                raise ValueError('explicit public declaration protocol required')
            eligible.append(row)
    if not eligible:
        raise ValueError('no eligible validation declaration examples')
    # Stable ordering and round-robin families avoid selecting one family first
    # when a runner requests a smaller fixed diagnostic budget.
    by_family = {family: sorted([row for row in eligible if row['supervision']['family'] == family],
                               key=lambda row: row['public']['sample_id']) for family in FAMILIES}
    selected = []
    for index in range(max(map(len, by_family.values()))):
        for family in FAMILIES:
            if index < len(by_family[family]):
                selected.append(by_family[family][index])
    if max_source_rows is not None:
        selected = selected[:max_source_rows]
    paired, aliased = [], []
    for source in selected:
        public = source['public']
        objects = [obj for obj in public['scene']['objects'] if obj['object_valid']]
        _, target_id = _public_target(source)
        position = next(i for i, obj in enumerate(objects) if obj['instance_id'] == target_id)
        if len(objects) < 2:
            raise ValueError('at least two valid candidates required')
        variants = []
        for target_index in (position, (position + 1) % len(objects)):
            variant = make_example(public, source.get('private_variants', [source['private_labels']]),
                                   'validation', source['supervision']['family'], target_index,
                                   image_sha256=source['supervision']['source']['image_sha256'])
            validate_example(variant)
            variants.append(variant)
        first, second = variants
        paired.append(_pair('paired_evidence', source, first, second))
        renamed, mapping = _joint_alias(first)
        aliased.append(_pair('joint_alias', source, first, renamed, alias_mapping=mapping))
    unique_ids = {pair[side]['public']['sample_id'] for pair in paired + aliased for side in ('first', 'second')}
    return {
        'schema_version': SCHEMA_VERSION,
        'scope': 'validation-only public declaration diagnostics; no training, EEG or action evaluation',
        'source_split': 'validation', 'public_only_teacher': True,
        'formal_test_accessed': False, 'calibration_accessed': False,
        'paired_evidence': paired, 'joint_alias': aliased,
        'counts': {'source_rows': len(selected), 'paired_evidence_pairs': len(paired),
                   'joint_alias_pairs': len(aliased), 'unique_public_examples': len(unique_ids)},
    }
