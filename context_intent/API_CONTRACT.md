Internal implementation contract

Python package: `scene_context`; no robot SDK imports or execution.

Observation (plain JSON dictionaries):
```
{"objects":[{"id":"g1","label":"grapes","position":[0.2,0.5],
"attributes":[],"affordances":["pickable"],"detection_confidence":1.0}],
"relations":[{"subject":"g1","predicate":"left_of","object":"b1","confidence":1.0}],
"selection":{"object_id":"g1","probabilities":{}},
"context":{"goal":"put_away","confirmed_action":null,
"confirmed_destination_id":null,"destination_hint":"nearest_source","history":[]}}
```
Position is normalized image xy [0,1]. Supported actions: pick, place, fold. Goals: unspecified, pick, put_away, fold. Destination hints: unspecified, leftmost, rightmost, nearest_source. Affordances: pickable, container, foldable. Objects have instance IDs, not only class names. Annotation must never enter predictor input.

Case format: `{"case_id":str,"scene_id":str,"observation":Observation,"annotation":{"status":"ready|clarify|reject","intent":{"action":...,"source_id":...,"destination_id":str|null}|null,"clarification_slot":"source|action|destination"|null}}`. More metadata allowed outside observation. A physical scene and all its variants must stay in one split.

Root-owned `schema.py`: ACTIONS, GOALS, RELATIONS; validate_observation(dict)->dict; validate_case(dict)->dict; public_observation(case)->dict; load_cases(path)->list; assert_disjoint_scenes(train,val,test).
Root-owned `planning.py`: generate_candidates(obs)->list of intent dicts, candidate_key(intent)->str; structural_clarification(obs)->dict|None; finalize_prediction(obs,proposal,confidence=None,threshold=0.0,source='')->dict. Proposal has status, intent, clarification_slot. Final result has status, intent, instruction, confidence, clarification, reason, requires_confirmation and raw_proposal. Ready is a proposed instruction requiring explicit user confirmation; nothing executes. All baselines must share this finalizer. Raw and guarded results must both be evaluated.

Root-owned `rendering.py`: render_instruction(obs,intent)->str, raises ValueError when visually ungroundable. Supported finite task grammar only, no opaque IDs in natural-language output.

Agent model owns `model.py`, `training.py`. Expose `ModelConfig` dataclass (variant full/no_relations/no_bci/no_history/mlp), `IntentModel(config)`, `NeuralPredictor(checkpoint_path,threshold=None).predict(observation)->dict`, `train_model(train_path,val_path,output_dir,variant='full',seed=0,epochs=20,**kwargs)->dict`. Pure torch + standard library. CPU compatible. Candidate scorer + status head; relation-aware graph attention; BCI-conditioned query. Loss uses annotation only during training. Save checkpoint + manifest + validation threshold settings. Never train on test. Synthetic scores must be labeled smoke tests.

Agent evaluation owns `baselines.py`, `llm.py`, `evaluation.py`. Expose `RulePredictor().predict(obs)`, `LLMPredictor(model='gemini-2.5-flash',api_key=None,**kwargs).predict(obs)` (no API calls unless explicitly invoked; env key only, never print/save key), `build_prompt(obs)->str`; `evaluate(predictor,cases,output_dir=None,bootstrap_samples=1000,seed=0)->dict`. LLM receives exactly public observation and allowed action schema, not annotations. Explicit CLI authorization required for cloud calls. Record raw + guarded metrics, latency and errors; scene-level bootstrap, not trial pseudoreplication. Parsing/API failure must reject, no guessed pickup fallback. Preserve raw outputs locally only with configurable retention and never log auth.

Agent tests/docs owns tests (unittest, no network/robot) and docs/EXPERIMENTS_ZH.md, docs/ANNOTATION_ZH.md, configs/experiment.json. Main code API is above; coordinate early if changes needed. Tests should meaningfully exercise ambiguous intent, instance grounding, leakage, malformed input, invalid model outputs, relation/model permutation behavior. They can inspect pending files after root adds them.

Root owns synthetic.py, cli.py, integration.py, README.md, pyproject.toml, examples and scripts. All implementation must be local under this new directory. No modification of existing server files, no robot activation, no paid API calls by any agent.
