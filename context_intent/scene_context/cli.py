"""Command line tools. Imports torch only for explicit neural operations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .schema import assert_disjoint_scenes, load_cases, validate_observation

VARIANTS = ("full", "no_relations", "no_bci", "no_history", "mlp")


def _predictor(args):
    if args.method == "rule":
        from .baselines import RulePredictor
        return RulePredictor()
    if args.method == "neural":
        from .model import NeuralPredictor
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for neural inference")
        return NeuralPredictor(args.checkpoint)
    from .llm import LLMPredictor
    if not args.cache_only and not args.allow_cloud:
        raise ValueError("Gemini needs --allow-cloud or --cache-only")
    if args.allow_cloud and not args.cache_only and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment; never pass keys in arguments")
    return LLMPredictor(model=args.model, allow_cloud=args.allow_cloud,
        cache_dir=args.cache_dir, cache_only=args.cache_only, retain_raw=False)


def run_suite(args):
    from .baselines import RulePredictor
    from .evaluation import evaluate, paired_scene_comparison
    from .model import NeuralPredictor
    from .training import train_model
    paths={name:Path(args.data_dir)/(name+".jsonl") for name in ("train","val","test")}
    cases={name:load_cases(path) for name,path in paths.items()}
    assert_disjoint_scenes(*cases.values())
    synthetic=any(str(c.get("provenance","")).startswith("synthetic") for split in cases.values() for c in split)
    if synthetic and not args.synthetic:
        raise ValueError("Synthetic suite must explicitly use --synthetic; its results are not paper evidence")
    variants=args.variants.split(",")
    if set(variants)-set(VARIANTS):
        raise ValueError("unsupported ablation variant")
    seeds=[int(x) for x in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate training seeds")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    reports={"rule":evaluate(RulePredictor(),cases["test"],out/"rule",args.bootstrap,args.seed)}
    manifest={"scope":"synthetic_smoke_only" if synthetic else "held_out_offline_context_intent_only",
        "warning":"No robot task success, fatigue, safety or clinical benefit can be inferred from this offline suite.",
        "evaluation_case_ids":[x["case_id"] for x in cases["test"]],"variants":variants,"seeds":seeds,"runs":[],"comparisons":{}}
    for seed in seeds:
        for variant in variants:
            name=f"{variant}_seed{seed}"
            print(f"Training {name}",flush=True)
            trained=train_model(paths["train"],paths["val"],out/name,variant=variant,seed=seed,
                epochs=args.epochs,batch_size=args.batch_size,num_threads=args.threads,synthetic=synthetic,
                target_precision=args.target_precision)
            report=evaluate(NeuralPredictor(trained["checkpoint_path"]),cases["test"],out/name/"evaluation",args.bootstrap,args.seed)
            reports[name]=report
            manifest["runs"].append({"name":name,"parameters":trained["parameter_count"],"checkpoint":trained["checkpoint_path"],
                "metrics":report["metrics"],"training_seconds":trained["training"]["seconds"]})
            print(json.dumps({"run":name,"metrics":report["metrics"]},ensure_ascii=False),flush=True)
    if args.include_gemini:
        if len(cases["test"])>args.max_cloud_cases and not args.cache_only:
            raise ValueError("Test split exceeds --max-cloud-cases; explicitly raise budget or freeze a smaller independent test split for ALL methods")
        args.method="gemini"
        predictor=_predictor(args)
        reports["gemini"]=evaluate(predictor,cases["test"],out/"gemini",args.bootstrap,args.seed)
    for seed in seeds:
        full=f"full_seed{seed}"
        if full in reports:
            for other in ["rule", "gemini", *[f"{v}_seed{seed}" for v in variants if v!="full"]]:
                if other in reports:
                    manifest["comparisons"][f"{full}_minus_{other}"]=paired_scene_comparison(reports[other],reports[full],bootstrap_samples=args.bootstrap,seed=args.seed)
    manifest["rule_metrics"]=reports["rule"]["metrics"]
    if "gemini" in reports:
        manifest["gemini_metrics"]=reports["gemini"]["metrics"]
    (out/"suite.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"suite":str((out/"suite.json").resolve()),"scope":manifest["scope"],"run_count":len(manifest["runs"])}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    generate=sub.add_parser("generate-smoke",help="generate labeled synthetic software fixtures, never research evidence")
    generate.add_argument("--out",required=True)
    generate.add_argument("--scenes",type=int,default=60)
    generate.add_argument("--seed",type=int,default=7)
    validate=sub.add_parser("validate")
    validate.add_argument("--data",nargs="+",required=True)
    train=sub.add_parser("train")
    train.add_argument("--train",required=True)
    train.add_argument("--val",required=True)
    train.add_argument("--out",required=True)
    train.add_argument("--variant",choices=VARIANTS,default="full")
    train.add_argument("--seed",type=int,default=0)
    train.add_argument("--epochs",type=int,default=20)
    train.add_argument("--batch-size",type=int,default=16)
    train.add_argument("--threads",type=int,default=1)
    train.add_argument("--target-precision",type=float,default=.9)
    train.add_argument("--synthetic",action="store_true")
    for name in ("evaluate","predict"):
        item=sub.add_parser(name)
        item.add_argument("--method",choices=("rule","neural","gemini"),required=True)
        item.add_argument("--checkpoint")
        item.add_argument("--model",default="gemini-2.5-flash")
        item.add_argument("--allow-cloud",action="store_true")
        item.add_argument("--cache-only",action="store_true")
        item.add_argument("--cache-dir")
        if name=="evaluate":
            item.add_argument("--data",required=True)
            item.add_argument("--out",required=True)
            item.add_argument("--bootstrap",type=int,default=1000)
            item.add_argument("--seed",type=int,default=0)
            item.add_argument("--max-cloud-cases",type=int,default=200)
        else:
            item.add_argument("--input",required=True,help="observation JSON, not a labeled case")
    suite=sub.add_parser("suite")
    suite.add_argument("--data-dir",required=True)
    suite.add_argument("--out",required=True)
    suite.add_argument("--variants",default=",".join(VARIANTS))
    suite.add_argument("--seeds",default="0,1,2")
    suite.add_argument("--seed",type=int,default=0,help="evaluation bootstrap seed")
    suite.add_argument("--epochs",type=int,default=20)
    suite.add_argument("--batch-size",type=int,default=16)
    suite.add_argument("--threads",type=int,default=1)
    suite.add_argument("--target-precision",type=float,default=.9)
    suite.add_argument("--bootstrap",type=int,default=1000)
    suite.add_argument("--synthetic",action="store_true")
    suite.add_argument("--include-gemini",action="store_true")
    suite.add_argument("--model",default="gemini-2.5-flash")
    suite.add_argument("--allow-cloud",action="store_true")
    suite.add_argument("--cache-only",action="store_true")
    suite.add_argument("--cache-dir")
    suite.add_argument("--max-cloud-cases",type=int,default=200)
    args=parser.parse_args(argv)
    try:
        if args.command=="generate-smoke":
            from .synthetic import make_synthetic_splits
            result=make_synthetic_splits(args.out,args.scenes,args.seed)
        elif args.command=="validate":
            splits=[load_cases(path) for path in args.data]
            assert_disjoint_scenes(*splits)
            result={"valid":True,"counts":[len(x) for x in splits],"scene_counts":[len({c['scene_id'] for c in x}) for x in splits]}
        elif args.command=="train":
            from .training import train_model
            result=train_model(args.train,args.val,args.out,variant=args.variant,seed=args.seed,epochs=args.epochs,
                batch_size=args.batch_size,num_threads=args.threads,target_precision=args.target_precision,synthetic=args.synthetic)
        elif args.command=="suite":
            result=run_suite(args)
        elif args.command=="predict":
            obs=validate_observation(json.loads(Path(args.input).read_text(encoding="utf-8-sig")))
            result=_predictor(args).predict(obs)
        else:
            from .evaluation import evaluate
            cases=load_cases(args.data)
            if args.method=="gemini" and args.allow_cloud and not args.cache_only and len(cases)>args.max_cloud_cases:
                raise ValueError("Test split exceeds --max-cloud-cases")
            report=evaluate(_predictor(args),cases,args.out,args.bootstrap,args.seed)
            result={"metrics":report["metrics"],"output_dir":str(Path(args.out).resolve())}
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,FileNotFoundError,ImportError) as exc:
        parser.error(str(exc))


if __name__=="__main__":
    main()
