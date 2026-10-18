"""Promote the actual failed binding diagnosis to the handoff, without new runs."""
import importlib.util, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'execution/v4'


def write(path, text):
    path.write_text(text,encoding='utf-8-sig',newline='\r\n')


def main():
    audit=json.loads((OUT/'08_training_audit.json').read_text(encoding='utf-8-sig'))
    rows=[r for r in audit['development_challenges']['runs'] if r['method']=='Full']
    assert len(rows)==3
    diagnosis={'scope':'Same allowed validation scenes; three seeds do not increase independent sample size',
        'restricted_training_complete':audit['restricted_development_training_complete'],
        'full_step08_complete':False,'on_prefix_semantic_gate_passed':False,
        'runs':[{'seed':r['seed'],'pointer':r['auxiliary']['pointer'],
                 'paired_evidence':{k:r['pairs']['paired_evidence'][k] for k in ('both_correct','count')},
                 'joint_alias':{k:r['pairs']['joint_alias'][k] for k in ('both_correct','count')}} for r in rows],
        'observed_failure':'Object pointer succeeds on the restricted original validation points, but generated task aliases do not reliably follow evidence changes or alias rebinding.',
        'not_proven':'The experiment localizes a language-binding failure; it does not establish that training data alone or a particular architecture is its sole cause.',
        'next_work':['Balance train object choices and alias permutations within scenes using training groups only',
                     'Add same-prompt different-choice paired training and explicit alias-span supervision',
                     'Recheck evidence changes and joint alias binding before confidence calibration or action dispatch'],
        'no_new_training_or_test_access':True}
    write(OUT/'08_binding_diagnosis.json',json.dumps(diagnosis,ensure_ascii=False,indent=2)+'\n')
    audit['binding_diagnosis']=diagnosis
    write(OUT/'08_training_audit.json',json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('finalize08',OUT/'finalize_step08.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    write(ROOT/'docs/v4/08_training.txt',module.document(audit))
    note='关键语义缺口：Full三个种子的证据切换与别名重绑定均0/8（双侧同时正确）；原声明对象指针均16/16。下一步先修内部token→目标语言绑定，不能进入动作闭环或宣称意图推理已通过。\n'
    for path in (ROOT/'README.txt',ROOT/'execution/handoff.txt',OUT/'handoff.txt',OUT/'handoff_step08.txt'):
        text=path.read_text(encoding='utf-8-sig')
        if note.strip() not in text:
            write(path,note+'\n'+text)
    progress_path=ROOT/'execution/v4_plan_progress.json'
    progress=json.loads(progress_path.read_text(encoding='utf-8-sig'))
    progress['step08_dependency']='Fix observed token-to-language object binding first; remaining hidden-intent/placement supervision and external validation'
    step=next(s for s in progress['steps'] if s['step_id']=='V4-08')
    step['functional_binding_gate']='FAILED_DEVELOPMENT_PAIR_DIAGNOSTIC'
    step['binding_diagnosis']='execution/v4/08_binding_diagnosis.json'
    write(progress_path,json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    worklog=ROOT/'execution/worklog.txt'
    text=worklog.read_text(encoding='utf-8-sig')
    if note.strip() not in text:
        write(worklog,text.rstrip()+'\n\n'+note)
    print(json.dumps({'training_complete':True,'binding_gate':'FAILED','full_step08_complete':False}))


if __name__=='__main__':
    main()
