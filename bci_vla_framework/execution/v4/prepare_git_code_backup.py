"""Select a reviewable source branch without datasets, weights or credentials."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'bci_vla_framework/execution/v4'
BRANCH='codex/step11-upstream-subtask-20261018'

def git_paths(*args):
    return [s for s in subprocess.check_output(['git',*args,'-z'],cwd=ROOT).decode('utf-8').split('\0') if s]

def selected(name):
    p=Path(name);s=p.as_posix()
    if '/' not in s:return p.name.startswith('BCI_VLA') and p.suffix=='.txt'
    prefix='bci_vla_framework/'
    if not s.startswith(prefix):return False
    rel=s[len(prefix):]
    if rel.startswith(('bci_vla/','tests/')):return p.suffix=='.py'
    if rel.startswith(('configs/','docs/v4/','examples/v4/')):return p.suffix in {'.py','.json','.txt'}
    if rel.startswith('third_party/openpi_with_subtask/'):
        return p.suffix in {'.py','.toml','.lock','.md','.txt','.yml','.yaml','.sh','.in','.typed'} or p.name in {'LICENSE','.gitmodules','.python-version','.gitignore','.dockerignore'}
    if rel.startswith('execution/v4/09_binding_data_v2/validation/'):
        return p.name in {'public_input.jsonl','supervision.jsonl'}
    if rel.startswith('execution/v4/'):
        return len(Path(rel).parts)==3 and (p.suffix in {'.py','.json','.txt','.log'} or p.name=='11_code_bundle1.zip')
    if rel.startswith('execution/'):
        return p.suffix in {'.py','.json','.txt'} and not rel.startswith('execution/v4_plan_review_')
    return rel in {'figures/pi05_interaction_transformer_en_v1.png','figures/pi05_interaction_transformer_en_v1_prompt.txt'}

def main():
    untracked=git_paths('ls-files','--others','--exclude-standard')
    modified=git_paths('diff','--name-only')
    assert not git_paths('diff','--cached','--name-only'),'Existing staged files require separate review'
    names=sorted(set(modified+[n for n in untracked if selected(n)]))
    patterns={'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
              'github_token':rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
              'openai_token':rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}\b',
              'aws_access_key':rb'\bAKIA[0-9A-Z]{16}\b'}
    findings=[];files=[]
    for name in names:
        path=ROOT/name;data=path.read_bytes()
        if path.suffix not in {'.png','.zip'}:
            for kind,pattern in patterns.items():
                if re.search(pattern,data):findings.append({'path':name,'kind':kind})
        files.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    assert not findings,json.dumps(findings)
    report={'branch':BRANCH,'files':files,'total_bytes':sum(f['bytes'] for f in files),
            'excluded_untracked_count':len(set(untracked)-set(names)),
            'scope':'V3/V4 source/config/plans/tests/summary reports; two explicit development validation fixtures; pinned upstream source',
            'excluded':['model weights','EEG raw streams','full datasets','large result archives','simulation videos','credentials','runtime caches'],
            'secret_pattern_scan_findings':findings,'step11_real_weight_gate':'NOT_RUN',
            'backup_is_not_full_external_environment_or_data_restore':True}
    target=OUT/'11_git_backup_manifest.json'
    # The manifest deliberately has no self-hash.
    names=[n for n in names if n!=target.relative_to(ROOT).as_posix()]
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    names.append(target.relative_to(ROOT).as_posix())
    (ROOT/'.runtime/git_step11_pathspec.txt').write_bytes(b''.join(n.encode()+b'\0' for n in names))
    print(json.dumps({'branch':BRANCH,'files_to_stage':len(names),'total_bytes':report['total_bytes'],
                     'excluded':report['excluded_untracked_count'],'extensions':dict(Counter(Path(n).suffix for n in names)),
                     'secret_pattern_scan_findings':findings},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
