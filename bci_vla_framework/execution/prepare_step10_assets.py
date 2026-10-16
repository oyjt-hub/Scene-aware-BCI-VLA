"""Create development-only baseline registry and tuning/ablation declarations."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def write(p,v):
    path=ROOT/p; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    names={
        'B0':('原LLM固定补选链','external_trace_required','原服务器源码冻结；缺实际API/完整耗时与共同确认包装，不填假结果'),
        'B1':('NOIR-inspired固定选择','offline_adaptation','共享二选一目标而非原对象/技能层级；不冒充完整NOIR'),
        'B1P':('NOIR-inspired预测核验与纠正','offline_adaptation','外部公共目标预测；任务问题核验，冲突后纠正，再共同含噪最终确认'),
        'B2':('固定通道允许目标变化','offline_finite','每次预测把通道投影到normal；世界模型仍允许通道变'),
        'B3':('目标惯性与独立被动通道监测','offline_finite','目标gap不变、g/z独立重组；无主动probe；不是最优漂移监测复现'),
        'B4':('联合滤波无主动已知目标探测','offline_finite','任务澄清允许，probe禁用'),
        'B5':('固定周期probe','offline_finite','按诊断次数计周期，初次为探测相位；间隔可调，探测题用同模型风险选择'),
        'B6':('冲突重复或延长','offline_finite','公共历史目标冲突触发；延长需unconsumed许可，重试需fresh许可'),
        'B7':('冲突或质量阈值probe','offline_finite','质量未知不填成良好；不能读取评价真值'),
        'B8':('通道信息增益再语义澄清','offline_finite','每秒I(Z;Y)而非任务收益；熵阈值待验证集调参'),
        'B9':('标准一步任务VoI','offline_finite','与M相同信息、动作、后果、预算、含噪确认；独立树实现'),
        'B10':('RBC-inspired递归Bayes/停时/语义VoI组合','restricted_binary_adaptation','只有一个已验证二元码本；不能代表完整MMI重编码或窗内动态停时'),
        'M':('主动诊断默认一步','offline_finite','步骤09原实现；若等价B9就不声称求解器创新'),
        'M2':('主动诊断两步候选','offline_finite','同一有限模型，多一个诊断展望；与horizon2的O1对照'),
        'O1':('同信息精确枚举','offline_exact_reference','独立递归，不调用步骤09 plan/_evaluate/_branches；共享校验模型与合法动作'),
        'O2':('全知g/z非现实参考','non_deployable_reference','每个决策揭示状态，但仍保留同样含噪最终确认和成本，不能作在线方法')}
    write('configs/comparisons/method_registry.json',{
        'schema_version':'comparison-registry/1.0','status':'offline_implementation_only','deployment_allowed':False,
        'shared':['calibration models','goal vocabulary','scene and contracts','action registry and legal durations',
                  'wall-time budget','noisy final confirmation','independent STOP','original world loss for scoring'],
        'methods':[{'id':key,'name':v[0],'status':v[1],'adaptation':v[2],
            'online_hidden_labels':False,'confirmation':'same_noisy_model','executor':'same_unverified_simulation_backend'} for key,v in names.items()],
        'real_deployment_verified':False,'strongest_real_baseline':None})
    write('configs/comparisons/parameters.json',{
        'status':'development_only','deployment_allowed':False,
        'period':2,'confidence':.9,'conflict_threshold':.35,'quality_threshold':.5,
        'channel_entropy_stop':.05,'fixed_duration_s':1.0,
        'provenance':'illustrative_defaults_not_tuned'})
    write('configs/comparisons/tuning_and_selection.json',{
        'status':'pending_independent_validation','chosen_baseline':None,
        'grids':{'period':[1,2,3,4],'confidence':[.7,.8,.9,.95],
            'conflict_threshold':[.2,.35,.5,.7],'quality_threshold':[.25,.5,.75],
            'channel_entropy_stop':[0,.05,.2,.5]},
        'splits':['development','calibration','validation','test'],
        'split_unit':'user/session/physical_scene groups; all episode prefixes together',
        'selection_rule':{'status':'awaiting_independent_pilot_limits','source_split':'calibration','source_ref':None,
            'wrong_start_rate_ceiling':None,'all_episode_mean_time_ceiling_s':None,
            'rank':['maximum all-episode faithful completion rate','minimum wrong-start rate',
                    'minimum all-episode mean time','minimum abstention rate','candidate_id']},
        'freeze_required':['complete paired validation candidate grid','exact config hashes','partition hashes',
            'rule from separate calibration pilot before validation','common protocol/deployment checks'],
        'formal_test_selection_forbidden':True,'performance_claim_allowed':False})
    write('configs/comparisons/ablations.json',{
        'status':'virtual_only','deployment_allowed':False,'all_keep':['noisy confirmation','STOP','budget','contract versions'],
        'methods':[
            {'id':'A_no_probe','change':'remove known-target probes'},
            {'id':'A_factorized','change':'factor joint after every predicted and observed update'},
            {'id':'A_periodic','change':'replace active choice by tuned fixed-period rule'},
            {'id':'A_no_task_consequence','change':'planning uses symmetric goal 0/1 loss; evaluator retains original outcomes'},
            {'id':'A_mix_sources','change':'deliberately treat probe response as task answer; world emissions unchanged'},
            {'id':'A_pseudo_labels','change':'own predicted goal/mode labels adapt task emission counts; never external truth'},
            {'id':'A_ignore_probe_cost','change':'planning probe cost zero; actual time and evaluator cost still charged'},
            {'id':'A_fixed_duration','change':'use only 1s final windows; no prefix reuse'}],
        'future_environment_sensitivities':['wrong channel prior','wrong scene prior','probe noncompliance'],
        'note':'environment sensitivities and complete fault/episode sweep belong to step12; not run here'})
    write('execution/10_references.json',{
        'checked_on':'2026-10-16','purpose':'adaptation provenance, not claims of full reproductions',
        'sources':[
            {'id':'NOIR','url':'https://proceedings.mlr.press/v229/zhang23f.html','verified':'publisher abstract: objects/actions and user-adapted intent prediction'},
            {'id':'NOIR2','url':'https://arxiv.org/abs/2511.20848','verified':'author preprint metadata; adaptation not full reproduction'},
            {'id':'RBC','url':'https://pubmed.ncbi.nlm.nih.gov/27416602/','verified':'author abstract: recursive Bayes and MMI with confusion matrices'},
            {'id':'dynamic_stopping','url':'https://www.imstat.org/publications/aoas/aoas_20_1/aoas_20_1.pdf','verified':'publisher abstract via search: early stopping and dynamic stimulus selection; actor-critic/GP not reproduced'},
            {'id':'dynamic_stopping_code','url':'https://github.com/BangyaoZhao/P300RL','verified':'author repository reproducibility description; not executed'},
            {'id':'semantic_VoI','url':'https://arxiv.org/abs/2601.06407','verified':'author abstract: task utility and clarification effort'},
        ],'access_notes':['initial PMC request hit challenge; RBC verified via PubMed',
            'dynamic stopping DOI direct access failed; publisher search excerpt and author repository found']})
    print('Development registry, ablation, selection and reference configurations written.')
