"""Development-only selection; no test outcomes accepted at this boundary."""
from collections import defaultdict
import math
import time
import numpy as np
from .selection import ARMS, FEATURES, KINDS, _fit, _predict, _nested_alpha, _best_fixed

ALPHAS=(0.1,1.,10.,100.)
MODES=('ridge','mean_probe','worst_probe','none')
ALLOWED={'task_id','family','kind','prior','seed','arm','features','probe_val','final_val',
         'train_seconds','probe_seconds','remaining_seconds','setup_seconds','diagnostic_seconds'}

def tasks_from_records(records, development=False):
    grouped=defaultdict(dict)
    for rec in records:
        if set(rec)-ALLOWED: raise ValueError('Selector input has forbidden fields: '+str(set(rec)-ALLOWED))
        r=dict(rec)
        if r['arm'] not in ARMS or r['kind'] not in KINDS: raise ValueError('Invalid candidate')
        if set(r['features'])!=set(FEATURES):raise ValueError('Invalid selector features')
        if not all(np.isfinite(v) for v in r['features'].values()):raise ValueError('Nonfinite selector features')
        if not development:r.pop('final_val',None)
        for key in ('probe_val','final_val'):
            if key in r:
                v=float(r[key]);r[key]=float(np.clip(v if np.isfinite(v) else 1e6,1e-12,1e6))
        if development and 'final_val' not in r:raise ValueError('Missing development validation label')
        if r['arm'] in grouped[r['task_id']]:raise ValueError('Duplicate selector candidate')
        grouped[r['task_id']][r['arm']]=r
    tasks=[]
    for tid,arms in sorted(grouped.items()):
        if set(arms)!=set(ARMS):raise ValueError('All four arms required for selection')
        b=arms['none']
        for r in arms.values():
            if any(r[k]!=b[k] for k in ('family','kind','prior','seed','features')):raise ValueError('Unpaired candidates')
        tasks.append(dict(task_id=tid,family=b['family'],kind=b['kind'],prior=b['prior'],seed=b['seed'],features=b['features'],candidates=arms))
    return tasks

def fit_bundle(tasks,alpha):
    families=sorted({t['family'] for t in tasks})
    return dict(model=_fit(tasks,alpha),family_omission_models=[_fit([t for t in tasks if t['family']!=f],alpha) for f in families],training_families=families)

def choose(bundle,task,mode):
    if mode=='none':return 'none'
    predictions={a:_predict(bundle['model'],task,a) for a in ARMS}
    if mode=='ridge':return min(ARMS,key=lambda a:predictions[a])
    values={a:[predictions[a]]+[_predict(m,task,a) for m in bundle['family_omission_models']] for a in ARMS}
    score={a:float(np.mean(v) if mode=='mean_probe' else max(v)) for a,v in values.items()}
    eligible=['none']+[a for a in ARMS[1:] if score[a]<0 and task['candidates'][a]['probe_val']<task['candidates']['none']['probe_val']]
    return min(eligible,key=lambda a:score[a])

def fit_selector(records):
    """Select the regularization of the family-omission/probe guard on development.

    The guard is a robustness heuristic, not a statistical safety guarantee.
    Hyperparameters use six already-observed equation families. Their CV scores
    are development summaries; only the two new equations are confirmation.
    """
    started=time.perf_counter();tasks=tasks_from_records(records,True)
    families=sorted({t['family'] for t in tasks});scores=[]
    if len(families)<4:raise ValueError('Four or more development families required')
    for alpha in ALPHAS:
        fold={f:fit_bundle([t for t in tasks if t['family']!=f],alpha) for f in families}
        for mode in MODES:
            losses=[];harms=[];augmentation=[]
            for f in families:
                ts=[t for t in tasks if t['family']==f]
                choices=[choose(fold[f],t,mode) for t in ts]
                logs=[math.log(t['candidates'][a]['final_val']/t['candidates']['none']['final_val']) for t,a in zip(ts,choices)]
                losses.append(float(np.mean(logs)));harms.append(float(np.mean(np.array(logs)>0)))
                augmentation.append(float(np.mean([a!='none' for a in choices])))
            scores.append(dict(alpha=alpha,mode=mode,mean_log_ratio=float(np.mean(losses)),se=float(np.std(losses,ddof=1)/np.sqrt(len(losses))),harm_fraction=float(np.mean(harms)),augmentation_fraction=float(np.mean(augmentation)),fold_log_ratios=dict(zip(families,losses))))
    # The v2 mechanism is specifically agreement between family-omission
    # predictions and observed probe loss. Other modes are development ablations,
    # not a hidden route to select a favorable confirmation result.
    selected=min([r for r in scores if r['mode']=='worst_probe'],key=lambda r:(r['mean_log_ratio'],r['alpha']))
    bundle=fit_bundle(tasks,selected['alpha'])
    original_alpha,original_cv=_nested_alpha(tasks,ALPHAS)
    fixed={k:_best_fixed([t for t in tasks if t['kind']==k])[0] for k in KINDS}
    return dict(method='family_omission_guard_v2',selected=selected,development_cv=scores,bundle=bundle,
                original_ridge=_fit(tasks,original_alpha),original_cv=original_cv,best_fixed=fixed,
                fit_seconds=time.perf_counter()-started,development_families=families,
                warning='CV used for tuning; no claim of external performance or calibrated safety from these scores.')

def decide(model,records):
    decisions=[]
    for task in tasks_from_records(records):
        ridge=min(ARMS,key=lambda a:_predict(model['original_ridge'],task,a))
        if _predict(model['original_ridge'],task,ridge)>math.log(.95):ridge='none'
        choices=dict(none='none',smooth='smooth',probe_best=min(ARMS,key=lambda a:task['candidates'][a]['probe_val']),
                     best_fixed=model['best_fixed'][task['kind']],original_ridge=ridge,
                     guarded=choose(model['bundle'],task,model['selected']['mode']))
        decisions.append({k:task[k] for k in ('task_id','family','kind','prior','seed')}|dict(choices=choices))
    return decisions
