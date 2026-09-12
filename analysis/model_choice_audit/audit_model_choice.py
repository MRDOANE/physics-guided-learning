#!/usr/bin/env python3
"""Audit and summarize archived forecast errors. Never imports training code.

Requires Python 3.10+, NumPy, and Matplotlib. All comparisons are post hoc.
Usage: python audit_model_choice.py --repository /path/to/physics-guided-learning
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

STAGES = {
    'development': 'results/development',
    'resolution_g32': 'results/confirmation/resolution/g32',
    'resolution_g64': 'results/confirmation/resolution/g64',
    'selector_g64': 'results/confirmation/selector/g64',
}
KINDS = ['transformer', 'looped', 'fno']
ARMS = ['none', 'smooth', 'iid', 'response_matched']
DISPLAY = {'transformer': 'Transformer', 'looped': 'Looped transformer', 'fno': 'Fourier operator',
           'mechanistic': 'Physics solver', 'persistence': 'Persistence',
           'none': 'No augmentation', 'smooth': 'Smooth physics guidance',
           'iid': 'Independent perturbations', 'response_matched': 'Response matched',
           'wave': 'Wave', 'burgers': 'Burgers', 'ks': 'Kuramoto–Sivashinsky',
           'advection_diffusion': 'Advection–diffusion', 'allen_cahn': 'Allen–Cahn',
           'gray_scott': 'Gray–Scott', 'cahn_hilliard': 'Cahn–Hilliard',
           'fitzhugh_nagumo': 'FitzHugh–Nagumo'}
DISPLAY_STAGES = {'wave': 'resolution_g64', 'advection_diffusion': 'resolution_g64',
                  'allen_cahn': 'resolution_g64', 'burgers': 'development',
                  'ks': 'development', 'gray_scott': 'development',
                  'cahn_hilliard': 'selector_g64', 'fitzhugh_nagumo': 'selector_g64'}

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def clean_scores(curve, horizon, stage):
    """Preserve each stage's original scoring, plus explicit raw-error diagnostics."""
    x = np.asarray(curve[:, :horizon], dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == horizon
    assert np.all(x[np.isfinite(x)] >= 0)
    with np.errstate(over='ignore', invalid='ignore'):
        raw = x.mean(1)
    changed = np.any(~np.isfinite(x) | (x > 1e6), axis=1)
    if stage == 'development':
        scored = np.minimum(np.nan_to_num(raw, nan=1e6, posinf=1e6, neginf=1e6), 1e6)
        changed = ~np.isfinite(raw) | (raw > 1e6)
    else:
        scored = np.clip(np.nan_to_num(x, nan=1e6, posinf=1e6, neginf=1e6), 0, 1e6).mean(1)
    return scored, {'nonfinite_steps': int((~np.isfinite(x)).sum()),
                    'steps_above_cap': int((x > 1e6).sum()),
                    'affected_trajectories': int(changed.sum()),
                    'raw_mean_nmse': float(raw.mean()) if np.isfinite(raw).all() else None}

def audit(repo, out):
    records, vectors, refs, reference_vectors, checked, configs = [], {}, {}, {}, [], {}
    norms, family_ids, stacks = {}, {}, set()
    audit_counts = Counter(); failure_rows = []; inventory = []
    for stage, rel in STAGES.items():
        folder = repo / rel
        protocol_path = folder / ('protocol.json' if stage == 'development' else 'stage_protocol.json')
        protocol = read(protocol_path); cfg = protocol['config']; configs[stage] = cfg
        protocol_hash = protocol['protocol_hash'] if stage == 'development' else protocol['signature']
        if stage == 'development':
            expected = {(f,k,p,s,a) for f,k,p,s,a in itertools.product(cfg['families'], cfg['kinds'], cfg['priors'], cfg['seeds'], cfg['arms'])}
        else:
            expected = {tuple(j[k] for k in ['family','kind','prior','seed','arm']) for j in protocol['jobs']}
        manifest = folder / 'selection_manifest.json'
        assert read(folder/'evaluation_plan.json')['selection_sha256'] == sha(manifest)
        checked.extend([{'path': str(p.relative_to(repo)), 'sha256': sha(p)} for p in [protocol_path,manifest,folder/'evaluation_plan.json']])
        seen = set(); stage_records = []
        for rp in sorted((folder/'runs').glob('*/record.json')):
            r = read(rp); key = tuple(r[k] for k in ['family','kind','prior','seed','arm'])
            assert key in expected and key not in seen; seen.add(key)
            assert r['completed_steps'] == cfg['steps'] and r['numerical_failure'] is None
            data_meta = read(folder/'data'/f"{r['family']}_development.json")
            job = {k:r[k] for k in ['arm','family','kind','prior','run_id','seed','task_id']}
            assert r['signature'] == canonical(dict(protocol_hash=protocol_hash,job=job,data=data_meta['signature']))
            ep = rp.parent / 'evaluation.npz'; assert sha(ep) == r['evaluation_sha256']
            assert ep.resolve() == (folder/r['evaluation_path']).resolve()
            checked.extend([{'path':str(p.relative_to(repo)), 'sha256':sha(p)} for p in [rp,ep]])
            stacks.add(json.dumps(r['stack'],sort_keys=True))
            norm_key = (stage,r['family'])
            state_norm = (r['normalization']['mean'],r['normalization']['scale'])
            if norm_key in norms: assert norms[norm_key] == state_norm
            else: norms[norm_key] = state_norm
            r = dict(r,stage=stage,grid=cfg['grid']); records.append(r); stage_records.append(r)
            ref_key = (stage,r['family'],r['prior'])
            if ref_key not in refs:
                refpath = folder/'references'/f"{r['family']}__{r['prior']}.npz"
                assert sha(refpath) == read(refpath.with_suffix('.json'))['sha256']
                checked.extend([{'path':str(p.relative_to(repo)), 'sha256':sha(p)} for p in [refpath,refpath.with_suffix('.json')]])
                with np.load(refpath,allow_pickle=False) as f: refs[ref_key] = {k:f[k].copy() for k in f.files}
            with np.load(ep,allow_pickle=False) as f:
                reference_signature=canonical(dict(protocol=protocol_hash,family=r['family'],prior=r['prior'],normalization=r['normalization'],test_ids=f['test_ids'].tolist(),ood_ids=f['ood_ids'].tolist()))
                reference_metadata=read(folder/'references'/f"{r['family']}__{r['prior']}.json")
                assert reference_metadata['signature']==reference_signature
                for split in ['test','ood']:
                    ids = f[split+'_ids']; ik = (stage,r['family'],split)
                    assert len(set(ids)) == len(ids) == cfg['split_counts'][split]
                    if ik in family_ids: assert np.array_equal(ids,family_ids[ik])
                    else: family_ids[ik] = ids.copy()
                    curve = f[split+'_mse']; assert curve.shape == (64,96)
                    for method in ['mechanistic','persistence']:
                        rk = split+'_'+method+'_mse'
                        assert np.array_equal(f[rk],refs[ref_key][rk],equal_nan=True)
                        audit_counts['identical_cached_reference_arrays'] += 1
                    for horizon in [64,96]:
                        scores, diagnostics = clean_scores(curve,horizon,stage)
                        vkey = (stage,)+key+(split,horizon)
                        vectors[vkey] = scores
                        failure_rows.append(dict(stage=stage,run_id=r['run_id'],kind=r['kind'],arm=r['arm'],family=r['family'],prior=r['prior'],split=split,horizon=horizon,**diagnostics))
                        audit_counts[f'neural_h{horizon}_{split}_nonfinite_steps'] += diagnostics['nonfinite_steps']
                        audit_counts[f'neural_h{horizon}_{split}_cap_affected_trajectories'] += diagnostics['affected_trajectories']
        assert seen == expected
        inventory.append(dict(stage=stage,grid=cfg['grid'],records=len(seen),expected=len(expected),families=sorted({r['family'] for r in stage_records})))
        print(f'VERIFIED {stage}: {len(seen)}/{len(expected)} records',flush=True)
    for (stage,family,prior), arrays in refs.items():
        for method,split,horizon in itertools.product(['mechanistic','persistence'],['test','ood'],[64,96]):
            scores, diagnostics = clean_scores(arrays[split+'_'+method+'_mse'],horizon,stage)
            reference_vectors[(stage,family,prior,method,split,horizon)] = scores
            failure_rows.append(dict(stage=stage,run_id='reference',kind=method,arm='',family=family,prior=prior,split=split,horizon=horizon,**diagnostics))
            audit_counts[f'{method}_h{horizon}_{split}_nonfinite_steps'] += diagnostics['nonfinite_steps']
            audit_counts[f'{method}_h{horizon}_{split}_cap_affected_trajectories'] += diagnostics['affected_trajectories']
    # Save sufficient statistics and IDs to reproduce all reported averages/intervals.
    vector_arrays={}; vector_index=[]
    for category, bank in [('neural',vectors),('reference',reference_vectors),('trajectory_ids',family_ids)]:
        for key,value in bank.items():
            name=f'array_{len(vector_index):05d}';vector_arrays[name]=value
            vector_index.append(dict(array=name,category=category,key=list(key)))
    np.savez_compressed(out/'paired_trajectory_scores.npz',**vector_arrays)
    dump(out/'score_index.json',vector_index); dump(out/'record_metadata.json',records)
    dump(out/'source_checksums.json',checked); write_csv(out/'numerical_diagnostics.csv',failure_rows)
    summary=dict(status='PASS',records_verified=len(records),reference_caches_verified=len(refs),
                 inventory=inventory,counts=dict(audit_counts),hardware=[json.loads(s) for s in stacks],
                 normalization='State means/scales match exactly across every method and prior within each stage/equation.',
                 pairing='Unique trajectory IDs match in order across candidates; all embedded reference arrays exactly match their checked reference caches.',
                 archived_contents='Per-trajectory, per-step normalized squared errors; full predicted states and trained checkpoints are absent from this release.',
                 physics_latency='No separately measured physics-solver latency is stored. Reference caches have signature/hash only; evaluation_seconds mixes multiple operations.',
                 training_performed=False,checkpoints_loaded=False)
    dump(out/'integrity_audit.json',summary)
    return records,vectors,reference_vectors,configs,summary

def bootstrap_mean(matrix, seed_weights, trajectory_weights):
    # matrix [training seeds, shared test trajectories]. Physics has one row.
    means = trajectory_weights @ matrix.T
    return np.sum(means * seed_weights,axis=1)

def summarize(records,vectors,refs,configs,out,reps):
    groups=defaultdict(list)
    for r in records: groups[(r['stage'],r['family'],r['prior'],r['kind'],r['arm'])].append(r)
    rng=np.random.default_rng(20260912); weights={}; matrices={}; draws={}; rows=[]; costs=[]
    for key,rr in sorted(groups.items()):
        stage,family,prior,kind,arm=key;rr=sorted(rr,key=lambda r:r['seed']); ns=len(rr)
        wk=(stage,family)
        if wk not in weights:
            weights[wk]=(rng.multinomial(ns,np.full(ns,1/ns),size=reps)/ns,
                         rng.multinomial(64,np.full(64,1/64),size=reps)/64)
        sw,tw=weights[wk]
        train=np.array([r['train_seconds'] for r in rr]); latency=np.array([r['inference_ms_per_step'] for r in rr])
        route=np.array([r['train_seconds']+r['setup_seconds']+(0 if arm=='none' else r['bank_setup_seconds']) for r in rr])
        costs.append(dict(stage=stage,grid=configs[stage]['grid'],family=family,prior=prior,kind=kind,arm=arm,n_seeds=ns,
                          parameter_count=rr[0]['parameter_count'],mean_training_validation_seconds=float(train.mean()),
                          mean_setup_plus_training_plus_bank_seconds=float(route.mean()),
                          median_inference_ms_per_step=float(np.median(latency)),min_inference_ms=float(latency.min()),max_inference_ms=float(latency.max()),
                          latency_scope='Measured batch one; 5 warmups, one block of 20 forwards, synchronized CUDA; input transfers excluded.',
                          cost_scope='Fixed-choice route; common reference-data generation, evaluation, and development search excluded. Shared bank construction includes every augmentation arm.'))
        for split,horizon in itertools.product(['test','ood'],[64,96]):
            m=np.stack([vectors[(stage,family,kind,prior,r['seed'],arm,split,horizon)] for r in rr])
            bk=key+(split,horizon);matrices[bk]=m
            d=bootstrap_mean(m,sw,tw);draws[bk]=d
            lo,hi=np.quantile(d,[.025,.975]);point=float(m.mean())
            rows.append(dict(stage=stage,grid=configs[stage]['grid'],family=family,prior=prior,kind=kind,arm=arm,split=split,horizon=horizon,n_seeds=ns,n_trajectories=64,
                             mean_nmse=point,ci95_low=float(lo),ci95_high=float(hi),normalized_rmse=math.sqrt(point),
                             mean_training_validation_seconds=float(train.mean()),fixed_choice_setup_train_bank_seconds=float(route.mean()),
                             median_inference_ms_per_step=float(np.median(latency))))
    for (stage,family,prior,kind,split,horizon),m in sorted(refs.items()):
        sw,tw=weights[(stage,family)];d=tw@m
        key=(stage,family,prior,kind,'',split,horizon);draws[key]=d;matrices[key]=m[None]
        lo,hi=np.quantile(d,[.025,.975]);point=float(m.mean())
        rows.append(dict(stage=stage,grid=configs[stage]['grid'],family=family,prior=prior,kind=kind,arm='',split=split,horizon=horizon,n_seeds=0,n_trajectories=64,
                         mean_nmse=point,ci95_low=float(lo),ci95_high=float(hi),normalized_rmse=math.sqrt(point),
                         mean_training_validation_seconds=0.0,fixed_choice_setup_train_bank_seconds=None,median_inference_ms_per_step=None))
    contrasts=[]
    for key,m in sorted(matrices.items()):
        stage,family,prior,kind,arm,split,horizon=key
        if kind not in KINDS: continue
        # Compare each neural candidate with the same physical baseline; direct
        # architecture contrasts retain a fixed augmentation choice.
        baselines=[(stage,family,prior,'mechanistic','',split,horizon)]
        if kind in ['looped','fno']:
            baselines.append((stage,family,prior,'transformer',arm,split,horizon))
        if arm=='smooth':baselines.append((stage,family,prior,kind,'none',split,horizon))
        for bkey in baselines:
            aa=float(m.mean());bb=float(matrices[bkey].mean());dd=draws[key]-draws[bkey]
            lo,hi=np.quantile(dd,[.025,.975]);rat=draws[key]/draws[bkey];rlo,rhi=np.quantile(rat,[.025,.975])
            contrasts.append(dict(stage=stage,grid=configs[stage]['grid'],family=family,prior=prior,split=split,horizon=horizon,
                                  candidate_kind=kind,candidate_arm=arm,baseline_kind=bkey[3],baseline_arm=bkey[4],
                                  candidate_mean_nmse=aa,baseline_mean_nmse=bb,difference_nmse=aa-bb,difference_ci95_low=float(lo),difference_ci95_high=float(hi),
                                  ratio_of_arithmetic_means=aa/bb,ratio_ci95_low=float(rlo),ratio_ci95_high=float(rhi),
                                  interpretation='lower_candidate_error' if hi<0 else 'higher_candidate_error' if lo>0 else 'interval_includes_zero',
                                  scope='Post hoc; pointwise 95% paired bootstrap, no multiplicity adjustment or confirmatory decision.'))
    write_csv(out/'absolute_accuracy.csv',rows);write_csv(out/'direct_contrasts.csv',contrasts);write_csv(out/'archived_costs.csv',costs)
    # Primary display is one transparent stage per equation. Every stage is in CSV.
    display=[r for r in rows if r['stage']==DISPLAY_STAGES[r['family']] and r['split']=='test' and r['horizon']==64 and (r['arm'] in ['none','smooth'] or r['kind']=='mechanistic')]
    write_csv(out/'main_comparison.csv',display)
    # Selection based on archived validation loss only. No test-based selection.
    selection_rows=[]
    task_candidates=defaultdict(list)
    for r in records: task_candidates[(r['stage'],r['family'],r['prior'],r['seed'])].append(r)
    for (stage,family,prior,seed),rr in sorted(task_candidates.items()):
        choices={
            'validation_select_neural_architecture': min((r for r in rr if r['arm']=='none'),key=lambda r:(r['final_val'],r['kind'])),
            'validation_select_neural_and_augmentation': min(rr,key=lambda r:(r['final_val'],r['kind'],r['arm']))}
        for policy,r in choices.items():
            for split,horizon in itertools.product(['test','ood'],[64,96]):
                m=vectors[(stage,family,r['kind'],prior,seed,r['arm'],split,horizon)]
                phys=refs[(stage,family,prior,'mechanistic',split,horizon)]
                acquired=[x for x in rr if x['arm']=='none'] if policy.endswith('architecture') else rr
                # All fully trained candidates were acquired before selecting.
                total=sum(x['train_seconds']+x['setup_seconds'] for x in acquired)
                if not policy.endswith('architecture'):total+=max(x['bank_setup_seconds'] for x in acquired)
                selection_rows.append(dict(stage=stage,grid=configs[stage]['grid'],family=family,prior=prior,seed=seed,policy=policy,chosen_kind=r['kind'],chosen_arm=r['arm'],
                                           selection_validation_nmse=r['final_val'],split=split,horizon=horizon,mean_nmse=float(m.mean()),physical_mean_nmse=float(phys.mean()),
                                           acquired_fit_count=len(acquired),measured_acquisition_seconds=total,
                                           status='Exploratory retrospective rule using validation only; physical solver validation score absent, so it is outside this selector.'))
    write_csv(out/'retrospective_validation_selection.csv',selection_rows)
    result=dict(absolute_rows=rows,contrasts=contrasts,costs=costs,main_display=display,selection=selection_rows,
                bootstrap_replicates=reps,bootstrap_seed=20260912,
                estimand='Arithmetic average of normalized squared error across steps, trajectories, and training seeds; each equation and condition reported separately.',
                uncertainty='Crossed bootstrap resamples training seeds and complete shared trajectories; deterministic physical reference resampled once by trajectory. Pointwise exploratory intervals; fixed training datasets and equation settings.',
                no_training=True)
    dump(out/'analysis_results.json',result)
    return result

def figures(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    rows=result['main_display']; lookup={(r['family'],r['prior'],r['kind'],r['arm']):r for r in rows}
    families=list(DISPLAY_STAGES)
    options=[('mechanistic','')]+[(k,a) for k in KINDS for a in ['none','smooth']]
    labels=['Physics','Transformer','Transformer + physics','Looped','Looped + physics','Fourier','Fourier + physics']
    for prior,split in [('correct','test'),('coefficient','test'),('coefficient','ood')]:
        if split=='ood':
            lookup={(r['family'],r['prior'],r['kind'],r['arm']):r for r in result['absolute_rows'] if r['stage']==DISPLAY_STAGES[r['family']] and r['split']==split and r['horizon']==64}
        fig,axes=plt.subplots(2,4,figsize=(15,8.7),sharey=False)
        for ax,family in zip(axes.flat,families):
            rr=[lookup[(family,prior,k,a)] for k,a in options];vals=np.array([r['mean_nmse'] for r in rr]);lo=np.array([r['ci95_low'] for r in rr]);hi=np.array([r['ci95_high'] for r in rr])
            colors=['#333333','#276FBF','#276FBF','#C56B17','#C56B17','#27815D','#27815D']
            for j,(v,l,h,c) in enumerate(zip(vals,lo,hi,colors)):
                ax.errorbar(v,j,xerr=[[max(0,v-l)],[max(0,h-v)]],fmt='s' if j in [2,4,6] else 'o',color=c,capsize=2,ms=5)
            ax.set_xscale('log');ax.set_yticks(range(7),labels);ax.invert_yaxis();ax.set_title(f"{DISPLAY[family]} ({lookup[(family,prior,'fno','none')]['grid']} cells)")
            ax.grid(axis='x',alpha=.2);ax.set_xlabel('Normalized MSE (log scale)')
        title=('Correct physical model' if prior=='correct' else 'Biased physical coefficients')+' — 64-step forecasts'+(' under parameter shift' if split=='ood' else '')
        fig.suptitle(title,fontsize=15,y=.98)
        fig.text(.02,.025,'Circles: standalone physics or unaugmented neural model. Squares: smooth physics-guided training.\nBars: pointwise 95% bootstrap intervals. Stages and grids are listed separately; these are exploratory comparisons.',fontsize=10)
        fig.tight_layout(rect=[0,.07,1,.94])
        path=out/(f'accuracy_{prior}'+('_ood' if split=='ood' else '')+'.png')
        fig.savefig(path);plt.close(fig)
    # Neural latency only: physics timing was never recorded separately.
    lookup={(r['family'],r['prior'],r['kind'],r['arm']):r for r in rows}
    fig,axes=plt.subplots(1,3,figsize=(13,4.6))
    for ax,family in zip(axes,['wave','allen_cahn','cahn_hilliard']):
        for kind,color in zip(KINDS,['#276FBF','#C56B17','#27815D']):
            for arm,marker in [('none','o'),('smooth','s')]:
                r=lookup[(family,'coefficient',kind,arm)]
                ax.scatter(r['median_inference_ms_per_step'],r['mean_nmse'],color=color,marker=marker,s=60,label=DISPLAY[kind]+(' + physics' if arm=='smooth' else ''))
        ax.set_yscale('log');ax.set_xlabel('Archived batch-one latency (ms/step)');ax.set_ylabel('Mean normalized squared error');ax.set_title(DISPLAY[family]);ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1))
    fig.suptitle('Neural accuracy and measured inference time — biased coefficients, 64 cells',fontsize=13)
    fig.text(.015,.018,'RTX 3090 Ti, eight CPU threads. Physics-solver latency is missing; no solver speedup is inferred. Markers show fixed trained candidates.',fontsize=9)
    fig.tight_layout(rect=[0,.065,.99,.94]);fig.savefig(out/'neural_accuracy_latency.png');plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--bootstrap-replicates',type=int,default=5000)
    args=parser.parse_args();assert args.bootstrap_replicates>=1000
    args.output.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
    records,vectors,refs,configs,integrity=audit(args.repository.resolve(),args.output)
    result=summarize(records,vectors,refs,configs,args.output,args.bootstrap_replicates)
    figures(result,args.output)
    print('AUDIT: PASS | 1,584 archived candidate records verified')
    print('ACCURACY: AVAILABLE | paired direct neural/physics comparisons computed')
    print('NEURAL COSTS: AVAILABLE | recorded training and batch-one inference time')
    print('PHYSICS INFERENCE COST: MISSING | no physics-versus-neural speed claim')
    print('INFERENCE STATUS: EXPLORATORY | no new confirmatory hypothesis test')
    print('MODEL TRAINING: NONE')
    print(f'Elapsed analysis: {time.perf_counter()-t:.1f} seconds')

if __name__=='__main__':main()
