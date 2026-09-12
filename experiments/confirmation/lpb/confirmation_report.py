"""Evidence, execution, and costs have separate labels; no minimum effect hurdle."""
import csv
import io
import json
import math
from pathlib import Path
import sys
import numpy as np
from .core import atomic_json,canonical_hash,sha256

def read(p):return json.loads(Path(p).read_text())

def paired_effect(pairs,reps=20000,seed=1901):
    """Paired crossed seed/trajectory bootstrap, fixed family/architecture/prior strata.

    Input per family: [fixed strata, training seeds, shared test trajectories].
    Training seeds and complete trajectories are resampled, never time steps.
    Reported uncertainty is conditional on these equations and training datasets.
    """
    if not pairs:raise ValueError('Missing paired outcomes')
    cleaned=[];points=[]
    for a,b in pairs:
        if a.shape!=b.shape or a.ndim!=3:raise ValueError('Unpaired outcome shapes')
        a=np.clip(np.nan_to_num(a,nan=1e6,posinf=1e6,neginf=1e6),1e-12,1e6)
        b=np.clip(np.nan_to_num(b,nan=1e6,posinf=1e6,neginf=1e6),1e-12,1e6)
        cleaned.append((a,b));points.append(float(np.log(a.mean(-1)/b.mean(-1)).mean()))
    point=float(np.mean(points));rng=np.random.default_rng(seed);draws=[]
    for start in range(0,reps,256):
        count=min(256,reps-start);values=[]
        ns=cleaned[0][0].shape[1];si=rng.integers(0,ns,(count,ns))
        for a,b in cleaned:
            if a.shape[1]!=ns:raise ValueError('Training seed inventory differs across families')
            n=a.shape[-1];weights=rng.multinomial(n,np.full(n,1/n),size=count)/n
            aa=np.einsum('kse,be->bks',a,weights);bb=np.einsum('kse,be->bks',b,weights)
            logs=np.log(aa/bb)
            values.append(np.take_along_axis(logs,si[:,None,:],axis=2).mean(axis=(1,2)))
        draws.extend(np.mean(values,axis=0).tolist())
    draws=np.array(draws);lo,hi=np.quantile(draws,[.025,.975])
    p=float((1+np.count_nonzero(np.abs(draws-point)>=abs(point)))/(reps+1))
    return dict(mse_ratio=float(np.exp(point)),ratio_ci95=[float(np.exp(lo)),float(np.exp(hi))],
                mse_reduction_percent=float(100*(1-np.exp(point))),reduction_ci95_percent=[float(100*(1-np.exp(hi))),float(100*(1-np.exp(lo)))],
                p_two_sided=p,bootstrap_replicates=reps,training_seeds=ns,test_trajectories_per_family=cleaned[0][0].shape[-1],
                scope='Fixed equations/architectures/prior regimes; paired seeds and complete trajectories; conditional on generated training datasets.')

def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p_two_sided']);last=0.
    for rank,i in enumerate(order):
        last=max(last,(len(rows)-rank)*rows[i]['p_two_sided']);rows[i]['p_holm']=min(1.,last)
    return rows

def label(row,expected='benefit',eligible=True):
    if not eligible:return 'NOT_EVALUATED'
    lo,hi=row['ratio_ci95'];sig=row.get('p_holm',row['p_two_sided'])<=.05
    if not sig or lo<=1<=hi:return 'YELLOW'
    beneficial=hi<1
    return 'GREEN' if beneficial==(expected=='benefit') else 'RED'

def summarize_stage(child):
    protocol=read(child/'stage_protocol.json');cfg=protocol['config'];rows=[];arrays={};errors=[]
    for job in protocol['jobs']:
        folder=child/'runs'/job['run_id'];rp=folder/'record.json'
        if not rp.exists():continue
        r=read(rp)
        if any(r.get(k)!=v for k,v in job.items()):errors.append('Job metadata mismatch: '+job['run_id']);continue
        if r.get('completed_steps')!=cfg['steps'] or not r.get('evaluation_sha256'):continue
        ep=folder/'evaluation.npz'
        if not ep.exists() or sha256(ep)!=r['evaluation_sha256']:errors.append('Evaluation checksum mismatch: '+job['run_id']);continue
        expected=canonical_hash(dict(protocol_hash=protocol['signature'],job=job,data=read(child/'data'/f"{job['family']}_development.json")['signature']))
        if r.get('signature')!=expected:errors.append('Record signature mismatch: '+job['run_id']);continue
        with np.load(ep,allow_pickle=False) as f:
            arr={k:f[k].copy() for k in f.files}
        for split in ('test','ood'):
            if arr[split+'_mse'].shape!=(cfg['split_counts'][split],cfg['eval_horizon']):errors.append('Evaluation shape mismatch: '+job['run_id'])
        arrays[(r['task_id'],r['arm'])]=arr;rows.append(r)
    frozen=child/'selection_manifest.json';plan=child/'evaluation_plan.json'
    if rows:
        if not frozen.exists() or not plan.exists():errors.append('Missing freeze/evaluation lock')
        else:
            manifest=read(frozen)
            if read(plan)['selection_sha256']!=sha256(frozen):errors.append('Freeze checksum mismatch')
            if manifest['inputs_sha256']!=sha256(child/'selection_inputs.json'):errors.append('Frozen input checksum mismatch')
            if manifest['selector_model_sha256']!=sha256(child.parents[1]/'selector_model.json'):errors.append('Selector model checksum mismatch')
            if manifest['protocol_signature']!=protocol['signature']:errors.append('Frozen protocol mismatch')
            if child.parent.name=='selector':
                from .guarded import decide
                replay=decide(read(child.parents[1]/'selector_model.json'),read(child/'selection_inputs.json'))
                if replay!=manifest['decisions']:errors.append('Selector choices do not reproduce from frozen inputs')
        audit=read(child/'numerical_audit.json')
        if not all(r['passed'] for r in audit['temporal']):errors.append('Temporal reference check failed')
        ids={}
        for r in rows:
            for split in ('test','ood'):
                observed=arrays[(r['task_id'],r['arm'])][split+'_ids']
                key=(r['family'],split)
                if key in ids and not np.array_equal(ids[key],observed):errors.append('Unpaired test trajectory identities')
                ids[key]=observed
    expected=len(protocol['jobs']);complete=len(rows)==expected and not errors
    return dict(complete=complete,verified=len(rows),expected=expected,errors=errors,rows=rows,arrays=arrays,config=cfg)

def tensor(stage,family,priors,arm_or_policy,split='test',decisions=None,horizon=None):
    cfg=stage['config'];horizon=horizon or cfg['primary_horizon'];out=[]
    seeds=cfg['resolution_seeds'] if decisions is None else cfg['confirmation_seeds']
    for kind in cfg['kinds']:
        for prior in priors:
            per_seed=[]
            for seed in seeds:
                tid=f'{family}__{kind}__{prior}__s{seed}'
                arm=arm_or_policy if decisions is None else decisions[tid]['choices'][arm_or_policy]
                raw=stage['arrays'][(tid,arm)][split+'_mse'][:,:horizon]
                raw=np.clip(np.nan_to_num(raw,nan=1e6,posinf=1e6,neginf=1e6),0,1e6)
                per_seed.append(raw.mean(1))
            out.append(np.stack(per_seed))
    return np.stack(out)

def route_cost(candidates,arm,policy,guard_mode):
    """Counterfactual route reconstructed from measured component times."""
    if policy in ('none','smooth','best_fixed') or policy=='guarded' and guard_mode=='none':
        r=candidates[arm]
        return r['train_seconds']+r['setup_seconds']+(0 if arm=='none' else r['bank_setup_seconds'])
    probes=sum(r['probe_seconds']+r['probe_setup_seconds'] for r in candidates.values())
    chosen=candidates[arm]
    return probes+chosen['remaining_seconds']+chosen['continuation_setup_seconds']+max(r['bank_setup_seconds'] for r in candidates.values())

def timing(stages,output):
    training=setup=bank=cpu_data=cpu_audit=evaluation=0.;updates=0;by_kind={}
    for child,st in stages:
        all_records=[read(p) for p in (child/'runs').glob('*/record.json')]
        seen=set()
        for r in all_records:
            training+=r['train_seconds'];setup+=r['setup_seconds'];evaluation+=r.get('evaluation_seconds',0);updates+=r['completed_steps']
            key=(r['family'],r['prior'],r['seed'])
            if key not in seen:bank+=r['bank_setup_seconds'];seen.add(key)
            d=by_kind.setdefault(r['kind'],dict(seconds=0.,updates=0))
            d['seconds']+=r['train_seconds'];d['updates']+=r['completed_steps']
        for p in (child/'data').glob('*.json'):cpu_data+=read(p)['generation_seconds']
        if (child/'numerical_audit.json').exists():cpu_audit+=read(child/'numerical_audit.json')['seconds']
    execution=read(output/'execution.json') if (output/'execution.json').exists() else {}
    wall=execution.get('wall_seconds',0.);rate=execution.get('hourly_rate',0.)
    for d in by_kind.values():d['seconds_per_update_including_validation']=d['seconds']/max(d['updates'],1)
    return dict(wall_seconds=wall,candidate_training_and_validation_seconds=training,model_setup_seconds=setup,shared_bank_construction_seconds=bank,
                cpu_reference_generation_seconds=cpu_data,cpu_numerical_audit_seconds=cpu_audit,evaluation_seconds=evaluation,by_architecture=by_kind,
                compute_cost_usd=wall/3600*rate if rate else None,hourly_rate=rate,
                interpretation='Component timings include host overhead and validation; they are not GPU utilization measurements. Dataset generation/audits run on CPU. Neural training runs on the chosen device. Initial expectations should be replaced with timings from your own GPU benchmark.')

def report(output,print_output=True):
    output=Path(output);p=output/'protocol.json'
    if not p.exists():raise ValueError('No experiment protocol in '+str(output))
    protocol=read(p);cfg=protocol['config'];eligible=cfg['profile']=='full';stages=[];errors=[]
    for child in sorted(output.glob('*/g*')):
        if (child/'stage_protocol.json').exists():
            st=summarize_stage(child);stages.append((child,st));errors.extend(st['errors'])
    resolution=[];selector=[];selector_by_family=[];selector_by_prior=[];secondary=[];costs=[]
    for child,st in stages:
        if not st['complete']:continue
        cc=st['config']
        if child.parent.name=='resolution':
            for family in cc['resolution_families']:
                for prior in ('correct','coefficient'):
                    a=tensor(st,family,[prior],'smooth');b=tensor(st,family,[prior],'none')
                    result=paired_effect([(a,b)],cc['bootstrap_replicates'],cfg['bootstrap_seed'])
                    expected='harm' if prior=='coefficient' and family in ('wave','advection_diffusion') else 'benefit'
                    resolution.append(dict(family=family,prior=prior,grid=cc['grid'],expected_direction=expected,**result))
        elif child.parent.name=='selector':
            decisions={r['task_id']:r for r in read(child/'selection_manifest.json')['decisions']}
            for baseline in ('none','probe_best'):
                pairs=[]
                for family in cc['confirmation_families']:
                    a=tensor(st,family,['correct','coefficient','structural'],'guarded',decisions=decisions)
                    b=tensor(st,family,['correct','coefficient','structural'],baseline,decisions=decisions)
                    pairs.append((a,b))
                    r=paired_effect([(a,b)],cc['bootstrap_replicates'],cfg['bootstrap_seed'])
                    selector_by_family.append(dict(family=family,baseline=baseline,**r,role='descriptive family breakdown; not a separate confirmatory test'))
                r=paired_effect(pairs,cc['bootstrap_replicates'],cfg['bootstrap_seed'])
                selector.append(dict(baseline=baseline,**r))
                for prior in ('correct','coefficient','structural'):
                    pp=[(tensor(st,f,[prior],'guarded',decisions=decisions),tensor(st,f,[prior],baseline,decisions=decisions)) for f in cc['confirmation_families']]
                    rr=paired_effect(pp,cc['bootstrap_replicates'],cfg['bootstrap_seed'])
                    selector_by_prior.append(dict(prior=prior,baseline=baseline,**rr,role='descriptive breakdown to distinguish equation transfer from new structural prior errors'))
            for split,h in [('test',cc['eval_horizon']),('ood',cc['primary_horizon'])]:
                for baseline in ('none','probe_best','best_fixed','original_ridge','smooth'):
                    pairs=[(tensor(st,f,['correct','coefficient','structural'],'guarded',split,decisions,h),tensor(st,f,['correct','coefficient','structural'],baseline,split,decisions,h)) for f in cc['confirmation_families']]
                    ratio=float(np.exp(np.mean([np.log(np.maximum(a.mean(-1),1e-12)/np.maximum(b.mean(-1),1e-12)).mean() for a,b in pairs])))
                    secondary.append(dict(split=split,horizon=h,baseline=baseline,mse_ratio=ratio,role='descriptive sensitivity; no confirmatory label'))
            model=read(output/'selector_model.json');policies=['none','smooth','probe_best','best_fixed','original_ridge','guarded']
            for policy in policies:
                times=[]
                for tid,d in decisions.items():
                    candidates={r['arm']:r for r in st['rows'] if r['task_id']==tid}
                    times.append(route_cost(candidates,d['choices'][policy],policy,model['selected']['mode']))
                costs.append(dict(policy=policy,mean_target_training_seconds=float(np.mean(times)),median_target_training_seconds=float(np.median(times)),
                                  interpretation='Reconstructed from measured component times; full bank was actually trained. Includes probe/bank/setup work, excludes shared test evaluation. No claim of measured deployment speedup.'))
    # Two distinct prespecified hypothesis families. No fishing across grids.
    primary_resolution=[r for r in resolution if r['grid']==64]
    holm(primary_resolution);holm(selector)
    for r in resolution:r['label']=label(r,r['expected_direction'],eligible and r['grid']==64) if r['grid']==64 else 'DESCRIPTIVE'
    for r in selector:r['label']=label(r,'benefit',eligible)
    expected_total=0;expected_by_stage={}
    for stage in ('resolution','selector'):
        grids=cfg['resolution_grids'] if stage=='resolution' else [cfg['confirmation_grid']]
        from .confirmation_run import inventory
        expected_by_stage[stage]=sum(len(list(inventory(cfg,stage,g))) for g in grids)
        expected_total+=expected_by_stage[stage]
    verified=sum(st['verified'] for _,st in stages)
    execution=read(output/'execution.json') if (output/'execution.json').exists() else {}
    requested=execution.get('requested_stages',['resolution','selector'])
    requested_expected=sum(expected_by_stage[s] for s in requested)
    requested_verified=sum(st['verified'] for ch,st in stages if ch.parent.name in requested)
    completion='RED' if errors else ('GREEN' if requested_verified==requested_expected else 'YELLOW')
    failures=sum(r['numerical_failure'] is not None for _,st in stages for r in st['rows'])
    nonfinite=sum(int((~np.isfinite(a['test_mse'])).sum()) for _,st in stages for a in st['arrays'].values())
    result=dict(profile=cfg['profile'],completion=dict(label=completion,verified=requested_verified,expected=requested_expected,requested_stages=requested),
                full_package_coverage=dict(verified=verified,expected=expected_total),
                stage_completion={str(ch.relative_to(output)):dict(complete=st['complete'],verified=st['verified'],expected=st['expected']) for ch,st in stages},
                scientific_eligibility='ELIGIBLE_COMPLETED_STAGES' if eligible and not errors and any(st['complete'] for _,st in stages) else 'NOT_EVALUATED',
                evidence_legend={'GREEN':'Evidence supports the stated direction.','YELLOW':'Evidence is inconclusive; not acceptance of a null or proof of no effect.','RED':'Evidence supports the opposite direction; not a judgment of paper quality.','NOT_EVALUATED':'Smoke, benchmark, incomplete, or integrity failure.'},
                resolution=resolution,selector_primary=selector,selector_family_descriptive=selector_by_family,selector_prior_descriptive=selector_by_prior,secondary=secondary,costs=costs,timing=timing(stages,output),errors=errors,
                numerical_training_failures=failures,nonfinite_test_array_cells=nonfinite,
                statistical_policy='Two-sided paired bootstrap p-values; Holm control at 0.05 across six high-resolution replication comparisons, separately across two primary selector comparisons. Displayed intervals are pointwise 95%, not simultaneous. No effect-size threshold, win-count hurdle, noninferiority margin, or overall scientific pass/fail.',
                caveats=['Single training/validation dataset per equation and grid; seed/trajectory uncertainty is conditional on it.','Two new equations do not establish universal cross-family transfer.','Same simulator family supplies labels and physical priors; no real-world validation claim.','Prediction/augmentation distinctions are not four mutually exclusive model classes.','Architectures share a step budget, not equal parameters or FLOPs.'],
                exit_code=2 if errors else 0)
    if errors:
        for row in resolution+selector:row['label']='NOT_EVALUATED'
    provenance=output/'source/development/provenance.json'
    if provenance.exists():
        historical=read(provenance)
        result['historical_development']=dict(candidate_training_seconds=historical['historical_candidate_training_seconds'],archive_wall_seconds=historical['historical_archive_wall_seconds'],reused_records=historical['rows'],role='Already incurred development cost; not part of this new run or a free deployment input.')
    folder=output/'reports';folder.mkdir(exist_ok=True)
    atomic_json(folder/'report.json',result)
    lines=['PHYSICS CONFIRMATION RESULTS',f'COMPLETION: {completion} | {requested_verified}/{requested_expected} requested candidate records verified',
           'Requested stages: '+', '.join(requested),f'Full-package coverage: {verified}/{expected_total} (unrequested stages are optional).',
           'Scientific evidence: '+result['scientific_eligibility'],
           'No minimum improvement requirement; no required count of successful systems.',
           'GREEN=support | YELLOW=inconclusive | RED=opposite direction | NOT_EVALUATED=not scientific evidence.']
    if errors:lines.extend('INTEGRITY ERROR: '+e for e in errors)
    if not primary_resolution:lines.append('RESOLUTION EVIDENCE: NOT_EVALUATED | stage not completed')
    for r in primary_resolution:
        lo,hi=r['reduction_ci95_percent'];lines.append(f"RESOLUTION {r['family']}/{r['prior']} | {r['label']} | expected {r['expected_direction']} | MSE reduction {r['mse_reduction_percent']:+.2f}% [95% CI {lo:+.2f}, {hi:+.2f}] | Holm p={r['p_holm']:.4f}")
    if not selector:lines.append('SELECTOR EVIDENCE: NOT_EVALUATED | stage not completed')
    for r in selector:
        lo,hi=r['reduction_ci95_percent'];lines.append(f"SELECTOR vs {r['baseline']} | {r['label']} | MSE reduction {r['mse_reduction_percent']:+.2f}% [95% CI {lo:+.2f}, {hi:+.2f}] | Holm p={r['p_holm']:.4f}")
    for r in costs:lines.append(f"COST {r['policy']}: estimated target route {r['mean_target_training_seconds']:.2f} s (mean; measured components)")
    t=result['timing'];lines.extend([f"TIME: training/validation {t['candidate_training_and_validation_seconds']/3600:.3f} h; CPU data+audits {(t['cpu_reference_generation_seconds']+t['cpu_numerical_audit_seconds'])/3600:.3f} h; elapsed {t['wall_seconds']/3600:.3f} h.",f'Training failures retained in scoring: {failures}.',
        'Scientific labels describe evidence, not publication eligibility. Mixed results can support a useful paper.',str(folder/'report.json')])
    if t['compute_cost_usd'] is not None:lines.append(f"Estimated Pod compute cost at your supplied rate: ${t['compute_cost_usd']:.2f}; storage/idle time excluded.")
    if 'historical_development' in result:lines.append(f"HISTORICAL DEVELOPMENT: {result['historical_development']['archive_wall_seconds']/3600:.2f} prior hours, reused; not added to this run's bill.")
    if (output/'hardware_benchmark.json').exists():
        hardware=read(output/'hardware_benchmark.json');result['hardware_benchmark']=hardware
        lines.append('HARDWARE: '+str(hardware['gpu'] or 'CPU only; GPU throughput was not measured.'))
        for row in hardware['rows']:lines.append(f"TRAINING SPEED {row['architecture']} | {row['cpu_threads']} CPU threads | {row['training_ms_per_update']:.2f} ms/update")
        if hardware['device']=='cuda' and len(t['by_architecture'])==3:
            estimated=sum(d['seconds_per_update_including_validation']*240*4000 for d in t['by_architecture'].values())/3600
            result['benchmark_full_training_projection_hours']=[.6*estimated,1.8*estimated]
            lines.append(f'FULL TRAINING PROJECTION: {0.6*estimated:.1f}-{1.8*estimated:.1f} hours, heuristic from short GPU prefixes; excludes unmeasured full-size data/bank/report overhead.')
        atomic_json(folder/'report.json',result)
    text='\n'.join(lines)+'\n';(folder/'report.md').write_text('```text\n'+text+'```\n')
    with (folder/'primary_effects.csv').open('w',newline='') as f:
        fields=['comparison','label','mse_ratio','ratio_ci95_low','ratio_ci95_high','mse_reduction_percent','p_holm']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in primary_resolution+selector:
            w.writerow(dict(comparison=(r['family']+'/'+r['prior']) if 'family' in r else 'selector vs '+r['baseline'],label=r['label'],mse_ratio=r['mse_ratio'],ratio_ci95_low=r['ratio_ci95'][0],ratio_ci95_high=r['ratio_ci95'][1],mse_reduction_percent=r['mse_reduction_percent'],p_holm=r['p_holm']))
    if print_output:
        for line in lines:
            color='31' if ' | RED |' in line or 'COMPLETION: RED' in line else ('33' if 'YELLOW' in line else ('32' if 'GREEN' in line else '36'))
            print(f'\033[{color}m{line}\033[0m' if sys.stdout.isatty() else line,flush=True)
    return result
