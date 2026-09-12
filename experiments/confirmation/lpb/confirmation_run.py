"""Resumable staged execution with explicit freeze and evaluation barriers."""
import itertools
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
from datetime import datetime,timezone
import fcntl
import numpy as np
import torch
from .core import ROOT,atomic_json,canonical_hash,sha256,stack
from .datasets import get_data,audit_family
from .engine import train_to,evaluate
from .guarded import ALLOWED,decide,fit_selector
from . import physics

def now():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(Path(path).read_text())

def source_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).parts[0] in ('lpb','configs','development','tests') and p.suffix in ('.json','.py'))+[ROOT/'run_experiment.py',ROOT/'launch.sh',ROOT/'requirements.txt']

def digest_source():return canonical_hash({str(p.relative_to(ROOT)):sha256(p) for p in source_files()})

def inventory(cfg,stage,grid):
    families=cfg['resolution_families'] if stage=='resolution' else cfg['confirmation_families']
    seeds=cfg['resolution_seeds'] if stage=='resolution' else cfg['confirmation_seeds']
    priors=['correct','coefficient'] if stage=='resolution' else ['correct','coefficient','structural']
    arms=['none','smooth'] if stage=='resolution' else ['none','iid','smooth','response_matched']
    for f,k,p,s in itertools.product(families,cfg['kinds'],priors,seeds):
        tid=f'{f}__{k}__{p}__s{s}'
        for a in arms:yield dict(task_id=tid,family=f,kind=k,prior=p,seed=s,arm=a,run_id=f'{tid}__{a}')

def export(output):
    path=output/'full_results.zip';tmp=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for f in sorted(output.rglob('*')):
            if not f.is_file() or f.suffix not in ('.json','.md','.csv','.npz','.py','.sh','.txt'):continue
            rel=f.relative_to(output)
            # Keep per-trajectory error arrays, not multi-GB training data/banks/checkpoints.
            if f.suffix=='.npz' and 'runs' not in rel.parts and 'references' not in rel.parts:continue
            if '__pycache__' in rel.parts:continue
            z.write(f,str(rel))
    os.replace(tmp,path);return path

def spatial_audit(family,cfg):
    """A diagnostic projection comparison, not a continuum proof or success gate."""
    n=cfg['grid'];rows=[]
    for ood in (False,True):
        rng=np.random.default_rng(np.random.SeedSequence([72491,int(canonical_hash(family)[:8],16),int(ood)]))
        params=physics.sample_params(family,4,rng,ood)
        state=rng.bit_generator.state
        low=physics.initial_states(family,4,n,rng)
        rng.bit_generator.state=state
        high=physics.initial_states(family,4,2*n,rng)
        actions=physics.make_actions(family,4,cfg['trajectory_steps']+cfg['burn_in'],rng)
        lo=physics.simulate(family,params,low,actions,'fine')[:,cfg['burn_in']:]
        hi=physics.simulate(family,params,high,actions,'fine')[:,cfg['burn_in']:]
        coeff=np.fft.rfft(hi,axis=-1)[...,:n//2+1]*(n/(2*n))
        coeff[...,-1]=2*coeff[...,-1].real
        projected=np.fft.irfft(coeff,n=n,axis=-1)
        scale=np.maximum(projected.std(axis=(0,1,3),keepdims=True),1e-6)
        rows.append(dict(family=family,grid=n,reference_grid=2*n,ood=ood,normalized_rmse=float(np.sqrt(np.mean(((lo-projected)/scale)**2))),steps=cfg['trajectory_steps'],role='descriptive spatial sensitivity; no arbitrary pass threshold'))
    return rows

class BudgetPause(Exception):pass

def run(args):
    profile='smoke' if args.command in ('smoke','benchmark') else 'full'
    cfg=load(ROOT/'configs'/f'{profile}.json')
    if args.command=='benchmark':
        # Same full model/batch/grid, short prefix only. Scientifically ineligible.
        cfg=load(ROOT/'configs/full.json')|dict(profile='benchmark',steps=40,probe_steps=20,checkpoint_every=20,validation_every=20,
              resolution_families=['wave'],resolution_grids=[64],resolution_seeds=[9701],confirmation_families=['cahn_hilliard'],confirmation_seeds=[9801],
              trajectory_steps=40,burn_in=4,split_counts=dict(train=16,val=6,test=6,ood=6),collocation_windows=256,primary_horizon=16,eval_horizon=24,bootstrap_replicates=400)
        profile='benchmark'
    output=Path(args.output).resolve()/profile;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'run.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another process is using this experiment directory')
    torch.set_num_threads(args.threads)
    if args.device=='cuda':
        try:
            x=torch.ones((128,128),device='cuda');(x@x).sum().item();torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(f'GPU preflight failed: {exc}. NVIDIA_VISIBLE_DEVICES={os.environ.get("NVIDIA_VISIBLE_DEVICES")!r}. A visible nvidia-smi device alone is insufficient; restart/recreate a GPU-enabled RunPod template. The launcher does not replace PyTorch or alter GPU visibility.') from exc
    if profile=='full' and args.device=='cpu':raise ValueError('Use a GPU for full training; CPU is supported for smoke and benchmark.')
    source=digest_source();payload=dict(config=cfg,source_hash=source);signature=canonical_hash(payload)
    path=output/'protocol.json'
    if path.exists():
        if load(path)['signature']!=signature:raise ValueError('Source/configuration changed; use a separate output directory')
    else:
        atomic_json(path,dict(payload,signature=signature,created_at=now(),disclosure='Prospective follow-up to already observed results; local freeze, not an externally registered preregistration.'))
        for f in source_files():
            dest=output/'source'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    environment=stack(args.device)
    if (output/'environment.json').exists() and load(output/'environment.json')!=environment:raise ValueError('Resume requires identical Python/NumPy/PyTorch/GPU/thread stack')
    atomic_json(output/'environment.json',environment)
    started=time.perf_counter();old=load(output/'execution.json') if (output/'execution.json').exists() else {}
    prior_wall=old.get('wall_seconds',0.)
    stages=['resolution','selector'] if args.command in ('full','smoke','benchmark') else [args.command]
    requested=sorted(set(old.get('requested_stages',[]))|set(stages))
    def checkpoint(phase,job=None):
        elapsed=time.perf_counter()-started
        atomic_json(output/'execution.json',dict(state='RUNNING',phase=phase,last_job=job,requested_stages=requested,updated_at=now(),wall_seconds=prior_wall+elapsed,hourly_rate=args.hourly_rate))
        if args.max_hours and elapsed>args.max_hours*3600:raise BudgetPause('Requested wall-time allowance reached between jobs. Pod billing continues until you stop it.')
    try:
        if args.command=='benchmark' and not (output/'hardware_benchmark.json').exists():
            from .hardware import probe
            atomic_json(output/'hardware_benchmark.json',probe(cfg,args.device,args.threads))
            print('[HARDWARE BENCHMARK] Full-width updates measured at 1, 4, and configured CPU thread counts (duplicates removed).',flush=True)
        bank=ROOT/'development/validation_bank.json';prov=load(ROOT/'development/provenance.json')
        if sha256(bank)!=prov['included_bank_sha256']:raise ValueError('Historical development data changed')
        modelpath=output/'selector_model.json'
        if not modelpath.exists():
            model=fit_selector(load(bank));model.update(frozen_at=now(),development_sha256=sha256(bank),protocol_signature=signature)
            atomic_json(modelpath,model)
        model=load(modelpath)
        if model['development_sha256']!=sha256(bank) or model['protocol_signature']!=signature:raise ValueError('Frozen selector provenance mismatch')
        print(f"[MODEL FROZEN] {model['selected']['mode']}; historical development only. Fresh target outcomes cannot tune it.",flush=True)
        checkpoint('selector frozen')
        for stage in stages:
            grids=cfg['resolution_grids'] if stage=='resolution' else [cfg['confirmation_grid']]
            for grid in grids:
                child=output/stage/f'g{grid}';child.mkdir(parents=True,exist_ok=True)
                cc=cfg|dict(grid=grid);jobs=list(inventory(cc,stage,grid));families=sorted({j['family'] for j in jobs})
                if (child/'stage_complete.json').exists():
                    print(f'[RESUME] {stage}/g{grid} completed; integrity will be checked in the report.',flush=True);continue
                childsig=canonical_hash(dict(parent=signature,stage=stage,grid=grid))
                atomic_json(child/'stage_protocol.json',dict(config=cc,signature=childsig,jobs=jobs,selector_model_sha256=sha256(modelpath)))
                if not (child/'numerical_audit.json').exists():
                    begin=time.perf_counter();temporal=[];spatial=[]
                    for family in families:
                        temporal.extend(audit_family(family,cc));spatial.extend(spatial_audit(family,cc))
                        print(f'[NUMERICAL CHECK] {family} g{grid}: finite time-refinement reference; spatial sensitivity recorded.',flush=True)
                    atomic_json(child/'numerical_audit.json',dict(temporal=temporal,spatial=spatial,seconds=time.perf_counter()-begin))
                data={}
                for family in families:
                    data[family]=get_data(child,family,cc,source,'development');checkpoint('data preparation',family)
                records={}
                for i,job in enumerate(jobs,1):
                    records[job['run_id']]=train_to(child,job,data[job['family']],cc,childsig,args.device,cc['probe_steps'])
                    print(f'[{stage}/g{grid} PROBE] {i}/{len(jobs)} {job["run_id"]}',flush=True);checkpoint('probes',job['run_id'])
                frozen=child/'selection_manifest.json'
                if not frozen.exists():
                    if any(r['completed_steps']>cc['probe_steps'] and not r['numerical_failure'] for r in records.values()):raise ValueError('Missing freeze after target continuation; a fresh output directory is required')
                    inputs=[{k:v for k,v in r.items() if k in ALLOWED and k not in ('final_val','remaining_seconds')} for r in records.values()]
                    atomic_json(child/'selection_inputs.json',inputs)
                    decisions=decide(model,inputs) if stage=='selector' else []
                    atomic_json(frozen,dict(protocol_signature=childsig,selector_model_sha256=sha256(modelpath),inputs_sha256=sha256(child/'selection_inputs.json'),frozen_at=now(),decisions=decisions))
                manifest=load(frozen)
                if manifest['selector_model_sha256']!=sha256(modelpath) or manifest['inputs_sha256']!=sha256(child/'selection_inputs.json'):raise ValueError('Frozen selection changed')
                for i,job in enumerate(jobs,1):
                    records[job['run_id']]=train_to(child,job,data[job['family']],cc,childsig,args.device,cc['steps'])
                    print(f'[{stage}/g{grid} FIT] {i}/{len(jobs)} {job["run_id"]}',flush=True);checkpoint('candidate completion',job['run_id'])
                ep=child/'evaluation_plan.json'
                if not ep.exists():atomic_json(ep,dict(selection_sha256=sha256(frozen),locked_at=now()))
                if load(ep)['selection_sha256']!=sha256(frozen):raise ValueError('Evaluation lock mismatch')
                for family in families:
                    test=get_data(child,family,cc,source,'test');data[family].update({k:test[k] for k in ('test','ood')});checkpoint('test generation',family)
                for i,job in enumerate(jobs,1):
                    records[job['run_id']]=evaluate(child,job,data[job['family']],cc,childsig,args.device)
                    print(f'[{stage}/g{grid} TEST] {i}/{len(jobs)} {job["run_id"]}',flush=True);checkpoint('evaluation',job['run_id'])
                atomic_json(child/'bank.json',list(records.values()))
                atomic_json(child/'stage_complete.json',dict(finished_at=now(),bank_sha256=sha256(child/'bank.json'),expected_records=len(jobs)))
                from .confirmation_report import report
                report(output)
        checkpoint('complete')
        status=load(output/'execution.json');status['state']='COMPLETE_REQUESTED_STAGES';atomic_json(output/'execution.json',status)
        from .confirmation_report import report
        result=report(output);archive=export(output);print(f'Results ZIP: {archive}',flush=True)
        if args.command=='benchmark':print('[BENCHMARK] Timings are in reports/report.json. No scientific conclusions; full settings still require full training.',flush=True)
        return result['exit_code']
    except (BudgetPause,KeyboardInterrupt) as exc:
        atomic_json(output/'execution.json',dict(state='PAUSED',requested_stages=requested,reason=str(exc),wall_seconds=prior_wall+time.perf_counter()-started,updated_at=now(),hourly_rate=args.hourly_rate))
        print(f'[PAUSED] {exc}\nRerun the same command to resume. Stopping this process does not stop RunPod billing.',flush=True)
        from .confirmation_report import report
        report(output);print('Partial ZIP:',export(output));return 0 if isinstance(exc,BudgetPause) else 130
    except Exception as exc:
        atomic_json(output/'execution.json',dict(state='ERROR',requested_stages=requested,reason=str(exc),wall_seconds=prior_wall+time.perf_counter()-started,updated_at=now(),hourly_rate=args.hourly_rate))
        print(f'[RED] EXECUTION ERROR: {exc}\nScientific hypotheses are NOT_EVALUATED for incomplete stages.',flush=True)
        export(output);raise
    finally:lock.close()
