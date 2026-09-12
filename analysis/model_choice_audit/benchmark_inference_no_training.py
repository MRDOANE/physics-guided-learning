#!/usr/bin/env python3
"""Optional matched inference benchmark. No optimizer, fitting, or test scoring.

Uses the archived source and configuration. Trained checkpoints are absent from
the public release: neural weights are initialized solely to time the fixed
forward graph. These measurements must be labeled architecture execution times.
Historical accuracy continues to come from the archived trained-model errors.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

SETTINGS = {'wave':'resolution_g64','advection_diffusion':'resolution_g64',
            'allen_cahn':'resolution_g64','burgers':'development',
            'ks':'development','gray_scott':'development',
            'cahn_hilliard':'selector_g64','fitzhugh_nagumo':'selector_g64'}
STAGES = {'development':'results/development',
          'resolution_g64':'results/confirmation/resolution/g64',
          'selector_g64':'results/confirmation/selector/g64'}

def read(p):return json.loads(Path(p).read_text())

def load_package(repo, stage):
    which='development' if stage=='development' else 'confirmation'
    prefix='timing_'+which+'_lpb'
    path=repo/'experiments'/which/'lpb'
    if prefix not in sys.modules:
        spec=importlib.util.spec_from_file_location(prefix,path/'__init__.py',submodule_search_locations=[str(path)])
        module=importlib.util.module_from_spec(spec);sys.modules[prefix]=module;spec.loader.exec_module(module)
    return tuple(importlib.import_module(prefix+'.'+name) for name in ['physics','models','training'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('fresh_inference_timings.json'))
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--batch-sizes',type=int,nargs='+',default=[1])
    parser.add_argument('--blocks',type=int,default=7)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--families',nargs='+',choices=list(SETTINGS),default=list(SETTINGS))
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Choose a new --output path; the existing timing file is preserved.')
    if min(args.threads,args.blocks,args.repeats,*args.batch_sizes)<1 or max(args.batch_sizes)>64:
        raise SystemExit('Use positive settings and batch sizes no larger than 64.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit('Use the existing PyTorch/NumPy environment. This script does not install or replace packages.') from exc
    torch.set_num_threads(args.threads)
    if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA is unavailable. Use a working GPU environment or explicitly select --device cpu.')
    device=torch.device(args.device)
    torch.manual_seed(20260912);torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():torch.backends.cudnn.benchmark=False
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    rows=[];started=time.perf_counter();order_rng=np.random.default_rng(23091)
    for family in args.families:
        stage=SETTINGS[family];folder=args.repository.resolve()/STAGES[stage]
        protocol=read(folder/('protocol.json' if stage=='development' else 'stage_protocol.json'));cfg=protocol['config']
        physics,models,training=load_package(args.repository.resolve(),stage)
        candidates=[read(p) for p in sorted((folder/'runs').glob(f'{family}__transformer__*__none/record.json'))]
        if not candidates:raise SystemExit('No archived candidates found for '+family)
        # Recreate just the initial context at the test distribution. Sampling
        # arrays have the original lengths so RNG order matches the source.
        seed=cfg['test_seed'];family_seed=int(hashlib.sha256(json.dumps(family,separators=(',',':')).encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(np.random.SeedSequence([seed,family_seed,2]))
        params=physics.sample_params(family,64,rng,ood=False)
        initial=physics.initial_states(family,64,cfg['grid'],rng)
        actions=physics.make_actions(family,64,cfg['trajectory_steps']+cfg['burn_in'],rng)
        count=cfg['burn_in']+cfg['context']-1
        states=np.concatenate([physics.simulate(family,params[i:i+32],initial[i:i+32],actions[i:i+32,:count],'fine') for i in range(0,64,32)])
        hist=torch.tensor(states[:,cfg['burn_in']:cfg['burn_in']+cfg['context']],dtype=torch.float32,device=device)
        acts=torch.tensor(actions[:,cfg['burn_in']:cfg['burn_in']+cfg['context']],dtype=torch.float32,device=device)
        true_params=torch.tensor(params,dtype=torch.float32,device=device)
        for prior in sorted({r['prior'] for r in candidates}):
            rec=next(r for r in candidates if r['prior']==prior)
            approximate=physics.approx_params(family,true_params,prior)
            predictors={kind:training.RawPredictor(models.build_model(kind,physics.spec(family,cfg['grid']),cfg),rec['normalization']).to(device).eval() for kind in ['transformer','looped','fno']}
            for batch in args.batch_sizes:
                h,a,p=hist[:batch],acts[:batch],approximate[:batch]
                def physics_forward():return physics.step(family,h[:,-1],a[:,-1],p,mismatch=prior,fidelity='coarse')
                calls={'mechanistic':physics_forward}
                for kind,predictor in predictors.items():
                    calls[kind]=lambda predictor=predictor:predictor(h,a,p)
                timings={name:[] for name in calls}
                with torch.no_grad():
                    for call in calls.values():
                        for _ in range(5):call()
                    sync()
                    for _ in range(args.blocks):
                        for name in order_rng.permutation(list(calls)):
                            sync();t=time.perf_counter()
                            for __ in range(args.repeats):calls[name]()
                            sync();timings[name].append(1000*(time.perf_counter()-t)/args.repeats)
                for name,values in timings.items():
                    rows.append(dict(stage=stage,family=family,grid=cfg['grid'],prior=prior,kind=name,batch_size=batch,
                                     median_ms_per_batch_step=float(np.median(values)),block_ms_per_batch_step=values,
                                     parameter_count=None if name=='mechanistic' else sum(p.numel()*(2 if p.is_complex() else 1) for p in predictors[name].parameters()),
                                     weights='none' if name=='mechanistic' else 'initialized; no training'))
                print(f'TIMED {family}/{prior}/batch={batch}: '+', '.join(f'{k} {np.median(v):.3f} ms' for k,v in timings.items()),flush=True)
    result=dict(status='COMPLETED_EXECUTION_BENCHMARK',model_training=False,accuracy_evaluation=False,
                scope='Fresh fixed-graph execution timings. Neural weights are initialized; historical accuracy is a separate measurement. Do not mix fresh physics latency with old neural latency.',
                timing_scope='Resident float32 inputs; batch-step forward includes full neural normalization or the implemented coarse physical step. Fixed observed context reused; no recursive rollout, host transfer, or error calculation timed.',
                limitations='Physics coefficients and transforms are recomputed by the existing step implementation. No solver optimization or accuracy-versus-solver-tolerance sweep. Initial neural output heads are zero; forward graphs still execute in full.',
                environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,threads=torch.get_num_threads(),cuda_runtime=torch.version.cuda),
                blocks=args.blocks,repeats_per_block=args.repeats,elapsed_seconds=time.perf_counter()-started,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETION: GREEN | timing records saved')
    print('MODEL TRAINING: NONE | ACCURACY CLAIMS: NOT EVALUATED')
    print('NEURAL WEIGHTS: INITIALIZED | report as architecture execution benchmark')
    print(args.output.resolve())

if __name__=='__main__':main()
