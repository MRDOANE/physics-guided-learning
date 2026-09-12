"""Whole-trajectory splits; held-out test generation requires a frozen policy."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from . import physics
from .core import atomic_json, canonical_hash, save_npz, sha256

def get_data(output, family, cfg, source, phase='development'):
    if phase not in ('development','test'):
        raise ValueError(phase)
    output = Path(output)
    if phase=='test':
        locked = output/'evaluation_plan.json'
        if not locked.exists():
            raise RuntimeError('Test data are locked until all selection decisions are committed')
        manifest = json.loads(locked.read_text())
        if manifest['selection_sha256'] != sha256(output/'selection_manifest.json'):
            raise ValueError('Frozen selection manifest was modified')
    names = ('train','val') if phase=='development' else ('test','ood')
    descriptor = dict(family=family,phase=phase,source_hash=source,grid=cfg['grid'],steps=cfg['trajectory_steps'],burn_in=cfg['burn_in'],counts={n:cfg['split_counts'][n] for n in names},seed=cfg['train_val_seed'] if phase=='development' else cfg['test_seed'])
    signature = canonical_hash(descriptor)
    path = output/'data'/f'{family}_{phase}.npz'
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['signature']!=signature or meta['sha256']!=sha256(path):
            raise ValueError('Dataset cache source/config/checksum mismatch')
    else:
        start = time.perf_counter(); arrays = {}
        family_seed = int(canonical_hash(family)[:8],16)
        for split in names:
            split_index = ('train','val','test','ood').index(split)
            rng = np.random.default_rng(np.random.SeedSequence([descriptor['seed'],family_seed,split_index]))
            n = cfg['split_counts'][split]; burn = cfg['burn_in']
            params = physics.sample_params(family,n,rng,ood=split=='ood')
            initial = physics.initial_states(family,n,cfg['grid'],rng)
            actions = physics.make_actions(family,n,cfg['trajectory_steps']+burn,rng)
            states = np.concatenate([physics.simulate(family,params[i:i+32],initial[i:i+32],actions[i:i+32],fidelity='fine') for i in range(0,n,32)])[:,burn:]
            if not np.isfinite(states).all():
                raise FloatingPointError(f'Nonfinite numerical reference {family}/{split}; no trajectories discarded')
            arrays.update({f'{split}_states':states.astype(np.float32),f'{split}_actions':actions[:,burn:].astype(np.float32),f'{split}_params':params.astype(np.float32),f'{split}_ids':np.array([f'{family}:{descriptor["seed"]}:{split}:{i}' for i in range(n)])})
        save_npz(path,**arrays)
        meta = dict(signature=signature,description=descriptor,sha256=sha256(path),generation_seconds=time.perf_counter()-start)
        atomic_json(meta_path,meta)
    with np.load(path,allow_pickle=False) as src:
        data = {split:{key:src[f'{split}_{key}'].copy() for key in ('states','actions','params','ids')} for split in names}
    data.update(spec=physics.spec(family,cfg['grid']),signature=signature,metadata=meta)
    return data

def audit_family(family,cfg):
    """Time-refinement check. This does not establish spatial convergence."""
    rows=[]
    for ood in (False,True):
        rng=np.random.default_rng(np.random.SeedSequence([cfg['train_val_seed'],int(canonical_hash(family)[:8],16),881,int(ood)]))
        p=physics.sample_params(family,4,rng,ood=ood)
        x=physics.initial_states(family,4,cfg['grid'],rng)
        a=physics.make_actions(family,4,min(cfg['trajectory_steps'],32),rng)
        fine=physics.simulate(family,p,x,a,fidelity='fine')
        check=physics.simulate(family,p,x,a,fidelity='check')
        channel_scale=np.maximum(check.std(axis=(0,1,3),keepdims=True),1e-6)
        error=float(np.sqrt(np.mean(((fine-check)/channel_scale)**2)))
        if not np.isfinite(error) or error>1e-3:
            raise FloatingPointError(f'{family} time-refinement audit failed: {error}')
        rows.append(dict(family=family,coefficient_regime='ood' if ood else 'id',normalized_rmse=error,threshold=.001,trajectories=4,steps=a.shape[1],passed=True))
    return rows
