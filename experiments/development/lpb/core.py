"""Immutable protocol, atomic artifacts, and a bounded inventory."""
from __future__ import annotations
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temp, path)

def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)

def source_hash():
    paths = sorted([*ROOT.glob('lpb/*.py'), *ROOT.glob('configs/*.json'), ROOT/'launch.sh', ROOT/'requirements.txt'])
    return canonical_hash({str(p.relative_to(ROOT)): sha256(p) for p in paths})

def config(profile):
    cfg = json.loads((ROOT/'configs'/f'{profile}.json').read_text())
    if cfg['probe_steps'] >= cfg['steps'] or cfg['probe_steps'] < 1:
        raise ValueError('Probe budget must be positive and smaller than full training')
    if cfg['eval_horizon'] + cfg['context'] > cfg['trajectory_steps']+1:
        raise ValueError('Rollout exceeds data length')
    if cfg['train_val_seed'] == cfg['test_seed']:
        raise ValueError('Test and development data seeds must differ')
    if cfg['selector']['development_families'] != [f for f in cfg['families'] if f != cfg['confirmation_family']]:
        raise ValueError('Final confirmation family must be excluded from development')
    if cfg['profile']=='full' and len(cfg['seeds']) < 6:
        raise ValueError('Full protocol requires six training seeds')
    return cfg

def tasks(cfg):
    for family, kind, prior, seed in itertools.product(cfg['families'],cfg['kinds'],cfg['priors'],cfg['seeds']):
        yield dict(task_id=f'{family}__{kind}__{prior}__s{seed}',family=family,kind=kind,prior=prior,seed=seed)

def jobs(cfg):
    for task in tasks(cfg):
        for arm in cfg['arms']:
            yield dict(task, arm=arm, run_id=f"{task['task_id']}__{arm}")

def stack(device):
    out = dict(torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),device=str(device),threads=torch.get_num_threads(),cuda_runtime=torch.version.cuda)
    if torch.device(device).type=='cuda':
        out.update(gpu=torch.cuda.get_device_name(device),visible_gpus=torch.cuda.device_count())
    return out
