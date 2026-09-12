"""Training-only paired perturbations with numerical nonlinear response matching."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import physics
from .core import atomic_json, canonical_hash, save_npz, sha256
from .training import _batch

def smooth_noise(noise):
    spectrum=torch.fft.rfft(noise,dim=-1)
    mask=torch.zeros(spectrum.shape[-1],device=noise.device);mask[1:4]=1
    smooth=torch.fft.irfft(spectrum*mask,n=noise.shape[-1],dim=-1)
    return smooth * (noise.square().mean(-1,keepdim=True)/smooth.square().mean(-1,keepdim=True).clamp_min(1e-30)).sqrt()

def rms(x):
    return x.flatten(1).square().mean(1).sqrt()

def match_response(operator,clean_state,delta_iid,delta_smooth,scale):
    """Bracketed bisection, not a wave-only affine approximation.

    Scale each complete IID context by one scalar. Match the last-state
    operator response to the paired smooth perturbation. A failed match stops
    the experiment rather than quietly omitting difficult examples.
    """
    clean=operator(clean_state)
    target=rms((operator(clean_state+delta_smooth[:,-1])-clean)/scale)
    lo=torch.zeros(len(clean_state),device=clean.device)
    hi=torch.ones_like(lo)
    def response(alpha):
        return rms((operator(clean_state+delta_iid[:,-1]*alpha[:,None,None])-clean)/scale)
    for _ in range(5):
        value=response(hi)
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite response while bracketing target match')
        need=value<target
        if not need.any():break
        hi=torch.where(need,hi*2,hi)
    if (response(hi)<target).any():
        raise FloatingPointError('Could not bracket a response-matched perturbation')
    for _ in range(23):
        mid=(lo+hi)/2; value=response(mid)
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite bisection response')
        below=value<target
        lo=torch.where(below,mid,lo);hi=torch.where(below,hi,mid)
    alpha=(lo+hi)/2
    matched=delta_iid*alpha[:,None,None,None]
    achieved=response(alpha)
    if not torch.allclose(achieved,target,rtol=2e-3,atol=2e-6):
        raise FloatingPointError('Nonlinear response matching missed its prespecified tolerance')
    return matched,dict(max_relative_error=float(((achieved-target).abs()/target.clamp_min(1e-12)).max()),scale_min=float(alpha.min()),scale_max=float(alpha.max()),scale_mean=float(alpha.mean()))

def build_bank(output,job,split,normalization,cfg,protocol_hash,device):
    # Independent of architecture: identical paired banks are shared across all
    # architectures, and all interventions use the same sampled clean contexts.
    key=f"{job['family']}__{job['prior']}__s{job['seed']}"
    folder=Path(output)/'banks'; path=folder/f'{key}.npz';mp=path.with_suffix('.json')
    signature=canonical_hash(dict(protocol=protocol_hash,key=key,normalization=normalization))
    if path.exists() and mp.exists():
        meta=json.loads(mp.read_text())
        if meta['signature']!=signature or sha256(path)!=meta['sha256']:
            raise ValueError('Collocation cache integrity mismatch')
    else:
        start=time.perf_counter()
        family=job['family']; k=cfg['context'];count=cfg['collocation_windows']
        rng=np.random.default_rng(job['seed']+99171)
        generator=torch.Generator(device='cpu').manual_seed(job['seed']+3907)
        transitions=split['actions'].shape[1]-k+1
        ids=rng.integers(0,len(split['states']),count)
        ts=rng.integers(k-1,k-1+transitions,count)
        mean=torch.tensor(normalization['mean'],device=device)[None,:,None]
        scale=torch.tensor(normalization['scale'],device=device)[None,:,None]
        values={key:[] for key in ('iid_history','smooth_history','response_matched_history','iid_target','smooth_target','response_matched_target','actions','params')}
        aggregate={key:0. for key in ('gap','increment','input_iid','input_smooth','response_iid','response_smooth')};numel=0
        match_max=0.;alpha_min=float('inf');alpha_max=0.;alpha_sum=0.
        with torch.no_grad():
            for first in range(0,count,128):
                hist,acts,p,target=_batch(split,ids[first:first+128],ts[first:first+128],k,device)
                noise=torch.randn(hist.shape,generator=generator).to(device)
                iid=cfg['physics_jitter']*scale[:,None]*noise
                smooth=cfg['physics_jitter']*scale[:,None]*smooth_noise(noise)
                def op(state):return physics.step(family,state,acts[:,-1],p,mismatch=job['prior'],fidelity='coarse')
                clean=op(hist[:,-1])
                matched,matching=match_response(op,hist[:,-1],iid,smooth,scale)
                match_max=max(match_max,matching['max_relative_error']);alpha_min=min(alpha_min,matching['scale_min']);alpha_max=max(alpha_max,matching['scale_max']);alpha_sum+=matching['scale_mean']*len(hist)
                for arm,delta in (('iid',iid),('smooth',smooth),('response_matched',matched)):
                    y=op((hist+delta)[:,-1]);z=(y-mean)/scale
                    if not torch.isfinite(z).all():raise FloatingPointError(f'Nonfinite {family} {arm} pseudo-label')
                    values[arm+'_history'].append((hist+delta).cpu().numpy());values[arm+'_target'].append(z.cpu().numpy())
                    if arm!='response_matched':
                        aggregate['input_'+arm]+=float((delta[:,-1]/scale).square().sum())
                        aggregate['response_'+arm]+=float(((y-clean)/scale).square().sum())
                aggregate['gap']+=float(((clean-target)/scale).square().sum())
                aggregate['increment']+=float(((target-hist[:,-1])/scale).square().sum())
                numel+=target.numel()
                values['actions'].append(acts.cpu().numpy());values['params'].append(p.cpu().numpy())
        features=dict(gap_relative=float(np.sqrt(aggregate['gap']/max(aggregate['increment'],1e-30))),increment_rms=float(np.sqrt(aggregate['increment']/numel)),gain_iid=float(np.sqrt(aggregate['response_iid']/max(aggregate['input_iid'],1e-30))),gain_smooth=float(np.sqrt(aggregate['response_smooth']/max(aggregate['input_smooth'],1e-30))))
        states=split['states'].numpy().astype(np.float64)
        states=(states-np.asarray(normalization['mean'])[None,None,:,None])/np.asarray(normalization['scale'])[None,None,:,None]
        power=np.abs(np.fft.rfft(states,axis=-1))**2
        weights=np.full(power.shape[-1],2.);weights[0]=weights[-1]=1
        energy=(power*weights).sum(axis=(0,1,2));energy/=max(energy.sum(),1e-30)
        features.update(low_power=float(energy[1:4].sum()),high_power=float(energy[4:].sum()))
        arrays={key:np.concatenate(value) for key,value in values.items()}
        save_npz(path,**arrays)
        meta=dict(signature=signature,features=features,matching=dict(max_relative_error=match_max,scale_min=alpha_min,scale_max=alpha_max,scale_mean=alpha_sum/count,rtol=.002,atol=.000002),windows=count,setup_seconds=time.perf_counter()-start,sha256=sha256(path))
        atomic_json(mp,meta)
    with np.load(path,allow_pickle=False) as src:
        arrays={key:torch.from_numpy(src[key].copy()) for key in src.files}
    return arrays,meta
