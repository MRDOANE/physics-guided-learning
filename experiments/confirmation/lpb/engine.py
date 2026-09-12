"""Two-stage candidate training; test evaluation is a separate locked phase."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .augmentation import build_bank
from .core import atomic_json, canonical_hash, save_npz, sha256, stack
from .models import build_model
from .training import (RawPredictor, _batch, _latency, _load_checkpoint, _prepare,
                       _restore_rng, _rng_state, _rollout, _save_checkpoint,
                       _seed_all, _sync)

def fit_config(cfg,job):
    return dict(cfg,kind=job['kind'],seed=job['seed'],mismatch=job['prior'],data_count=cfg['split_counts']['train'])

def validation(predictor,system,split,cfg,device):
    vc=dict(cfg,eval_horizon=cfg['validation_horizon'])
    _,trajectory,_,_=_rollout(predictor,system,split,vc,device,'model')
    return float(np.minimum(np.nan_to_num(trajectory,nan=cfg['loss_cap'],posinf=cfg['loss_cap'],neginf=cfg['loss_cap']),cfg['loss_cap']).mean())

def train_to(output,job,data,cfg,protocol_hash,device,target_steps):
    output=Path(output);folder=output/'runs'/job['run_id'];folder.mkdir(parents=True,exist_ok=True)
    cp=folder/'checkpoint.pt';rp=folder/'record.json'
    signature=canonical_hash(dict(protocol_hash=protocol_hash,job=job,data=data['signature']))
    current_stack=stack(device)
    if cp.exists():
        checkpoint=_load_checkpoint(cp,device)
        if checkpoint['signature']!=signature:raise ValueError('Checkpoint source/protocol/data mismatch')
        if checkpoint['stack']!=current_stack:raise ValueError('Resume requires the same software, device, and thread count')
        if checkpoint['step']>=target_steps:
            stored=json.loads(rp.read_text()) if rp.exists() else {}
            # The checkpoint commits before the human-readable record. Repair
            # a crash between those writes; never trust stale completion flags.
            if any(stored.get(key)!=value for key,value in checkpoint['record'].items()):
                atomic_json(rp,checkpoint['record'])
            return checkpoint['record']
    else:checkpoint=None
    _seed_all(job['seed']);sampler=np.random.default_rng(job['seed']+991)
    fc=fit_config(cfg,job)
    splits,norm=_prepare(job['family'],data,fc)
    bank,bank_meta=build_bank(output,job,splits['train'],norm,cfg,protocol_hash,device)
    start=time.perf_counter()
    model=build_model(job['kind'],data['spec'],cfg).to(device)
    predictor=RawPredictor(model,norm).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    _sync(device);setup=time.perf_counter()-start
    if checkpoint:
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        best=checkpoint['best_model'];best_val=checkpoint['best_val'];best_step=checkpoint['best_step']
        step=checkpoint['step'];history=checkpoint['validation_history'];record=checkpoint['record']
        _restore_rng(checkpoint['rng'],sampler)
        _sync(device)
        resumed_setup=time.perf_counter()-start
        record['setup_seconds']+=resumed_setup
        setup_key='probe_setup_seconds' if step<cfg['probe_steps'] else 'continuation_setup_seconds'
        record[setup_key]=record.get(setup_key,0.)+resumed_setup
    else:
        initial_validation_started=time.perf_counter()
        best_val=validation(predictor,job['family'],splits['val'],fc,device)
        _sync(device);setup+=time.perf_counter()-initial_validation_started
        best=copy.deepcopy(model.state_dict());best_step=step=0;history=[dict(step=0,loss=best_val)]
        record=dict(job,signature=signature,features=bank_meta['features'],probe_val=cfg['loss_cap'],probe_seconds=0.,remaining_seconds=0.,train_seconds=0.,setup_seconds=setup,probe_setup_seconds=setup,continuation_setup_seconds=0.,bank_setup_seconds=bank_meta['setup_seconds'],diagnostic_seconds=bank_meta['setup_seconds'],parameter_count=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),numerical_failure=None,completed_steps=0,best_step=0,normalization=norm,stack=current_stack,matching=bank_meta['matching'])
    k=cfg['context'];transitions=splits['train']['actions'].shape[1]-k+1
    prior_train=record['train_seconds'];phase_start=time.perf_counter()
    last_loss=None
    for step in range(step+1,target_steps+1):
        predictor.train()
        ids=sampler.integers(0,len(splits['train']['states']),size=cfg['batch_size'])
        times=sampler.integers(k-1,k-1+transitions,size=cfg['batch_size'])
        bid=torch.as_tensor(sampler.integers(0,cfg['collocation_windows'],size=cfg['batch_size']))
        hist,acts,p,y=_batch(splits['train'],ids,times,k,device)
        optimizer.zero_grad(set_to_none=True)
        pred=predictor.normalized(hist,acts,p);truth=(y-predictor.state_mean[:,0])/predictor.state_scale[:,0]
        loss=(pred-truth).square().mean()
        if job['arm']!='none':
            aug=bank[job['arm']+'_history'][bid].to(device)
            target=bank[job['arm']+'_target'][bid].to(device)
            pred_aug=predictor.normalized(aug,bank['actions'][bid].to(device),bank['params'][bid].to(device))
            loss=loss+cfg['physics_weight']*(pred_aug-target).square().mean()
        failed=not bool(torch.isfinite(loss))
        if not failed:
            loss.backward()
            norm_grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            failed=not bool(torch.isfinite(norm_grad))
        if failed:
            # Preserve the candidate in every comparison at the declared loss
            # cap; numerical failures cannot improve a policy by being omitted.
            record['numerical_failure']=dict(step=step,reason='nonfinite training loss or gradient')
            best_val=cfg['loss_cap'];step=cfg['steps']
        else:
            optimizer.step();last_loss=float(loss.detach())
        commit=(step%cfg['checkpoint_every']==0 or step in (cfg['probe_steps'],target_steps,cfg['steps']))
        if step%cfg['validation_every']==0 or commit:
            val=cfg['loss_cap'] if failed else validation(predictor,job['family'],splits['val'],fc,device)
            history.append(dict(step=step,loss=val))
            if val<best_val:
                best_val=val;best=copy.deepcopy(model.state_dict());best_step=step
        if commit:
            _sync(device)
            elapsed=prior_train+time.perf_counter()-phase_start
            record.update(train_seconds=elapsed,completed_steps=step,best_step=best_step,last_training_loss=last_loss)
            if step<=cfg['probe_steps']:
                record.update(probe_val=best_val,probe_seconds=elapsed)
            else:
                record['remaining_seconds']=elapsed-record['probe_seconds']
            if failed and record['numerical_failure']['step']<=cfg['probe_steps']:
                record.update(probe_val=cfg['loss_cap'],probe_seconds=elapsed,remaining_seconds=0.)
            if step>=cfg['steps']:record['final_val']=best_val
            payload=dict(signature=signature,stack=current_stack,step=step,model=model.state_dict(),best_model=best,optimizer=optimizer.state_dict(),best_val=best_val,best_step=best_step,validation_history=history,rng=_rng_state(sampler),record=copy.deepcopy(record))
            if step==cfg['probe_steps'] or (failed and not (folder/'probe.pt').exists()):
                _save_checkpoint(folder/'probe.pt',payload)
            _save_checkpoint(cp,payload);atomic_json(rp,record)
        if failed:break
    return record

def evaluate(output,job,data,cfg,protocol_hash,device):
    output=Path(output)
    plan=json.loads((output/'evaluation_plan.json').read_text())
    if plan['selection_sha256']!=sha256(output/'selection_manifest.json'):
        raise ValueError('Selection changed after test lock')
    folder=output/'runs'/job['run_id'];rp=folder/'record.json';path=folder/'evaluation.npz'
    record=json.loads(rp.read_text())
    expected_signature=canonical_hash(dict(protocol_hash=protocol_hash,job=job,data=data['signature']))
    if record['signature']!=expected_signature:
        raise ValueError('Evaluation source/protocol/job/development-data mismatch')
    if path.exists() and record.get('evaluation_sha256'):
        if sha256(path)!=record['evaluation_sha256']:raise ValueError('Evaluation array checksum mismatch')
        return record
    if record['completed_steps']!=cfg['steps']:raise RuntimeError('Cannot test an incomplete candidate')
    checkpoint=_load_checkpoint(folder/'checkpoint.pt',device)
    if checkpoint['signature']!=expected_signature:raise ValueError('Evaluation checkpoint signature mismatch')
    fc=fit_config(cfg,job);splits,norm=_prepare(job['family'],data,fc)
    model=build_model(job['kind'],data['spec'],cfg).to(device)
    predictor=RawPredictor(model,norm).to(device)
    arrays={};_sync(device);start=time.perf_counter()
    # Cache physical/persistence references across arms and training seeds. They
    # use identical data, approximate coefficients and train normalization.
    refpath=output/'references'/f"{job['family']}__{job['prior']}.npz"
    refmeta=refpath.with_suffix('.json')
    reference_signature=canonical_hash(dict(protocol=protocol_hash,family=job['family'],prior=job['prior'],normalization=norm,test_ids=data['test']['ids'].tolist(),ood_ids=data['ood']['ids'].tolist()))
    if refpath.exists() and refmeta.exists():
        metadata=json.loads(refmeta.read_text())
        if metadata['signature']!=reference_signature or metadata['sha256']!=sha256(refpath):
            raise ValueError('Physical baseline cache integrity mismatch')
        with np.load(refpath,allow_pickle=False) as src:refs={key:src[key].copy() for key in src.files}
    else:
        refs={}
        for split in ('test','ood'):
            for method in ('mechanistic','persistence'):
                _,_,curve,_=_rollout(predictor,job['family'],splits[split],fc,device,method)
                refs[f'{split}_{method}_mse']=curve
        save_npz(refpath,**refs)
        atomic_json(refmeta,dict(signature=reference_signature,sha256=sha256(refpath)))
    arrays.update(refs)
    for split in ('test','ood'):
        arrays[split+'_ids']=np.asarray(data[split]['ids'])
        for method,key in (('best_model',split+'_mse'),('model',split+'_exact_mse')):
            model.load_state_dict(checkpoint[method])
            if record['numerical_failure']:
                curve=np.full((len(data[split]['states']),cfg['eval_horizon']),np.inf)
            else:
                _,_,curve,_=_rollout(predictor,job['family'],splits[split],fc,device,'model')
            arrays[key]=curve
    model.load_state_dict(checkpoint['best_model'])
    record['inference_ms_per_step']=_latency(predictor,splits['val'],fc,device)
    _sync(device);record['evaluation_seconds']=time.perf_counter()-start
    save_npz(path,**arrays)
    record.update(evaluation_path=str(path.relative_to(output)),evaluation_sha256=sha256(path),checkpoint_sha256=sha256(folder/'checkpoint.pt'))
    atomic_json(rp,record)
    return record
