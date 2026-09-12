"""RunPod entry point with a physical test-data barrier and resumable phases."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from .core import ROOT, atomic_json, canonical_hash, config, jobs, sha256, source_hash, stack
from .datasets import audit_family, get_data
from .engine import evaluate, train_to
from .selection import freeze_choices
from .reporting import analyze, print_console

def now():return datetime.now(timezone.utc).isoformat()

def export_results(output):
    output=Path(output);path=output/'full_results.zip';temporary=output/'full_results.zip.tmp'
    included=[]
    for folder in ('runs','reports','source','references'):
        if (output/folder).exists():
            included.extend(p for p in (output/folder).rglob('*') if p.is_file() and (p.suffix in ('.json','.npz','.md','.csv','.py','.sh','.txt') or p.name=='LICENSE') and '__pycache__' not in p.parts)
    included.extend(p for p in output.glob('*.json') if p.is_file())
    included.extend((output/'banks').glob('*.json'))
    included.extend((output/'data').glob('*.json'))
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for p in sorted(set(included)):archive.write(p,str(p.relative_to(output)))
    os.replace(temporary,path)
    return path

def snapshot(output):
    dest=Path(output)/'source'
    for rel in ('lpb','configs','docs','tests'):
        for src in (ROOT/rel).glob('*'):
            if src.is_file() and src.suffix in ('.py','.json','.md'):
                target=dest/rel/src.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    for name in ('README.md','launch.sh','requirements.txt','SOURCE_CHECKSUMS.json','VALIDATION.md','LICENSE'):
        if (ROOT/name).exists():
            dest.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest/name)

def load_manifest(output):
    output=Path(output)
    manifest=json.loads((output/'selection_manifest.json').read_text())
    plan=json.loads((output/'evaluation_plan.json').read_text())
    if plan['selection_sha256']!=sha256(output/'selection_manifest.json'):
        raise ValueError('Frozen selector integrity mismatch')
    for decision in manifest['decisions']:
        decision['choices']['full_validation']=plan['full_validation_choices'][decision['task_id']]
    if (output/'execution_status.json').exists():
        status=json.loads((output/'execution_status.json').read_text())
        manifest['bank_wall_seconds']=status.get('committed_wall_seconds')
    return manifest

def report_run(output,color='auto'):
    output=Path(output);protocol=json.loads((output/'protocol.json').read_text());cfg=protocol['config']
    records=[json.loads((output/'runs'/job['run_id']/'record.json').read_text()) for job in jobs(cfg) if (output/'runs'/job['run_id']/'record.json').exists()]
    manifest=load_manifest(output) if (output/'evaluation_plan.json').exists() else {'decisions':[]}
    report=analyze(output,cfg,records,manifest,eligible=cfg['profile']=='full')
    print_console(report,color=color)
    return report

def run(args):
    cfg=config(args.profile);output=Path(args.output).expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    # A second lock also covers direct Python invocations, not only Bash.
    import fcntl
    lock=(output/'.python.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another process is using this output directory')
    torch.set_num_threads(args.threads)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    if cfg['profile']=='full' and device=='cpu' and not args.allow_cpu_full:
        raise RuntimeError('Full CPU runs require --allow-cpu-full. Use the smoke profile for a CPU check.')
    source=source_hash();payload=dict(schema=1,config=cfg,source_hash=source)
    protocol_hash=canonical_hash(payload);protocol_path=output/'protocol.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text())
        if protocol['protocol_hash']!=protocol_hash:raise ValueError('Output belongs to different code or configuration; use a new directory')
    else:
        protocol=dict(payload,protocol_hash=protocol_hash,created_at=now(),expected_fits=len(list(jobs(cfg))))
        atomic_json(protocol_path,protocol);snapshot(output)
    atomic_json(output/'environment_latest.json',stack(device))
    if not (output/'environment_training.json').exists():atomic_json(output/'environment_training.json',stack(device))
    started=time.perf_counter();inventory=list(jobs(cfg));records={};datasets={}
    status_path=output/'execution_status.json'
    old=json.loads(status_path.read_text()) if status_path.exists() else {}
    previous_wall=float(old.get('committed_wall_seconds',0.))
    def progress(phase,index,total,job=None):
        status=dict(status='RUNNING',phase=phase,phase_completed=index,phase_total=total,expected_fits=len(inventory),last_run=job['run_id'] if job else None,updated_at=now(),committed_wall_seconds=previous_wall+time.perf_counter()-started)
        atomic_json(status_path,status)
        suffix=f" | {job['run_id']}" if job else ''
        print(f'[{phase}] {index}/{total}{suffix}',flush=True)
    try:
        print(f"{cfg['profile'].upper()}: {len(inventory)} fits, {len(cfg['families'])} equation families, {device}; output {output}",flush=True)
        audit_path=output/'numerical_audit.json'
        if not audit_path.exists():
            rows=[]
            for family in cfg['families']:
                rows.extend(audit_family(family,cfg));print(f'[numerical audit] {family}: PASS',flush=True)
            atomic_json(audit_path,dict(passed=True,scope='temporal refinement at fixed spatial grid, not spatial convergence',rows=rows))
        for family in cfg['families']:
            datasets[family]=get_data(output,family,cfg,source,'development')
            print(f'[training/validation data] {family}: ready',flush=True)
        for index,job in enumerate(inventory,1):
            records[job['run_id']]=train_to(output,job,datasets[job['family']],cfg,protocol_hash,device,cfg['probe_steps'])
            progress('short probes',index,len(inventory),job)
        development=[job for job in inventory if job['family']!=cfg['confirmation_family']]
        for index,job in enumerate(development,1):
            records[job['run_id']]=train_to(output,job,datasets[job['family']],cfg,protocol_hash,device,cfg['steps'])
            progress('development completion',index,len(development),job)
        frozen=output/'selection_manifest.json'
        if not frozen.exists():
            if any(r['family']==cfg['confirmation_family'] and r['completed_steps']>cfg['probe_steps'] and not r['numerical_failure'] for r in records.values()):
                raise RuntimeError('Confirmation candidates were continued without a preserved frozen selector; use a fresh output directory')
            allowed={'task_id','family','kind','prior','seed','arm','features','probe_val','final_val','probe_seconds','remaining_seconds','setup_seconds','train_seconds','diagnostic_seconds'}
            inputs=[{k:v for k,v in record.items() if k in allowed and not (record['family']==cfg['confirmation_family'] and k in ('final_val','remaining_seconds'))} for record in records.values()]
            atomic_json(output/'selection_inputs.json',inputs)
            manifest=freeze_choices(inputs,cfg)
            manifest.update(protocol_hash=protocol_hash,inputs_sha256=sha256(output/'selection_inputs.json'),frozen_at=now(),confirmation_completed_before_freeze=False)
            atomic_json(frozen,manifest)
            print('[selection frozen] five development family folds and Gray–Scott confirmation; no test data generated',flush=True)
        else:
            manifest=json.loads(frozen.read_text())
            if manifest['protocol_hash']!=protocol_hash or manifest['inputs_sha256']!=sha256(output/'selection_inputs.json'):
                raise ValueError('Frozen selection inputs or protocol changed')
        confirmation=[job for job in inventory if job['family']==cfg['confirmation_family']]
        for index,job in enumerate(confirmation,1):
            records[job['run_id']]=train_to(output,job,datasets[job['family']],cfg,protocol_hash,device,cfg['steps'])
            progress('confirmation completion',index,len(confirmation),job)
        plan_path=output/'evaluation_plan.json'
        if not plan_path.exists():
            full_choices={}
            for decision in manifest['decisions']:
                candidates=[r for r in records.values() if r['task_id']==decision['task_id']]
                selected=min(candidates,key=lambda r:(r['final_val'],cfg['arms'].index(r['arm'])))
                full_choices[decision['task_id']]=selected['arm']
            atomic_json(plan_path,dict(protocol_hash=protocol_hash,selection_sha256=sha256(frozen),locked_at=now(),full_validation_choices=full_choices,full_validation_role='expensive all-candidate validation reference only; cannot change frozen selector'))
        for family in cfg['families']:
            testing=get_data(output,family,cfg,source,'test')
            datasets[family].update({name:testing[name] for name in ('test','ood')})
        for index,job in enumerate(inventory,1):
            records[job['run_id']]=evaluate(output,job,datasets[job['family']],cfg,protocol_hash,device)
            progress('locked test evaluation',index,len(inventory),job)
        atomic_json(output/'bank.json',list(records.values()))
        report=report_run(output,args.color)
        if report['completion']['status']!='GREEN':
            raise RuntimeError('Final evidence validation failed; inspect reports/report.json')
        atomic_json(status_path,dict(status='COMPLETE',phase='reported',expected_fits=len(inventory),evaluated_fits=len(records),updated_at=now(),committed_wall_seconds=previous_wall+time.perf_counter()-started))
        archive=export_results(output)
        print(f'Results ZIP: {archive}',flush=True)
        return 0
    except BaseException as exc:
        atomic_json(status_path,dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',error=str(exc),updated_at=now(),committed_wall_seconds=previous_wall+time.perf_counter()-started,expected_fits=len(inventory)))
        print(f'\n[RED] Run stopped: {exc}\nCommitted checkpoints are preserved. Rerun the same command to resume.',file=sys.stderr,flush=True)
        try:
            export_results(output)
            print(f'Partial diagnostic archive: {output/"full_results.zip"}',file=sys.stderr)
        except Exception:pass
        raise
    finally:lock.close()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run');p.add_argument('--profile',choices=('smoke','pilot','full'),default='full');p.add_argument('--output',required=True);p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto');p.add_argument('--threads',type=int,default=8);p.add_argument('--allow-cpu-full',action='store_true');p.add_argument('--color',choices=('auto','always','never'),default='auto')
    p=sub.add_parser('report');p.add_argument('--output',required=True);p.add_argument('--color',choices=('auto','always','never'),default='auto')
    p=sub.add_parser('plan');p.add_argument('--profile',choices=('smoke','pilot','full'),default='full')
    args=parser.parse_args(argv)
    if args.command=='plan':
        cfg=config(args.profile);n=len(list(jobs(cfg)))
        print(json.dumps(dict(profile=args.profile,families=cfg['families'],confirmation_family=cfg['confirmation_family'],fits=n,optimizer_updates=n*cfg['steps'],probe_steps=cfg['probe_steps'],nominal_selected_route_updates=len(cfg['arms'])*cfg['probe_steps']+cfg['steps']-cfg['probe_steps'],single_arm_updates=cfg['steps'],scientific_evaluation=args.profile=='full'),indent=2));return 0
    if args.command=='report':report_run(args.output,args.color);return 0
    if args.threads<1:parser.error('--threads must be positive')
    return run(args)

if __name__=='__main__':sys.exit(main())
