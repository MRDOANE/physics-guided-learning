#!/usr/bin/env python3
"""Calibrate physical coefficients and reuse archived neural forecast errors.

No neural optimizer or neural training module is imported. Train/validation
calibrations are frozen before held-out trajectories or archived scores are read.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import functools
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = os.environ.get('THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)
def save_npz(path, arrays):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream: np.savez_compressed(stream, **arrays)
    tmp.replace(path)
def verify_release():
    manifest = ROOT/'MANIFEST.sha256'
    if not manifest.exists(): raise RuntimeError('Release manifest is missing')
    for line in manifest.read_text().splitlines():
        expected, rel = line.split('  ', 1)
        path = (ROOT/rel).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != expected:
            raise RuntimeError('Release checksum failed: '+rel)
    return sha(manifest)

def choose_jobs(profile):
    jobs = read(ASSETS/'jobs.json')
    if profile == 'pilot':
        return [j for j in jobs if j['stage']=='resolution_g64' and j['family']=='allen_cahn']
    if profile == 'smoke':
        keys = {('resolution_g64','wave','coefficient'), ('resolution_g64','allen_cahn','coefficient'),
                ('selector_g64','cahn_hilliard','structural'), ('selector_g64','fitzhugh_nagumo','structural')}
        return [j for j in jobs if (j['stage'],j['family'],j['prior']) in keys]
    return jobs

def plan():
    print('CALIBRATED PHYSICS: RUN PLAN')
    print('30 settings across 8 physical families; 1,584 archived neural fits reused.')
    print('Three shared coefficient corrections per setting, learned from 128 training trajectories.')
    print('Six optimizer candidates plus unchanged physics; 24 validation trajectories select the candidate.')
    print('All choices freeze before test evaluation. 64 ID + 64 OOD trajectories; horizons 64 and 96.')
    print('SMOKE: 4 small diagnostic settings. PILOT: 2 complete Allen-Cahn settings at grid 64.')
    print('FULL: all 30 settings. REPORT: reconstruct results from saved outputs.')
    print('CPU-bound calibration: start with 8 vCPUs, JOBS=4, THREADS=1, 16 GB RAM.')
    print('GPU optional for matched forward timing; no neural training. CPU timing works without CUDA.')
    print('Outputs include paired accuracy tables, coefficient diagnostics, validation selection and measured cost.')
    print('Green/support, yellow/uncertain, red/opposite, blue/unchanged. No minimum required improvement.')
    print('Pilot reports measured timing. Full runtime depends on nonlinear solver convergence.')

def environment(backend_name):
    import numpy as np
    import scipy
    result = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  backend=backend_name, threads=int(os.environ.get('THREADS','1')), machine=platform.machine())
    if backend_name == 'torch':
        import torch
        torch.set_num_threads(result['threads'])
        result['torch'] = str(torch.__version__)
        result['cuda_runtime'] = torch.version.cuda
    return result

def load_data(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as src:
        names = sorted({k.split('_',1)[0] for k in src.files})
        return {name:{field:src[name+'_'+field].copy() for field in ('states','actions','params','ids')} for name in names}

def assert_frozen(out):
    frozen = read(out/'freeze.json')
    if frozen['protocol_sha256'] != sha(out/'protocol.json'):
        raise RuntimeError('Protocol changed after calibration freeze')
    for jobid, expected in frozen['calibrations_sha256'].items():
        if sha(out/'jobs'/jobid/'calibration.json') != expected:
            raise RuntimeError('Calibration changed after freeze: '+jobid)
    return frozen

def ensure_data(out, stage, family, cfg, phase, backend_name, profile):
    import numpy as np
    from data_backend import create_backend, regenerate
    if phase == 'test': assert_frozen(out)
    path = out/'data'/stage/f'{family}_{phase}.npz'; meta_path = path.with_suffix('.json')
    signature = canonical(dict(protocol=sha(out/'protocol.json'), stage=stage, family=family, phase=phase))
    if path.exists() and meta_path.exists():
        meta = read(meta_path)
        if meta['signature'] != signature or meta['sha256'] != sha(path):
            raise RuntimeError('Data cache mismatch: '+str(path))
        return path, 0.0
    start = time.perf_counter()
    backend = create_backend(backend_name)
    previous_threads = None
    if backend_name=='torch':
        import torch
        previous_threads=torch.get_num_threads()
        # The archive used eight CPU threads. FFT rounding can depend on this
        # count even though the generated tensors use float64 on CPU.
        torch.set_num_threads(8)
    try:
        data = regenerate(backend, family, cfg, phase, smoke=profile=='smoke')
    finally:
        if previous_threads is not None: torch.set_num_threads(previous_threads)
    arrays = {}
    names = ('train','val') if phase=='development' else ('test','ood')
    for split in names:
        for field in ('states','actions','params','ids'):
            arrays[split+'_'+field] = data[split][field]
    save_npz(path, arrays)
    original = read(ASSETS/stage/'data'/f'{family}_{phase}.json')
    meta = dict(signature=signature, sha256=sha(path), original_sha256=original['sha256'],
                original_description=original['description'], exact_archive_npz_hash=sha(path)==original['sha256'],
                backend=backend_name, generation_seconds=time.perf_counter()-start,
                archive_generation_threads=8 if backend_name=='torch' else None,
                smoke=profile=='smoke')
    dump(meta_path, meta)
    print(f'DATA {stage}/{family}/{phase}: {meta["generation_seconds"]:.1f} s; exact archive hash={meta["exact_archive_npz_hash"]}', flush=True)
    return path, meta['generation_seconds']

def supplied_split(backend, family, split, prior):
    import numpy as np
    # These are the same approximate coefficients available to the archived NN.
    # Original parameters are discarded from the object handed to calibration.
    params = np.asarray(split['params'], dtype=np.float32).copy()
    if prior == 'coefficient': params *= np.array([.70,1.30,1.40], dtype=np.float32)
    return dict(states=split['states'], actions=split['actions'], supplied_params=params)

def normalization(train):
    import numpy as np
    x = np.asarray(train['states'], dtype=np.float64)
    return dict(mean=x.mean(axis=(0,1,3)).astype(np.float32).tolist(),
                scale=np.maximum(x.std(axis=(0,1,3)),1e-6).astype(np.float32).tolist())

def fit_worker(payload):
    import numpy as np
    from calibrator import fit_setting
    from data_backend import create_backend
    out, job, backend_name, profile, archived_norm = payload
    out = Path(out); environment(backend_name)
    backend = create_backend(backend_name)
    path = out/'jobs'/job['id']/'calibration.json'
    signature = canonical(dict(protocol=sha(out/'protocol.json'), job=job,
                                data=sha(out/'data'/job['stage']/f'{job["family"]}_development.npz')))
    if path.exists():
        result = read(path)
        if result.get('signature') != signature: raise RuntimeError('Calibration resume mismatch: '+job['id'])
        return job['id'], result['total_seconds'], True
    if (out/'freeze.json').exists(): raise RuntimeError('Cannot add a calibration after test freeze')
    data = load_data(out/'data'/job['stage']/f'{job["family"]}_development.npz')
    norm = normalization(data['train'])
    used_norm = norm if profile=='smoke' else {k:archived_norm[k] for k in ('mean','scale')}
    norm_ok = all(np.allclose(norm[k],archived_norm[k],rtol=1e-5,atol=1e-7) for k in ('mean','scale'))
    train = supplied_split(backend,job['family'],data['train'],job['prior'])
    val = supplied_split(backend,job['family'],data['val'],job['prior'])
    result = fit_setting(functools.partial(backend.step,dtype=np.float64), job['family'], train, val,
                         used_norm['scale'], job['prior'], profile,
                         validation_step_fn=functools.partial(backend.step,dtype=np.float32), state_mean=used_norm['mean'])
    result.update(signature=signature, job=job, normalization=used_norm, regenerated_normalization=norm,
                  archived_normalization_compatible=bool(norm_ok), environment=environment(backend_name))
    dump(path, result)
    return job['id'], result['total_seconds'], False

def freeze(out, jobs, records):
    hashes = {j['id']:sha(out/'jobs'/j['id']/'calibration.json') for j in jobs}
    decisions = {}
    for job in jobs:
        c = read(out/'jobs'/job['id']/'calibration.json')
        rr = [r for r in records if all(r[k]==job[k] for k in ('stage','family','prior'))]
        rows = []
        for seed in sorted({r['seed'] for r in rr}):
            candidates = sorted((r for r in rr if r['seed']==seed),key=lambda r:(r['final_val'],r['kind'],r['arm']))
            nn = candidates[0]
            choices = [('uncalibrated', c['identity_validation_nmse']),
                       ('calibrated', c['validation_nmse']), ('neural',nn['final_val'])]
            selected, score = min(choices,key=lambda x:x[1])
            rows.append(dict(seed=seed, neural_kind=nn['kind'], neural_arm=nn['arm'],
                             neural_validation_nmse=nn['final_val'], selected_method=selected,
                             selected_kind=nn['kind'] if selected=='neural' else None,
                             selected_arm=nn['arm'] if selected=='neural' else None, validation_nmse=score,
                             historical_candidate_training_seconds=sum(r['train_seconds'] for r in candidates)))
        decisions[job['id']] = rows
    value = dict(protocol_sha256=sha(out/'protocol.json'), calibrations_sha256=hashes, decisions=decisions,
                 selection='Minimum 16-step validation NMSE. Ties use uncalibrated physics, calibrated physics, then neural.',
                 heldout_scores_read_for_selection=False,
                 note='This follow-up is exploratory; prior held-out results informed its design.')
    if (out/'freeze.json').exists():
        if read(out/'freeze.json') != value: raise RuntimeError('Frozen choices differ; resume stopped')
    else: dump(out/'freeze.json',value)
    assert_frozen(out)

def evaluate_job(out, job, backend_name, profile):
    import numpy as np
    from calibrator import rollout_error
    from data_backend import create_backend
    assert_frozen(out); environment(backend_name)
    folder = out/'jobs'/job['id']; c = read(folder/'calibration.json')
    signature = canonical(dict(freeze=sha(out/'freeze.json'),calibration=sha(folder/'calibration.json'),
                                test=sha(out/'data'/job['stage']/f'{job["family"]}_test.npz')))
    if (folder/'evaluation.json').exists():
        meta = read(folder/'evaluation.json')
        if meta.get('signature') != signature or meta['evaluation_sha256'] != sha(folder/'evaluation.npz'):
            raise RuntimeError('Evaluation resume checksum mismatch')
        return 0.0
    begin = time.perf_counter(); backend = create_backend(backend_name)
    raw = load_data(out/'data'/job['stage']/f'{job["family"]}_test.npz')
    horizon = 32 if profile=='smoke' else 96; arrays = {}; failures = {}
    step = functools.partial(backend.step,dtype=np.float32)
    scale = np.asarray(c['normalization']['scale'],dtype=np.float32)[None,None,:,None]
    for split in ('test','ood'):
        data = supplied_split(backend,job['family'],raw[split],job['prior'])
        for method, multipliers in [('uncalibrated',[1.,1.,1.]), ('calibrated',c['multipliers'])]:
            curve, failed = rollout_error(step,job['family'],data,c['normalization']['scale'],job['prior'],
                                          multipliers,horizon,numerical_dtype=np.float32,state_mean=c['normalization']['mean'])
            arrays[split+'_'+method] = curve; failures[split+'_'+method] = failed
        x = data['states']; arrays[split+'_persistence'] = np.mean(((x[:,3:4]-x[:,4:4+horizon])/scale)**2,axis=(2,3)).astype(np.float64)
        arrays[split+'_ids'] = raw[split]['ids']
    compatibility = dict(backend_is_original_torch=backend_name=='torch',full_data=profile!='smoke',
                         normalization=c['archived_normalization_compatible'],checks={},
                         atol=1e-7,rtol=1e-4,
                         scope='Reference error curves and trajectory IDs are numerical replay checks. They do not establish bitwise trajectory identity.')
    if profile!='smoke' and backend_name=='torch':
        with np.load(ASSETS/job['stage']/'references'/f'{job["family"]}__{job["prior"]}.npz',allow_pickle=False) as refs:
            for split in ('test','ood'):
                for method, archived_method in [('uncalibrated','mechanistic'),('persistence','persistence')]:
                    x = arrays[split+'_'+method]; y = refs[split+'_'+archived_method+'_mse']
                    ok = x.shape==y.shape and np.allclose(x,y,atol=1e-7,rtol=1e-4,equal_nan=False)
                    compatibility['checks'][split+'_'+method] = dict(passed=bool(ok), max_abs_difference=float(np.max(np.abs(x-y))) if np.isfinite(x).all() else None)
        index = read(ASSETS/'score_index.json')
        with np.load(ASSETS/'paired_trajectory_scores.npz',allow_pickle=False) as archived:
            for split in ('test','ood'):
                entry = next(v for v in index if v['category']=='trajectory_ids' and v['key']==[job['stage'],job['family'],split])
                compatibility['checks'][split+'_ids'] = dict(passed=bool(np.array_equal(arrays[split+'_ids'],archived[entry['array']])))
    compatibility['data_exact_hashes'] = {phase:read(out/'data'/job['stage']/f'{job["family"]}_{phase}.json')['exact_archive_npz_hash'] for phase in ('development','test')}
    exact_data = all(compatibility['data_exact_hashes'].values())
    ids_ok = all(compatibility['checks'].get(split+'_ids',{}).get('passed',False) for split in ('test','ood'))
    replay_ok = bool(compatibility['checks']) and all(r['passed'] for r in compatibility['checks'].values())
    compatible = bool(compatibility['backend_is_original_torch'] and compatibility['full_data'] and
                      compatibility['normalization'] and ids_ok and (exact_data or replay_ok))
    compatibility['reference_replay_passed']=replay_ok
    compatibility['eligibility_rule']='Original Torch source, full counts, ordered IDs and normalization; either exact development/test archive NPZ hashes or successful numerical reference replay.'
    compatibility['status'] = ('EXACT_DATA_HASH_AND_NUMERICAL_REPLAY' if replay_ok else 'EXACT_DATA_HASH_REFERENCE_ROUNDING_DIFFERS') if compatible and exact_data else ('NUMERICALLY_COMPATIBLE' if compatible else 'UNVERIFIED')
    compatibility['precision_note']='Exact input-data hashes establish common forecast targets even when CPU physical forecast arithmetic differs from the archived GPU reference. Reference discrepancies remain reported.'
    save_npz(folder/'evaluation.npz', arrays)
    seconds = time.perf_counter()-begin
    dump(folder/'evaluation.json',dict(signature=signature,evaluation_sha256=sha(folder/'evaluation.npz'),
         archive_compatible=compatible,compatibility=compatibility,seconds=seconds,failed_trajectories=failures,
         calibration_sha256=sha(folder/'calibration.json'),freeze_sha256=sha(out/'freeze.json')))
    print(f'EVALUATED {job["id"]}: archive={compatibility["status"]}; {seconds:.1f} s',flush=True)
    return seconds

def timing(out, jobs, configs, records, backend_name, profile):
    requested = os.environ.get('TIMING_DEVICE','auto').lower()
    if requested=='none' or backend_name!='torch':
        dump(out/'timing_status.json',dict(status='SKIPPED',reason='Disabled or NumPy diagnostic backend')); return
    import torch
    from archived_lpb import physics, models
    from matched_timing import benchmark_setting
    from data_backend import create_backend
    device = requested
    if requested=='auto':
        try:
            torch.zeros(1,device='cuda'); torch.cuda.synchronize(); device='cuda'
        except Exception: device='cpu'
    if device not in ('cpu','cuda'): raise ValueError('TIMING_DEVICE must be auto/cpu/cuda/none')
    failures = []
    for job in jobs:
        path=out/'timing'/f'{job["id"]}__{device}.json'
        timing_signature=canonical(dict(freeze=sha(out/'freeze.json'),job=job,device=device,
                 blocks=3 if profile=='smoke' else 5,repeats=10 if profile=='smoke' else 20,
                 threads=int(os.environ.get('THREADS','1')),environment=environment(backend_name)))
        if path.exists():
            previous=read(path)
            if previous.get('timing_signature') != timing_signature:
                raise RuntimeError('Timing cache differs from frozen job/environment: '+str(path))
            continue
        try:
            data=load_data(out/'data'/job['stage']/f'{job["family"]}_development.npz')
            c=read(out/'jobs'/job['id']/'calibration.json')
            norm=next(r['normalization'] for r in records if all(r[k]==job[k] for k in ('stage','family','prior')))
            if profile=='smoke':
                import numpy as np
                norm=dict(norm,**c['normalization']);p=supplied_split(None,job['family'],data['train'],job['prior'])['supplied_params']
                norm['param_mean']=p.mean(0).tolist();norm['param_scale']=np.maximum(p.std(0),1e-6).tolist()
            result=benchmark_setting(physics.step,models,job['family'],job['prior'],configs[job['stage']],
                        supplied_split(None,job['family'],data['val'],job['prior']),norm,c['multipliers'],
                        device=device,blocks=3 if profile=='smoke' else 5,repeats=10 if profile=='smoke' else 20,
                        threads=int(os.environ.get('THREADS','1')))
            result.update(job=job,freeze_sha256=sha(out/'freeze.json'),timing_signature=timing_signature)
            dump(path,result)
            print(f'TIMING {job["id"]}: {device}',flush=True)
        except Exception as exc:
            failures.append(dict(job=job['id'],error=f'{type(exc).__name__}: {exc}'))
            print(f'TIMING INCOMPLETE {job["id"]}: {exc}',flush=True)
    dump(out/'timing_status.json',dict(status='COMPLETE' if not failures else 'INCOMPLETE',device=device,failures=failures))

def package_results(out):
    path=out/'full_results.zip'; tmp=out/'full_results.zip.tmp'
    included=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p in (path,tmp) or p.suffix=='.tmp': continue
        if 'data' in p.relative_to(out).parts and p.suffix=='.npz': continue
        included.append(p)
    dump(out/'RESULTS_MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in included if p.name!='RESULTS_MANIFEST.json'})
    included=[p for p in included if p.name!='RESULTS_MANIFEST.json']+[out/'RESULTS_MANIFEST.json']
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in included: archive.write(p,str(p.relative_to(out)))
    tmp.replace(path)
    print('Results ZIP: '+str(path),flush=True)

def run(args):
    release_hash=verify_release()
    if args.profile=='plan': plan(); return
    if args.profile=='report':
        out=Path(os.environ.get('OUTPUT_DIR',str(ROOT/'outputs'/'full'))).resolve()
        from analysis import analyze
        analyze(out,ASSETS);package_results(out);return
    profile=args.profile;backend_name=os.environ.get('BACKEND','torch')
    if backend_name not in ('torch','numpy'): raise ValueError('BACKEND must be torch/numpy')
    if profile!='smoke' and backend_name!='torch': raise ValueError('Scientific profiles use the original Torch solver; NumPy is for diagnostic smoke only')
    jobs=choose_jobs(profile);workers=max(1,min(int(os.environ.get('JOBS','4')),len(jobs)))
    out=Path(os.environ.get('OUTPUT_DIR',str(ROOT/'outputs'/profile))).resolve();out.mkdir(parents=True,exist_ok=True)
    configs=read(ASSETS/'configurations.json');records=read(ASSETS/'record_metadata.json')
    protocol=dict(schema_version=1,release_sha256=release_hash,profile=profile,backend=backend_name,jobs=jobs,
                  environment=environment(backend_name),primary_horizon=16 if profile=='smoke' else 64,
                  archive_generation_threads=8,
                  secondary_horizon=32 if profile=='smoke' else 96,bootstrap_replicates=5000,
                  selection_horizon=16,coefficient_bounds=[.001,1000.],coarse_solver=True,
                  test_read_barrier='All requested training fits and validation selections must freeze first.',
                  scientific_scope='Exploratory follow-up on previously inspected synthetic systems.',
                  minimum_effect_requirement=None,required_successful_system_count=None,
                  confidence_intervals='Paired pointwise percentile bootstrap; conditional on fitted calibration; no multiplicity adjustment.')
    if (out/'protocol.json').exists():
        if read(out/'protocol.json') != protocol: raise RuntimeError('Output protocol/software differs. Use a new OUTPUT_DIR.')
    else: dump(out/'protocol.json',protocol)
    begin=time.perf_counter();data_seconds=0.;cal_seconds=0.;eval_seconds=0.;timing_seconds=0.
    prior_execution=read(out/'execution.json') if (out/'execution.json').exists() else {}
    for stage,family in sorted({(j['stage'],j['family']) for j in jobs}):
        _,s=ensure_data(out,stage,family,configs[stage],'development',backend_name,profile);data_seconds+=s
    payloads=[]
    for job in jobs:
        norm=next(r['normalization'] for r in records if all(r[k]==job[k] for k in ('stage','family','prior')))
        payloads.append((str(out),job,backend_name,profile,norm))
    fit_start=time.perf_counter()
    if workers==1: completed=map(fit_worker,payloads)
    else:
        pool=concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
        completed=(future.result() for future in concurrent.futures.as_completed([pool.submit(fit_worker,p) for p in payloads]))
    try:
        for jobid,seconds,reused in completed:
            print(f'CALIBRATION {jobid}: {seconds:.1f} s'+(' (reused)' if reused else ''),flush=True)
    finally:
        if workers>1: pool.shutdown(wait=True,cancel_futures=True)
    cal_seconds=time.perf_counter()-fit_start
    freeze(out,jobs,records);print('FROZEN: every requested calibration and validation selection. Held-out evaluation unlocked.',flush=True)
    for stage,family in sorted({(j['stage'],j['family']) for j in jobs}):
        _,s=ensure_data(out,stage,family,configs[stage],'test',backend_name,profile);data_seconds+=s
    for job in jobs: eval_seconds+=evaluate_job(out,job,backend_name,profile)
    tick=time.perf_counter();timing(out,jobs,configs,records,backend_name,profile);timing_seconds=time.perf_counter()-tick
    elapsed=time.perf_counter()-begin
    sessions=prior_execution.get('sessions',[])+[dict(elapsed_seconds=elapsed,data_seconds=data_seconds,
                  calibration_wall_seconds=cal_seconds,evaluation_seconds=eval_seconds,timing_seconds=timing_seconds,
                  jobs=workers,threads=int(os.environ.get('THREADS','1')))]
    execution={key:sum(s[key] for s in sessions) for key in ('elapsed_seconds','data_seconds','calibration_wall_seconds','evaluation_seconds','timing_seconds')}
    rate=os.environ.get('POD_HOURLY_RATE','').strip()
    execution.update(sessions=sessions,pod_hourly_rate=float(rate) if rate else None,new_neural_training=False,
                     cost_excludes='Storage, idle time, dependency installation, and previous neural training. Interrupted sessions before this checkpoint may be undercounted.')
    if profile=='pilot':
        total=sum(read(out/'jobs'/j['id']/'calibration.json')['total_seconds'] for j in jobs)
        estimate=total/len(jobs)*30/max(1,int(os.environ.get('JOBS','4')))
        execution['pilot_full_fit_wall_seconds_simple_projection']=estimate
        execution['projection_caveat']='Allen-Cahn timings project equal setting costs; other equation families and optimizer convergence differ. Excludes data/evaluation/timing.'
        print(f'PILOT: simple full calibration projection {estimate/60:.1f} minutes; family-dependent, excludes data and evaluation.',flush=True)
    dump(out/'execution.json',execution)
    from analysis import analyze
    analyze(out,ASSETS);package_results(out)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('plan','smoke','pilot','full','report'),default='plan')
    args=parser.parse_args()
    try: run(args)
    except Exception as exc:
        print(f'COMPLETION: RED | {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        print('Scientific direction is separate from execution completion. Existing valid files are preserved.',file=sys.stderr)
        traceback.print_exc();return 1
    return 0
if __name__=='__main__': raise SystemExit(main())
