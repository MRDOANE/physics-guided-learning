"""Small full-width training microbenchmark; no scientific evaluation data."""
import os
import statistics
import subprocess
import time
import torch
from .models import build_model
from .physics import spec

def probe(cfg,device,threads):
    rows=[];utilization=[]
    for count in sorted({1,min(4,threads),threads}):
        torch.set_num_threads(count)
        for kind in cfg['kinds']:
            model=build_model(kind,spec('wave',64),cfg).to(device)
            history=torch.randn((cfg['batch_size'],cfg['context'],2,64),device=device)
            actions=torch.randn((cfg['batch_size'],cfg['context'],1),device=device)
            params=torch.randn((cfg['batch_size'],3),device=device)
            target=torch.randn((cfg['batch_size'],2,64),device=device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'])
            for _ in range(5):
                optimizer.zero_grad(set_to_none=True);(model(history,actions,params)-target).square().mean().backward();optimizer.step()
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(40):
                optimizer.zero_grad(set_to_none=True);(model(history,actions,params)-target).square().mean().backward();optimizer.step()
            if device=='cuda':torch.cuda.synchronize()
            rows.append(dict(architecture=kind,cpu_threads=count,training_ms_per_update=(time.perf_counter()-start)*1000/40))
    torch.set_num_threads(threads)
    return dict(device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,cpu_logical_count=os.cpu_count(),configured_threads=threads,rows=rows,
        interpretation='Same full-width synthetic training workload across CPU thread counts. This measures training throughput, not model accuracy. A faster GPU may not help if host launches dominate. Choose the thread count before a full run; changing it invalidates exact resume. No GPU-versus-CPU comparison is inferred from a CPU-only benchmark.')
