#!/usr/bin/env python3
"""A no-training plan by default; all scientific execution is explicit."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',nargs='?',default='plan',choices=['plan','smoke','full','resolution','selector','report','benchmark'])
    p.add_argument('--output',default=str(ROOT/'outputs'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--max-hours',type=float,default=0,help='Soft limit per invocation; pause between committed jobs. Pod billing continues.')
    p.add_argument('--hourly-rate',type=float,default=0,help='Your actual total Pod price in USD/hour, for cost estimates only.')
    p.add_argument('--profile',choices=['smoke','full','benchmark'],default='full',help='Profile to show with report/plan.')
    args=p.parse_args()
    if args.threads<1 or args.max_hours<0 or args.hourly_rate<0:p.error('Invalid resource option')
    if args.command=='plan':
        cfg=json.loads((ROOT/'configs/full.json').read_text())
        r=len(cfg['resolution_families'])*len(cfg['resolution_grids'])*len(cfg['kinds'])*2*2*len(cfg['resolution_seeds'])
        s=len(cfg['confirmation_families'])*len(cfg['kinds'])*3*4*len(cfg['confirmation_seeds'])
        hours=(12,30)
        plan=dict(status='NOT RUN: this command does not import PyTorch or train models',resolution_fits=r,selector_fits=s,total_new_fits=r+s,
                  updates=(r+s)*cfg['steps'],reused_historical_validation_records=864,reference_grid=64,
                  hardware='One 24 GB GPU; 8-16 vCPUs; 32 GB RAM; 50 GB persistent storage.',
                  runtime_hours_unmeasured_gpu_estimate=hours,
                  estimated_compute_cost_usd=[round(h*args.hourly_rate,2) for h in hours] if args.hourly_rate else 'Supply --hourly-rate with the displayed Pod price.',
                  primary_limit='Expected GPU/training-launch throughput. CPU handles float64 simulation and data preparation. Benchmark prints both timings.',
                  success='GREEN supported; YELLOW inconclusive; RED evidence against the stated direction; NOT_EVALUATED for smoke/incomplete stages. No minimum gain or required number of winning systems.',
                  commands=['bash launch_physics_confirmation.sh benchmark --hourly-rate YOUR_RATE','bash launch_physics_confirmation.sh resolution','bash launch_physics_confirmation.sh selector','bash launch_physics_confirmation.sh full'])
        print(json.dumps(plan,indent=2));return 0
    if args.command=='report':
        from lpb.confirmation_report import report
        return report(Path(args.output)/args.profile)['exit_code']
    from lpb.confirmation_run import run
    return run(args)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print(f'[RED] EXECUTION OR INTEGRITY ERROR: {exc}\nThis error is not evidence against a scientific hypothesis.',file=sys.stderr)
        sys.exit(2)
