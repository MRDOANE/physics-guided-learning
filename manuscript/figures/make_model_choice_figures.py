#!/usr/bin/env python3
"""Rebuild revised manuscript figures from the archived calibration report.

Usage: python make_model_choice_figures.py --report /path/to/report.json --output figures
Plotting uses saved measurements only. No fitting, selection, or test-time tuning.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MultipleLocator, FormatStrFormatter
from matplotlib.lines import Line2D

CASES = [
 ('resolution_g64', 'wave', 'coefficient', 'Wave'),
 ('resolution_g64', 'advection_diffusion', 'coefficient', 'Advection–diffusion'),
 ('resolution_g64', 'allen_cahn', 'coefficient', 'Allen–Cahn'),
 ('development', 'burgers', 'coefficient', 'Burgers'),
 ('development', 'ks', 'coefficient', 'Kuramoto–Sivashinsky'),
 ('development', 'gray_scott', 'coefficient', 'Gray–Scott'),
 ('selector_g64', 'cahn_hilliard', 'coefficient', 'Cahn–Hilliard'),
 ('selector_g64', 'fitzhugh_nagumo', 'coefficient', 'FitzHugh–Nagumo'),
 ('selector_g64', 'cahn_hilliard', 'structural', 'Cahn–Hilliard'),
 ('selector_g64', 'fitzhugh_nagumo', 'structural', 'FitzHugh–Nagumo'),
]
METHODS = ['uncalibrated', 'calibrated', 'validation_best_neural']
LABELS = {'uncalibrated':'Supplied physics', 'calibrated':'Calibrated physics',
          'validation_best_neural':'Validation-selected neural'}
COLORS = {'uncalibrated':'#D17B18','calibrated':'#176D8F','validation_best_neural':'#80428E'}
MARKERS = {'uncalibrated':'s','calibrated':'o','validation_best_neural':'^'}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def save(fig, root, basename):
    for ext in ['png','pdf','svg']:
        fig.savefig(root / (basename+'.'+ext), dpi=300, facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('figures'))
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    r=json.loads(args.report.read_text())
    provenance={'source_report_sha256':hashlib.sha256(args.report.read_bytes()).hexdigest(),
                'source_report_file':'reports/report.json', 'source_schema_version':r['schema_version'],
                'python_plotting_only':True, 'new_fitting':False}
    assert r['verified_settings']==30 and r['requested_settings']==30
    assert len(r['integrity_errors'])==0
    absolute={(x['stage'],x['family'],x['prior'],x['split'],x['horizon'],x['method']):x
              for x in r['absolute_accuracy']}
    rows=[]
    for stage,family,prior,label in CASES:
        for method in METHODS:
            a=absolute[(stage,family,prior,'test',64,method)]
            rows.append({**a, 'display_family':label, 'mean_nmse_display':format(a['mean_nmse'],'.3g'),
                         'ci95_low_display':format(a['ci95_low'],'.3g'),
                         'ci95_high_display':format(a['ci95_high'],'.3g')})
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (args.output/'primary_cases_accuracy.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    write_json(args.output/'figure1_plotted_data.json',{'provenance':provenance,
               'selection_of_displayed_cases':'One setting per family at the highest available grid. Wave, advection–diffusion and Allen–Cahn use resolution_g64; Burgers, Kuramoto–Sivashinsky and Gray–Scott use development; Cahn–Hilliard and FitzHugh–Nagumo use selector_g64. Both structural settings are shown.',
               'split':'test', 'horizon':64, 'rows':rows})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,(a,b)=plt.subplots(2,1,figsize=(7.6,6.65),gridspec_kw={'height_ratios':[8,2.4],'hspace':.34})
    fig.subplots_adjust(left=.255,right=.98,bottom=.10,top=.87)
    for ax,cases,title in [(a,CASES[:8],'(a) Incorrect coefficients'),(b,CASES[8:],'(b) Missing equation terms')]:
        yy=np.arange(len(cases));
        for m,offset in zip(METHODS,[-.21,0,.21]):
            arr=[absolute[(stage,family,prior,'test',64,m)] for stage,family,prior,_ in cases]
            means=np.array([v['mean_nmse'] for v in arr]);lo=np.array([v['ci95_low'] for v in arr]);hi=np.array([v['ci95_high'] for v in arr])
            ax.errorbar(means,yy+offset,xerr=np.stack([means-lo,hi-means]),fmt=MARKERS[m],
                        color=COLORS[m],markersize=5.2,capsize=2,elinewidth=1.0,markeredgewidth=.7,
                        label=LABELS[m],zorder=3)
        ax.set_xscale('log');ax.set_xlim(7e-14,8);ax.set_ylim(len(cases)-.48,-.48)
        ax.set_yticks(yy);ax.set_yticklabels([f'{label} ({absolute[(stage,family,prior,"test",64,"calibrated")]["grid"]})' for stage,family,prior,label in cases],fontsize=9.3)
        ax.set_xticks([1e-12,1e-9,1e-6,1e-3,1]);ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.grid(axis='x',color='#DFE4E8',linewidth=.7);ax.tick_params(axis='y',length=0,pad=8)
        ax.set_title(title,loc='left',pad=11,fontweight='bold')
        ax.spines['left'].set_visible(False)
    a.tick_params(axis='x',labelbottom=False)
    b.set_xlabel('Normalized mean squared error at 64 steps (log scale; lower is better)',labelpad=9)
    handles=[Line2D([0],[0],marker=MARKERS[m],color=COLORS[m],linestyle='none',markersize=6,label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.983),ncol=3,frameon=False,
               columnspacing=1.4,handletextpad=.45,fontsize=9)
    save(fig,args.output,'figure1_absolute_accuracy')

    timing=[]
    for record in r['fresh_timing_records']:
        t=record['record']
        for row in t['rows']:
            timing.append({'job_id':t['job']['id'],'stage':t['job']['stage'],**row})
    write_json(args.output/'figure2_plotted_data.json',{'provenance':provenance,
               'measurement':'Median of five blocks of 20 repeated forward calls; one trajectory, one CPU intra-operation thread; initialized neural architectures.',
               'rows':timing})
    with (args.output/'matched_cpu_timing.csv').open('w',newline='') as f:
        keys=list(dict.fromkeys(k for row in timing for k in row));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(timing)
    kinds=['physics_calibrated','fno','transformer','looped']
    kinds_labels=['Calibrated physics','Fourier neural operator','Transformer','Looped transformer']
    fig,ax=plt.subplots(figsize=(7.6,3.8));fig.subplots_adjust(left=.255,right=.975,bottom=.2,top=.84)
    summaries={}
    for i,kind in enumerate(kinds):
        rr=[x for x in timing if x['kind']==kind];vals=np.array([x['median_ms_per_batch_step'] for x in rr])
        summaries[kind]={'n_settings':len(vals),'median_ms':float(np.median(vals)),
                         'minimum_ms':float(vals.min()),'maximum_ms':float(vals.max()),
                         'q25_ms':float(np.percentile(vals,25)),'q75_ms':float(np.percentile(vals,75))}
        color=COLORS['calibrated'] if kind=='physics_calibrated' else COLORS['validation_best_neural']
        # Box displays the distribution across different settings, not a sampling interval.
        bp=ax.boxplot([vals],vert=False,positions=[i],widths=.37,showfliers=False,patch_artist=True,
                      medianprops={'color':'#25343D','linewidth':1.8},
                      boxprops={'facecolor':'#F1F3F5','edgecolor':'#ADB5BD','linewidth':.8},
                      whiskerprops={'color':'#ADB5BD','linewidth':.8},capprops={'color':'#ADB5BD','linewidth':.8},zorder=1)
        for grid,marker in [(32,'o'),(64,'s')]:
            sel=[x for x in rr if x['grid']==grid]
            # Deterministic offsets avoid stochastic layout changes between renders.
            jitter=np.linspace(-.19,.19,len(sel)) if len(sel)>1 else np.array([0.])
            ax.scatter([x['median_ms_per_batch_step'] for x in sel],i+jitter,s=19,
                       marker=marker,facecolor=color,edgecolors='white',linewidth=.4,alpha=.82,zorder=3)
    ax.set_yticks(range(4));ax.set_yticklabels(kinds_labels,fontsize=9.5)
    ax.set_ylim(3.55,-.55);ax.set_xlim(0,3.0);ax.set_xlabel('CPU time per forward step (ms; lower is faster)',labelpad=9)
    ax.xaxis.set_major_locator(MultipleLocator(.5));ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y',length=0,pad=8);ax.grid(axis='x',color='#DFE4E8',linewidth=.7);ax.spines['left'].set_visible(False)
    handles=[Line2D([0],[0],marker=m,color='#667884',linestyle='none',label=f'{g} grid sites',markersize=5) for g,m in [(32,'o'),(64,'s')]]
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,1.2),frameon=False,ncol=2,fontsize=9)
    save(fig,args.output,'figure2_cpu_execution')

    def count_comparison(prior,candidate,baseline,split,horizon):
        rr=[x for x in r['comparisons'] if (x['prior'],x['candidate'],x['baseline'],x['split'],x['horizon'])==(prior,candidate,baseline,split,horizon)]
        return dict(collections.Counter(x['label'] for x in rr))
    evidence={'provenance':provenance,'completion':r['completion'],'n_settings':30,'n_families':8,
              'scientific_scope':r['scientific_evidence'], 'uncertainty_scope':r['uncertainty_scope'],
              'statistical_scope':r['statistical_scope'],
              'neural_selection_counts':dict(collections.Counter(d['neural_kind'] for j in r['jobs'] for d in j['frozen_decisions'])),
              'validation_decisions_total':sum(len(j['frozen_decisions']) for j in r['jobs']),
              'validation_decisions_note':'Seed-specific choices reuse test trajectories, physical fits and physical families. These are not 156 independent scientific replications.',
              'timing_summary':summaries,'execution':r['execution'],'priors':{}}
    for prior in ['correct','coefficient','structural']:
        jj=[j for j in r['jobs'] if j['prior']==prior]
        evidence['priors'][prior]={'settings':len(jj),'identity_retained':sum(j['selected_candidate']=='identity' for j in jj),
             'selected_options':dict(collections.Counter(d['selected_method'] for j in jj for d in j['frozen_decisions'])),
             'calibrated_vs_neural':{f'{split}_{h}':count_comparison(prior,'calibrated','validation_best_neural',split,h) for split in ['test','ood'] for h in [64,96]},
             'calibrated_vs_unchanged':{f'{split}_{h}':count_comparison(prior,'calibrated','uncalibrated',split,h) for split in ['test','ood'] for h in [64,96]}}
    ctimes=[j['calibration_total_seconds'] for j in r['jobs']]
    evidence['per_setting_calibration_seconds']={'median':float(np.median(ctimes)),'minimum':min(ctimes),'maximum':max(ctimes)}
    evidence['correct_coefficient_caveat']='Calibration made small numerical changes when coefficients were correct. At 64 steps in-distribution, four settings had higher calibrated error with pointwise intervals excluding zero (all Allen–Cahn stages and FitzHugh–Nagumo). Absolute errors remained around 1e-12 to 1e-11. This prevents a claim that calibration always improves an already-correct solver.'
    evidence['parameter_access_audit']='runner.supplied_split copies provided trajectory parameters, applies the known distortion once, and passes only states/actions/supplied_params to fit_setting. fit_setting keeps this allow-list. No ground-truth parameter field enters calibration. The external benchmark generator necessarily knows the true coefficients. Test generation and evaluation require a freeze file covering all requested fits and validation choices.'
    evidence['structural_multipliers']={j['family']:j['selected_multipliers'] for j in r['jobs'] if j['prior']=='structural'}
    expected=np.array([1/.7,1/1.3,1/1.4]);deltas=[np.max(np.abs(np.array(j['selected_multipliers'])/expected-1)) for j in r['jobs'] if j['prior']=='coefficient']
    evidence['coefficient_multiplier_max_relative_deviation']=float(max(deltas))
    evidence['correct_vs_neural_all_lower']=all(x['candidate_mean_nmse']<x['baseline_mean_nmse'] for x in r['comparisons'] if x['prior']=='correct' and x['candidate']=='calibrated' and x['baseline']=='validation_best_neural')
    evidence['calibrated_physics_faster_than_all_neural_in_each_setting']=all(next(x['median_ms_per_batch_step'] for x in t['record']['rows'] if x['kind']=='physics_calibrated') < min(x['median_ms_per_batch_step'] for x in t['record']['rows'] if x['kind'] in ['fno','transformer','looped']) for t in r['fresh_timing_records'])
    write_json(args.output/'evidence_summary.json',evidence)
    captions={
      'figure1':'Absolute forecast error with incorrect coefficients and missing equation terms. Each point is mean normalized squared error across 64 ordinary test trajectories, averaged over a 64-step recursive forecast. The selected neural result averages the candidate chosen by 16-step validation error for each archived training seed. All selected neural candidates used the Fourier neural operator. Error bars show exploratory pointwise 95% bootstrap intervals; calibrated physical fits and the training dataset are held fixed. Panel (a) shows one setting per physical family using the highest available grid; panel (b) shows both missing-term settings. The number beside each family is the number of grid sites. The horizontal axis is logarithmic. Errors near numerical precision reflect noiseless data generated from the available equation family with a coefficient distortion represented by the calibration model.',
      'figure2':'Matched CPU execution times across 30 physical settings. Each point is a setting’s median time across five blocks of 20 forward calls. Circles and squares indicate 32 and 64 grid sites. Boxes show the median and interquartile range across settings, with conventional 1.5-interquartile-range whiskers. Calls used one trajectory and one CPU intra-operation thread with inputs resident in memory. Physics timing includes the coarse numerical step and application of fitted coefficient multipliers. Neural timing includes normalization and denormalization and uses initialized copies of the archived architectures; trained checkpoints were unavailable. These execution measurements are separate from the saved trained-model accuracy measurements. The test excludes data transfer, history updates, error calculation, fitting and full-rollout overhead. All measurements used PyTorch 2.8 on the same CPU environment.'
    }
    write_json(args.output/'captions.json',captions)
    (args.output/'captions.md').write_text('\n\n'.join(f'**{k.replace("figure", "Figure ")}.** {v}' for k,v in captions.items())+'\n')
    print(f'Wrote two figures, plot inputs, CSV tables, captions and evidence summary to {args.output}')

if __name__=='__main__': main()
