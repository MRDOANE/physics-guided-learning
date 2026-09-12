#!/usr/bin/env python3
"""Independently replay primary manuscript means and validation choices from saved arrays."""
import argparse,hashlib,json,collections
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 report=json.loads((a.run/'reports/report.json').read_text());freeze=json.loads((a.run/'freeze.json').read_text())
 metadata=json.loads((a.archive/'record_metadata.json').read_text());index=json.loads((a.archive/'score_index.json').read_text());saved=np.load(a.archive/'paired_trajectory_scores.npz',allow_pickle=False)
 lookup={(x['category'],tuple(x['key'])):x['array'] for x in index}
 errors={(x['job_id'],x['split'],x['horizon'],x['method']):x['mean_nmse'] for x in report['absolute_accuracy']}
 count=0;decision_count=0;max_abs=0.;max_rel=0.;all_means=[]
 for j in report['jobs']:
  folder=a.run/'jobs'/j['id'];cal=json.loads((folder/'calibration.json').read_text())
  assert hashlib.sha256((folder/'calibration.json').read_bytes()).hexdigest()==freeze['calibrations_sha256'][j['id']]
  jobrecords=[r for r in metadata if all(r[k]==j[k] for k in ['stage','family','prior'])]
  decisions=freeze['decisions'][j['id']]
  assert decisions==j['frozen_decisions']
  for d in decisions:
   neural=min([r for r in jobrecords if r['seed']==d['seed']],key=lambda r:(r['final_val'],r['kind'],r['arm']))
   assert (neural['kind'],neural['arm'])==(d['neural_kind'],d['neural_arm'])
   options=[('uncalibrated',cal['identity_validation_nmse']),('calibrated',cal['validation_nmse']),('neural',neural['final_val'])]
   chosen,score=min(options,key=lambda x:x[1]);assert chosen==d['selected_method'] and score==d['validation_nmse'];decision_count+=1
  with np.load(folder/'evaluation.npz',allow_pickle=False) as data:
   for split in ['test','ood']:
    ids=saved[lookup[('trajectory_ids',(j['stage'],j['family'],split))]]
    assert np.array_equal(ids,data[split+'_ids'])
    for h in [64,96]:
     means={};traj={}
     for method in ['uncalibrated','calibrated','persistence']:
      x=np.asarray(data[split+'_'+method][:,:h],dtype=np.float64)
      x=np.nan_to_num(x,nan=1e6,posinf=1e6,neginf=1e6)
      scores=np.minimum(x.mean(axis=1),1e6) if j['stage']=='development' else np.minimum(x,1e6).mean(axis=1)
      traj[method]=scores;means[method]=float(scores.mean())
     nn=[];route=[]
     for d in decisions:
      k=(j['stage'],j['family'],d['neural_kind'],j['prior'],d['seed'],d['neural_arm'],split,h)
      scores=saved[lookup[('neural',k)]];nn.append(scores)
      route.append(scores if d['selected_method']=='neural' else traj[d['selected_method']])
     means['validation_best_neural']=float(np.stack(nn).mean())
     means['validation_selected_route']=float(np.stack(route).mean())
     for method,v in means.items():
      target=errors[j['id'],split,h,method];diff=abs(v-target);rel=diff/max(abs(target),1e-30)
      assert np.isclose(v,target,atol=1e-15,rtol=1e-12),(j['id'],split,h,method,v,target)
      max_abs=max(max_abs,diff);max_rel=max(max_rel,rel);count+=1
      all_means.append({'job_id':j['id'],'split':split,'horizon':h,'method':method,'recomputed_mean_nmse':v})
 out={'status':'PASS','means_recomputed':count,'validation_choices_recomputed':decision_count,
      'frozen_calibration_hashes_verified':len(report['jobs']),'trajectory_id_pairs_verified':2*len(report['jobs']),
      'max_abs_mean_difference':max_abs,'max_relative_mean_difference':max_rel,
      'scope':'Replay from raw physical error curves, archived per-trajectory neural scores, archived validation scores and the frozen calibration outputs. Bootstrap intervals are supplied by the original analysis and were not refitted by this independent mean/selection check.',
      'means':all_means}
 a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='means'},indent=2))
if __name__=='__main__':main()
