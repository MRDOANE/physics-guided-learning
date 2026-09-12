import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from lpb import physics
from lpb.core import ROOT,canonical_hash
from lpb.guarded import fit_selector,decide,tasks_from_records
from lpb.confirmation_report import paired_effect,holm,label,route_cost
from lpb.confirmation_run import inventory
from lpb.datasets import get_data
from lpb.engine import train_to

class EvidenceTests(unittest.TestCase):
    def test_small_supported_gain_has_no_minimum_hurdle(self):
        b=np.ones((3,6,64));r=paired_effect([(b*.999,b)],1200,4);holm([r])
        self.assertEqual(label(r),'GREEN');self.assertLess(r['mse_reduction_percent'],1)
    def test_equal_outcomes_inconclusive(self):
        b=np.ones((3,6,16));r=paired_effect([(b,b)],600);holm([r]);self.assertEqual(label(r),'YELLOW')
    def test_opposite_direction_is_red(self):
        b=np.ones((3,6,16));r=paired_effect([(b*1.01,b)],600);holm([r]);self.assertEqual(label(r),'RED')
        self.assertEqual(label(r,expected='harm'),'GREEN')
    def test_smoke_is_not_evaluated(self):
        self.assertEqual(label(dict(ratio_ci95=[.7,.8],p_holm=.001),eligible=False),'NOT_EVALUATED')
    def test_holm(self):
        rows=[dict(p_two_sided=.01),dict(p_two_sided=.04),dict(p_two_sided=.02)]
        holm(rows);np.testing.assert_allclose([r['p_holm'] for r in rows],[.03,.04,.04])
    def test_seed_dependence_is_not_lost(self):
        b=np.ones((1,6,64));a=b*np.array([.1,.4,.8,1.5,3,8])[None,:,None]
        r=paired_effect([(a,b)],2000);self.assertLess(r['ratio_ci95'][0],1);self.assertGreater(r['ratio_ci95'][1],1)
    def test_failures_are_retained(self):
        b=np.ones((1,6,8));a=b.copy();a[:,0]=np.inf
        r=paired_effect([(a,b)],1000);self.assertGreater(r['mse_ratio'],1)
    def test_bad_pair_shape_fails(self):
        with self.assertRaises(ValueError):paired_effect([(np.ones((2,3)),np.ones((2,3)))])

class SelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.bank=json.loads((ROOT/'development/validation_bank.json').read_text());cls.model=fit_selector(cls.bank)
        tid=cls.bank[0]['task_id'];cls.target=[dict(r) for r in cls.bank if r['task_id']==tid]
    def test_no_test_fields_allowed(self):
        r=copy.deepcopy(self.target);r[0]['test_mse']=.01
        with self.assertRaises(ValueError):decide(self.model,r)
    def test_target_final_labels_cannot_change_choices(self):
        before=decide(self.model,self.target);r=copy.deepcopy(self.target)
        for x in r:x['final_val']=1e-12 if x['arm']=='iid' else 1e6
        self.assertEqual(before,decide(self.model,r))
    def test_decisions_ignore_family_prior_seed_as_features(self):
        r=copy.deepcopy(self.target)
        for x in r:x.update(family='new_equation',prior='new_prior',seed=991)
        self.assertEqual(decide(self.model,self.target)[0]['choices'],decide(self.model,r)[0]['choices'])
    def test_guard_is_fixed_mechanism(self):self.assertEqual(self.model['selected']['mode'],'worst_probe')
    def test_duplicated_arm_fails(self):
        with self.assertRaises(ValueError):tasks_from_records(self.target+[self.target[0]])
    def test_repeated_fit_reproduces(self):
        other=fit_selector(self.bank)
        self.assertEqual(other['selected'],self.model['selected']);self.assertEqual(other['bundle'],self.model['bundle'])

class PhysicsTests(unittest.TestCase):
    def test_mass_conservation(self):
        rng=np.random.default_rng(3);p=physics.sample_params('cahn_hilliard',2,rng);x=physics.initial_states('cahn_hilliard',2,64,rng)
        a=physics.make_actions('cahn_hilliard',2,80,rng);states=physics.simulate('cahn_hilliard',p,x,a)
        np.testing.assert_allclose(states.mean(-1),np.broadcast_to(x.mean(-1)[:,None],states.mean(-1).shape),atol=1e-12)
    def test_new_equations_against_independent_small_step_rk4(self):
        for name in ('cahn_hilliard','fitzhugh_nagumo'):
            rng=np.random.default_rng(55);x=torch.tensor(physics.initial_states(name,1,16,rng),dtype=torch.float64)
            p=torch.tensor(physics.sample_params(name,1,rng),dtype=torch.float64);drive=torch.tensor([[.3]],dtype=torch.float64)
            k=2*math.pi*torch.fft.rfftfreq(16,d=2*math.pi/16,dtype=torch.float64)
            a,b,c=p[0];phase=torch.arange(16,dtype=torch.float64)*2*math.pi/16
            def dxx(y):return torch.fft.irfft(-k.square()*torch.fft.rfft(y),n=16)
            def rhs(y):
                if name=='cahn_hilliard':return -a*dxx(dxx(y))+b*dxx(y.pow(3)-y)+c*drive[...,None]*phase.sin()
                u,v=y[:,0:1],y[:,1:2]
                return torch.cat((a*dxx(u)+u-u.pow(3)/3-v+c*drive[...,None]*phase.sin(),b*(u+.7-.8*v)),dim=1)
            y=x.clone();h=physics.spec(name)['dt']/1000
            for _ in range(1000):
                k1=rhs(y);k2=rhs(y+h*k1/2);k3=rhs(y+h*k2/2);k4=rhs(y+h*k3);y=y+h*(k1+2*k2+2*k3+k4)/6
            z=physics.step(name,x,drive,p,fidelity='check')
            np.testing.assert_allclose(z.numpy(),y.numpy(),rtol=1e-6,atol=1e-8)
    def test_structural_error_changes_equation(self):
        for f in ('cahn_hilliard','fitzhugh_nagumo'):
            rng=np.random.default_rng(4);x=torch.tensor(physics.initial_states(f,2,32,rng),dtype=torch.float64);p=torch.tensor(physics.sample_params(f,2,rng),dtype=torch.float64);a=torch.ones((2,1),dtype=torch.float64)
            self.assertGreater(float((physics.step(f,x,a,p)-physics.step(f,x,a,p,mismatch='structural')).abs().max()),1e-7)

class ExecutionTests(unittest.TestCase):
    def test_inventory_counts(self):
        cfg=json.loads((ROOT/'configs/full.json').read_text())
        self.assertEqual(sum(len(list(inventory(cfg,'resolution',g))) for g in cfg['resolution_grids']),288)
        self.assertEqual(len(list(inventory(cfg,'selector',64))),432)
    def test_test_data_locked(self):
        cfg=json.loads((ROOT/'configs/smoke.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):get_data(d,'wave',cfg,'unit','test')
    def test_exact_resume(self):
        cfg=json.loads((ROOT/'configs/smoke.json').read_text())|dict(grid=32)
        job=list(inventory(cfg,'resolution',32))[0]
        with tempfile.TemporaryDirectory() as d1,tempfile.TemporaryDirectory() as d2:
            data=get_data(d1,'wave',cfg,'unit');train_to(d1,job,data,cfg,'sig','cpu',cfg['steps'])
            train_to(d2,job,data,cfg,'sig','cpu',cfg['probe_steps']);train_to(d2,job,data,cfg,'sig','cpu',cfg['steps'])
            p1=torch.load(Path(d1)/'runs'/job['run_id']/'checkpoint.pt',weights_only=False)
            p2=torch.load(Path(d2)/'runs'/job['run_id']/'checkpoint.pt',weights_only=False)
            for k in p1['model']:self.assertTrue(torch.equal(p1['model'][k],p2['model'][k]),k)
    def test_none_cost_excludes_unused_bank(self):
        r=dict(train_seconds=10.,setup_seconds=1.,bank_setup_seconds=5.)
        self.assertEqual(route_cost({'none':r},'none','none','worst_probe'),11.)

if __name__=='__main__':unittest.main()
