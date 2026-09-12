"""Critical estimator and test-access safeguards; no neural training."""
import functools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import runner
from calibrator import fit_setting, rollout_error

class CalibrationProtocol(unittest.TestCase):
    def test_recovers_unknown_shared_coefficients_from_observed_transitions(self):
        rng=np.random.default_rng(914)
        calls=[]
        def step(family,state,action,params,mismatch,fidelity):
            calls.append(mismatch)
            return state+params[:,:,None]*action[:,None,:]
        true=rng.uniform(.5,1.5,(12,3)).astype(np.float32)
        actions=rng.uniform(.01,.05,(12,65,1)).astype(np.float32)
        states=np.zeros((12,66,3,1),dtype=np.float32)
        for t in range(65):states[:,t+1]=states[:,t]+true[:,:,None]*actions[:,t,None,:]
        supplied=true*np.array([.7,1.3,1.4],dtype=np.float32)
        data=dict(states=states,actions=actions,supplied_params=supplied)
        train={k:v[:8] for k,v in data.items()};val={k:v[8:] for k,v in data.items()}
        result=fit_setting(step,'fixture',train,val,[1,1,1],'structural','smoke')
        np.testing.assert_allclose(result['multipliers'],1/np.array([.7,1.3,1.4]),rtol=2e-4,atol=2e-5)
        self.assertLess(result['validation_nmse'],result['identity_validation_nmse'])
        self.assertEqual(set(calls),{'structural'})
        self.assertFalse(result['true_parameters_used_for_fitting'])

    def test_divergent_trajectory_remains_in_scoring(self):
        data=dict(states=np.zeros((2,41,1,1),np.float32),actions=np.zeros((2,40,1),np.float32),supplied_params=np.ones((2,3),np.float32))
        def broken(family,state,action,params,**kw):
            result=state.copy();result[0]=np.inf;return result
        curve,failed=rollout_error(broken,'fixture',data,[1],'correct',[1,1,1],horizon=32)
        self.assertEqual(failed,1);self.assertTrue(np.isinf(curve[0]).all());self.assertTrue((curve[1]==0).all())

    def test_modified_calibration_cannot_unlock_test_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);p=out/'jobs'/'j'/'calibration.json'
            runner.dump(out/'protocol.json',{'profile':'fixture'})
            runner.dump(p,{'multipliers':[1,1,1]})
            runner.dump(out/'freeze.json',{'protocol_sha256':runner.sha(out/'protocol.json'),'calibrations_sha256':{'j':runner.sha(p)}})
            runner.assert_frozen(out)
            runner.dump(p,{'multipliers':[2,1,1]})
            with self.assertRaises(RuntimeError):runner.assert_frozen(out)

    def test_test_data_requires_a_frozen_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                runner.ensure_data(Path(temp),'fixture','wave',{},'test','numpy','smoke')

if __name__=='__main__':unittest.main()
