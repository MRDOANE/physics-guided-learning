"""Integration checks for interrupted training, pairing, and test isolation."""
import json
from pathlib import Path
from unittest import mock
import numpy as np
import pytest
import torch
from lpb import engine
from lpb.core import config, jobs, atomic_json
from lpb.datasets import get_data
from lpb.training import _load_checkpoint

@pytest.fixture
def small(tmp_path):
    torch.set_num_threads(1)
    cfg=config('smoke');job=next(jobs(cfg));job.update(arm='smooth',run_id=job['task_id']+'__smooth')
    data=get_data(tmp_path/'data_source','wave',cfg,'unit-test-source')
    return cfg,job,data

def weights(folder,job):
    return _load_checkpoint(folder/'runs'/job['run_id']/'checkpoint.pt','cpu')

def test_probe_continuation_matches_uninterrupted(tmp_path,small):
    cfg,job,data=small
    engine.train_to(tmp_path/'direct',job,data,cfg,'protocol','cpu',cfg['steps'])
    probe=engine.train_to(tmp_path/'phased',job,data,cfg,'protocol','cpu',cfg['probe_steps'])
    engine.train_to(tmp_path/'phased',job,data,cfg,'protocol','cpu',cfg['steps'])
    a,b=weights(tmp_path/'direct',job),weights(tmp_path/'phased',job)
    assert a['validation_history']==b['validation_history']
    assert b['record']['probe_val']==probe['probe_val']
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key]),key

def test_actual_checkpoint_interruption_repairs_stale_record(tmp_path,small):
    cfg,job,data=small
    folder=tmp_path/'interrupted'
    engine.train_to(folder,job,data,cfg,'protocol','cpu',cfg['probe_steps'])
    original=engine._save_checkpoint
    def stop_after_commit(path,payload):
        original(path,payload)
        if Path(path).name=='checkpoint.pt' and payload['step']==cfg['steps']:
            raise KeyboardInterrupt('synthetic power interruption after checkpoint commit')
    with mock.patch.object(engine,'_save_checkpoint',side_effect=stop_after_commit):
        with pytest.raises(KeyboardInterrupt):engine.train_to(folder,job,data,cfg,'protocol','cpu',cfg['steps'])
    rp=folder/'runs'/job['run_id']/'record.json'
    assert json.loads(rp.read_text())['completed_steps']<cfg['steps']
    result=engine.train_to(folder,job,data,cfg,'protocol','cpu',cfg['steps'])
    assert json.loads(rp.read_text())['completed_steps']==cfg['steps']
    assert result['probe_val']==weights(folder,job)['record']['probe_val']

def test_early_failure_caps_previously_checkpointed_probe(tmp_path,small):
    cfg,job,data=small
    cfg=dict(cfg,probe_steps=4,checkpoint_every=1,validation_every=1)
    folder=tmp_path/'failure'
    engine.train_to(folder,job,data,cfg,'protocol','cpu',1)
    with mock.patch('torch.nn.utils.clip_grad_norm_',return_value=torch.tensor(float('nan'))):
        result=engine.train_to(folder,job,data,cfg,'protocol','cpu',cfg['probe_steps'])
    assert result['numerical_failure']['step']==2
    assert result['probe_val']==cfg['loss_cap']
    assert result['final_val']==cfg['loss_cap']
    assert result['remaining_seconds']==0

def test_test_data_requires_frozen_decisions(tmp_path,small):
    cfg,job,data=small
    with pytest.raises(RuntimeError,match='locked'):
        get_data(tmp_path,'gray_scott',cfg,'unit-test-source','test')

def test_changed_protocol_refuses_resume(tmp_path,small):
    cfg,job,data=small
    engine.train_to(tmp_path,job,data,cfg,'protocol','cpu',cfg['probe_steps'])
    with pytest.raises(ValueError,match='mismatch'):
        engine.train_to(tmp_path,job,data,cfg,'changed','cpu',cfg['steps'])
