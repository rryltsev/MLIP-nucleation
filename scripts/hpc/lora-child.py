"""Observe native LoRA training; gate zero-step, frozen weights and live export."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
from deepmd.pt.train.training import Trainer
from deepmd.pt.model.descriptor.sezm_nn.lora import has_lora

ROOT=Path(__file__).resolve().parent;WORK=Path.cwd()
spec=importlib.util.spec_from_file_location('lora_observer',ROOT/'al32-lora-run.py')
observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
base=observer.base
original_init=Trainer.__init__;original_save=Trainer.save_model
inventory={};initial={};holders=[]


def tensor_sha(p):return hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def adapter_name(name):return any(x.startswith(('A_','B_')) for x in name.split('.'))


def frozen_check(model):
    changed=[n for n,p in model.named_parameters() if not inventory[n]['trainable'] and tensor_sha(p)!=initial[n]]
    assert not changed,('Frozen parameters changed',changed)
    return changed


def observed_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs);holders.append(self)
    assert self._lora_enabled
    model=self.wrapper.model['Default'];assert has_lora(model)
    for name,p in model.named_parameters():
        initial[name]=tensor_sha(p);inventory[name]=dict(trainable=p.requires_grad,numel=p.numel(),shape=list(p.shape),initial_sha256=initial[name],adapter=adapter_name(name))
    assert any(v['adapter'] and v['trainable'] for v in inventory.values())
    assert any(not v['trainable'] for v in inventory.values())
    b=[p for n,p in model.named_parameters() if any(x.startswith('B_') for x in n.split('.'))]
    assert b and all(torch.count_nonzero(p).item()==0 for p in b)
    base.save(WORK/'parameter-inventory.json',dict(parameters=inventory,total=sum(v['numel'] for v in inventory.values()),trainable=sum(v['numel'] for v in inventory.values() if v['trainable']),frozen=sum(v['numel'] for v in inventory.values() if not v['trainable'])))
    (WORK/'merged').mkdir();(WORK/'export-checks').mkdir()
    torch.save({'model':self.wrapper.state_dict()},WORK/'initial-native.pt')
    self.save_model_merged(WORK/'initial-merged.pt')
    rows,atoms,labels=base.dataset(final=False);source,_=base.original('dpa4')
    original=base.predict('dpa4',source,None,rows,atoms)
    merged=base.predict('dpa4',str(WORK/'initial-merged.pt'),None,rows,atoms)
    live=observer.live_predict(model,rows,atoms)
    restore=observer.differences(original,merged,rows,allow_constant=True);observer.check_difference(restore)
    export=observer.differences(live,merged,rows);observer.check_difference(export)
    base.save(WORK/'initial-predictions.json',dict(original=original,merged=merged,live=live))
    base.save(WORK/'zero-step.json',dict(status='passed',all_adapter_B_zero=True,original_vs_initialized=restore,live_vs_merged=export,test_labels_opened=False,initial_merged_sha256=base.sha(WORK/'initial-merged.pt')))
    frozen_check(model)


def observed_save(self,save_path,lr=0.,step=0,**kwargs):
    original_save(self,save_path,lr=lr,step=step,**kwargs)
    progress=step+1;model=self.wrapper.model['Default'];frozen_check(model)
    path=WORK/f'merged/{progress:04d}.pt';self.save_model_merged(path,lr=lr,step=step)
    rows,atoms,labels=base.dataset(final=False)
    live=observer.live_predict(model,rows,atoms)
    merged=base.predict('dpa4',str(path),None,rows,atoms)
    diff=observer.differences(live,merged,rows);observer.check_difference(diff)
    state=torch.load(path,map_location='cpu',weights_only=False)['model']
    assert not any(adapter_name(n) or 'lora_scaling' in n for n in state)
    base.save(WORK/f'export-checks/{progress:04d}.json',dict(status='passed',progress=progress,native_checkpoint=str(save_path),native_sha256=base.sha(save_path),merged_sha256=base.sha(path),differences=diff,live_predictions=live,merged_predictions=merged))


Trainer.__init__=observed_init;Trainer.save_model=observed_save
from deepmd.main import main
try:main()
except SystemExit as e:
    if e.code not in (None,0):raise
assert len(holders)==1
model=holders[0].wrapper.model['Default'];frozen_check(model)
changes=[n for n,p in model.named_parameters() if tensor_sha(p)!=initial[n]]
adapters=[n for n in changes if inventory[n]['adapter']]
assert adapters,'No adapter parameters changed'
base.save(WORK/'parameter-update-check.json',dict(status='passed',frozen_changed=[],updated_adapters=adapters,updated_original_parameters=[n for n in changes if not inventory[n]['adapter']],final_sha256={n:tensor_sha(p) for n,p in model.named_parameters()}))
