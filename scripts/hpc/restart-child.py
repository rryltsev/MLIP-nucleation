"""Observe fully restored training state before the next optimizer update."""
import json
import os
from pathlib import Path
import sys

backend=sys.argv.pop(1)
snapshot=Path(os.environ['RESTART_SNAPSHOT']);snapshot.parent.mkdir(parents=True,exist_ok=True)
if backend=='dpa4':
    import torch
    from deepmd.pt.train.training import Trainer
    original=Trainer.__init__
    def observed(self,*args,**kwargs):
        original(self,*args,**kwargs)
        raw_step=self.start_step
        completed={int(v['step']) for v in self.optimizer.state_dict()['state'].values() if 'step' in v}
        assert len(completed)==1, 'Ambiguous Adam update counters'
        completed=completed.pop()
        assert completed in (raw_step,raw_step+1), (raw_step,completed)
        # DeepMD 3.2.0 stores a zero-based completed-loop index but uses it
        # directly as range(start, stop) on restart. Continue AFTER that update.
        if completed==raw_step+1:
            self.start_step=completed
            self.scheduler=self._create_lr_scheduler(self.optimizer,self.lr_schedule,self.start_step)
        torch.save({'model':self.wrapper.state_dict(),'optimizer':self.optimizer.state_dict()},snapshot)
        snapshot.with_suffix('.json').write_text(json.dumps({'step':self.start_step,'raw_checkpoint_step':raw_step,
            'optimizer_iterations':completed,'compatibility_step_adjustment':self.start_step-raw_step})+'\n')
    Trainer.__init__=observed
    from deepmd.main import main
else:
    from tensorpotential.tensorpot import TensorPotential
    original=TensorPotential.load_checkpoint
    def observed(self,*args,**kwargs):
        original(self,*args,**kwargs)
        if not Path(str(snapshot)+'.index').exists():
            self.model.get_flat_trainable_variables()
            self.optimizer.build(self.model.variables_to_train)
            self.checkpoint.write(str(snapshot))
            snapshot.with_suffix('.json').write_text(json.dumps({'step':self.step,'epoch':self.epoch,
                'optimizer_iterations':int(self.optimizer.iterations.numpy())})+'\n')
    TensorPotential.load_checkpoint=observed
    from tensorpotential.scripts.gracemaker import main
sys.exit(main())
