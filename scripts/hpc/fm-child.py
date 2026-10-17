"""Instrument the normal training CLI to preserve weights immediately after loading."""
import os
from pathlib import Path
import sys

backend = sys.argv.pop(1)
loaded = Path(os.environ["FM_LOADED_SNAPSHOT"])
loaded.parent.mkdir(parents=True, exist_ok=True)
if backend == "dpa4":
    import torch
    from deepmd.pt.train.training import Trainer
    original = Trainer.__init__

    def observed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        torch.save(self.wrapper.state_dict(), loaded)

    Trainer.__init__ = observed
    from deepmd.main import main
else:
    import tensorflow as tf
    from tensorpotential.tensorpot import TensorPotential
    original = TensorPotential.load_checkpoint

    def observed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        # Only observe the first load (pretrained initialization, before updates).
        if not Path(str(loaded) + ".index").exists():
            tf.train.Checkpoint(model=self.model).write(str(loaded))

    TensorPotential.load_checkpoint = observed
    from tensorpotential.scripts.gracemaker import main

sys.exit(main())
