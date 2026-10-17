"""Read-only CPU verification of an existing completed GPU run; never retrain."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

root = Path(os.environ['FM_SOURCE_RUN'])
backend = sys.argv[1]
spec = importlib.util.spec_from_file_location('fm_check', Path(__file__).with_name('fm-finetune.py'))
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
result = {'backend':backend,'status':'running','source_run':str(root), 'job_id':os.environ['SLURM_JOB_ID']}
try:
    previous = json.loads((root/backend/'result.json').read_text())
    assert previous['training']['exitcode']==0 and previous['optimizer_updates']==20
    assert previous['training']['gpu_samples']>0
    model = json.loads((root/(backend+'-checkpoint.json')).read_text())
    for v in model['files'].values(): assert helper.digest(Path(v['remote_path']))==v['sha256']
    if backend=='dpa4':
        source = Path(model['remote_root'])/model['filename']
        final = root/'dpa4/model.ckpt-20.pt'
    else:
        source = Path(next(v['remote_path'] for k,v in model['files'].items() if k.endswith('model.yaml'))).parent/'checkpoint'
        final = root/'grace/seed/42/checkpoints/checkpoint'
    result['weight_checks'] = helper.compare(source,root/backend/'loaded-initial/model',final,backend)
    result['optimizer_updates']=20
    result['checkpoint_sha256']={p.name:helper.digest(p) for p in final.parent.glob(final.name+'*') if p.is_file()}
    result['status']='passed'
except Exception:
    result['status']='failed'; result['error']=traceback.format_exc()
helper.save(Path(backend+'-verification.json'),result)
print(json.dumps(result,indent=2),flush=True)
sys.exit(0 if result['status']=='passed' else 1)
