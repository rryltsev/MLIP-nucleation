import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fm_cluster import collect


class TerminalCollection(unittest.TestCase):
    def test_collects_job_absent_from_live_queue(self):
        result=json.dumps({'status':'passed','cases':{'dpa4':{'final-metrics':{'status':'passed'}}}})
        records={'result.json':{'text':result,'sha256':hashlib.sha256(result.encode()).hexdigest()}}
        transfer=base64.b64encode(gzip.compress(json.dumps(records).encode())).decode()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json'
            p.write_text(json.dumps(dict(job_id='123',remote_run_dir='/example',stage='submitted')))
            with patch('fm_cluster.remote',side_effect=['123|COMPLETED|0:0|00:01:00|60|4|node\n','999|RUNNING|node\n',transfer]) as ssh:
                collect(p)
            self.assertNotIn('squeue -h -j',ssh.call_args_list[1].args[0])
            self.assertEqual(json.loads(p.read_text())['runtime_status'],'passed')
            self.assertEqual((p.parent/'result.json').read_text(),result)


if __name__=='__main__':unittest.main()
