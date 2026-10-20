import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import finetune_cluster


class LoRAReservations(unittest.TestCase):
    def test_pending_lora_counts_full_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'runs/fm-al32-lora-20261020-01/manifest.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(job_id='123',resources=dict(walltime='00:15:00',ntasks=1,cpus_per_task=4))))
            with patch.object(finetune_cluster,'ROOT',root),patch.object(finetune_cluster,'remote',return_value='123|PENDING|0\n123|fm-al32-lora-20261020-01|PENDING\n'):
                result=finetune_cluster.budget({'budget':{'maximum_CPU_seconds':3600}})
            self.assertEqual(result['prior_charged_CPU_seconds'],3600)
            self.assertEqual(result['worst_case_CPU_hours'],2)

    def test_unrecorded_related_job_blocks_submission(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(finetune_cluster,'ROOT',Path(d)),patch.object(finetune_cluster,'remote',return_value='456|fm-al32-lora-unrecorded|RUNNING\n'):
                with self.assertRaisesRegex(AssertionError,'Unrecorded related job'):
                    finetune_cluster.budget({'budget':{'maximum_CPU_seconds':3600}})


if __name__=='__main__':unittest.main()
