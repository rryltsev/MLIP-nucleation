"""Scientific label-contract tests; fixture numbers are NOT DFT observations.

Run with ASE 3.29 + NumPy: python -m unittest discover -s tests -v
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
from ase import units
from ase.build import bulk
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("al_pilot_hpc", ROOT/"scripts/hpc/al-pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
sys.path.insert(0,str(ROOT/'scripts'))
from al_pilot import budget_snapshot


def xml_fixture():
    structure = '<structure><crystal><varray name="basis"><v>4 0 0</v><v>0 4 0</v><v>0 0 4</v></varray></crystal><varray name="positions"><v>0 0 0</v><v>0.5 0.5 0.5</v></varray></structure>'
    energy = '<energy><i name="e_fr_energy">-7.0</i><i name="e_0_energy">-6.9</i></energy>'
    return ('<modeling><kpoints><varray name="kpointlist"><v>0 0 0</v></varray><varray name="weights"><v>1</v></varray></kpoints>'
      '<parameters><i name="NSW" type="int">0</i><i name="NELM" type="int">10</i></parameters>'
      '<atominfo><array name="atoms"><set><rc><c>Al</c></rc><rc><c>Al</c></rc></set></array></atominfo>'
      + structure.replace('<structure>', '<structure name="initialpos">')
      + '<calculation><scstep>'+energy+'</scstep>'+energy+structure
      + '<varray name="forces"><v>1 -2 3</v><v>-1 2 -3</v></varray>'
      '<varray name="stress"><v>1 6 5</v><v>6 2 4</v><v>5 4 3</v></varray>'
      '</calculation></modeling>')


class LabelContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for n in ('POSCAR','INCAR','KPOINTS','POTCAR'):
            (self.root/n).write_text('test fixture, not licensed content')
        (self.root/'vasprun.xml').write_text(xml_fixture())
        (self.root/'OUTCAR').write_text('aborting loop because EDIFF is reached\nGeneral timing and accounting informations for this job\n')

    def tearDown(self):
        self.temp.cleanup()

    def test_free_energy_and_tensile_stress(self):
        label = pilot.parse_label(self.root, expected_atoms=2)
        self.assertEqual(label['energy_free_eV'], -7.0)
        self.assertAlmostEqual(label['energy_sigma0_eV'], -6.9)
        # One kbar = 1e8 Pa = 1e8 / 1.602176634e11 eV/A^3.
        expected = -np.arange(1,7)*1e8/1.602176634e11
        np.testing.assert_allclose(label['stress_eV_per_A3'], expected, rtol=1e-7)

    def test_reject_truncated_xml(self):
        (self.root/'vasprun.xml').write_text(xml_fixture()[:-15])
        with self.assertRaises(Exception): pilot.parse_label(self.root, 2)

    def test_reject_unconverged(self):
        (self.root/'OUTCAR').write_text('General timing and accounting informations for this job')
        with self.assertRaises(ValueError): pilot.parse_label(self.root, 2)

    def test_reject_iteration_limit_even_if_marker_present(self):
        (self.root/'vasprun.xml').write_text(xml_fixture().replace('>10</i>','>1</i>'))
        with self.assertRaises(ValueError): pilot.parse_label(self.root, 2)

    def test_export_preserves_total_energy_stress_and_virial(self):
        cfg = json.loads((ROOT/'configs/systems/al32-pilot-v1.json').read_text())
        a = bulk('Al','fcc',a=4.05,cubic=True).repeat((2,2,2))
        rows, labels = [], []
        for split in ('train','validation','test'):
            p = self.root/'structures'/split/'POSCAR'
            p.parent.mkdir(parents=True)
            write(p,a,format='vasp')
            rows.append(dict(configuration_id=split,parent_id=split,split=split,poscar=str(p.relative_to(self.root))))
            labels.append(dict(configuration_id=split,split=split,energy_free_eV=-100.,
              forces_eV_per_A=np.full((32,3),0.125).tolist(),stress_eV_per_A3=[1.,2.,3.,4.,5.,6.]))
        pilot.save(self.root/'labels.json',labels)
        pilot.save(self.root/'dataset-manifest.json',rows)
        pilot.export(self.root,cfg,rows,labels)
        b = read(self.root/'export/train.extxyz')
        self.assertEqual(b.get_potential_energy(),-100.)
        w = np.load(self.root/'export/deepmd/train/set.000/virial.npy')[0].reshape(3,3)
        np.testing.assert_allclose(w/-a.get_volume(),[[1,6,5],[6,2,4],[5,4,3]])


class ConditionalLabelGate(unittest.TestCase):
    def test_failed_calibration_cannot_start_labeling(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source'; target=Path(tmp)/'target'
            source.mkdir(); target.mkdir()
            pilot.save(source/'calibration.json',{'passed':False})
            with self.assertRaisesRegex(ValueError,'Unvalidated protocol'):
                pilot.label_all(target,{},source)


class DenseCalibrationAndBudget(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'configs/dft/al32-kmesh-v2.json').read_text())

    def label(self, force=0.0):
        return dict(energy_free_eV=0.0,forces_eV_per_A=[[force,0,0]]*32,stress_eV_per_A3=[0.0]*6)

    def labels(self):
        return {cid:{name:self.label() for name,_,_,_ in self.cfg['calibration_cases']}
                for cid in self.cfg['calibration_configuration_ids']}

    def test_cheapest_candidate_must_pass_on_both_parents(self):
        labels=self.labels()
        labels['md-00-f1']['candidate6']=self.label(0.021)
        assessment=pilot.evaluate_calibration(self.cfg,labels)
        self.assertTrue(assessment['passed'])
        self.assertEqual(assessment['protocol']['gamma_mesh'],7)

    def test_reference_instability_blocks_all_candidates(self):
        labels=self.labels()
        labels['fcc-04']['mesh7']=self.label(0.021)
        assessment=pilot.evaluate_calibration(self.cfg,labels)
        self.assertFalse(assessment['passed'])
        self.assertIsNone(assessment['protocol'])

    def test_gamma6_selected_when_all_comparisons_pass(self):
        assessment=pilot.evaluate_calibration(self.cfg,self.labels())
        self.assertTrue(assessment['passed'])
        self.assertEqual(assessment['protocol']['gamma_mesh'],6)

    def test_global_budget_counts_prior_actual_and_pending_reservations(self):
        manifests=[dict(run_id='oldcal',stage='collected',job_id='1'),
                   dict(run_id='oldlabel',stage='collected',job_id='2'),
                   dict(run_id='newcal',stage='submitted',job_id='3',resources=dict(ntasks=16,walltime='02:00:00'))]
        accounting={'1':['1','FAILED','1:0','2353','16','37648'],
                    '2':['2','CANCELLED','0:0','0','0','0'],
                    '3':['3','PENDING','0:0','0','16','0']}
        request=dict(resources=dict(ntasks=16,walltime='05:00:00'))
        result=budget_snapshot(manifests,accounting,request,128)
        self.assertEqual(result['worst_case_total_CPU_seconds'],440848)
        request['resources']['walltime']='06:00:00'
        with self.assertRaisesRegex(ValueError,'exceeded'):
            budget_snapshot(manifests,accounting,request,128)

    def test_unknown_prior_submission_blocks_budget(self):
        request=dict(resources=dict(ntasks=16,walltime='02:00:00'))
        with self.assertRaisesRegex(ValueError,'Uncertain'):
            budget_snapshot([dict(run_id='ambiguous',stage='submission_intent',job_id=None)],{},request,128)

    def test_changed_configuration_cannot_start_labeling(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source'; target=Path(tmp)/'target'
            source.mkdir(); target.mkdir()
            pilot.save(source/'dataset-manifest.json',{'configurations':[]})
            pilot.save(source/'calibration.json',{'passed':True,'dataset_manifest_sha256':pilot.digest(source/'dataset-manifest.json')})
            (source/'system.json').write_text('original')
            (target/'system.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'configuration differ'):
                pilot.label_all(target,{},source)


if __name__ == '__main__':
    unittest.main()
