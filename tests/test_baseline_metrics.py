import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from baseline_metrics import evaluate


class EnergyAlignment(unittest.TestCase):
    def data(self):
        rows=[dict(configuration_id=c,parent_id=c,split=s) for c,s in [('a','train'),('b','train'),('c','test')]]
        ref={c:dict(energy_free_eV=e,forces_eV_per_A=[[0,0,0]]*2,stress_eV_per_A3=[0]*6) for c,e in [('a',2.),('b',4.),('c',6.)]}
        pred={c:dict(energy_eV=r['energy_free_eV']+10,forces_eV_per_A=[[1,0,0]]*2,stress_eV_per_A3=[.1]*6) for c,r in ref.items()}
        return rows,pred,ref

    def test_sign_and_units_and_unchanged_derivatives(self):
        rows,pred,ref=self.data();m=evaluate(rows,pred,ref)
        self.assertEqual(m['delta_Al_eV_per_atom'],-5)
        self.assertEqual(m['metrics_by_split']['test']['energy_aligned_eV_per_atom']['rmse'],0)
        self.assertEqual(m['metrics_by_split']['test']['forces_eV_per_A']['max_absolute'],1)
        self.assertAlmostEqual(m['metrics_by_split']['test']['stress_eV_per_A3']['rmse'],.1)

    def test_test_labels_cannot_change_energy_alignment(self):
        rows,pred,ref=self.data();before=evaluate(rows,pred,ref)
        ref['c']['energy_free_eV']+=1000
        after=evaluate(rows,pred,ref)
        self.assertEqual(before['delta_Al_eV_per_atom'],after['delta_Al_eV_per_atom'])
        self.assertNotEqual(before['metrics_by_split']['test'],after['metrics_by_split']['test'])


if __name__=='__main__':unittest.main()
