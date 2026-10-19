import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from finetune_selection import candidate_metrics,choose


class SelectionIsolation(unittest.TestCase):
    def test_validation_not_training_error_controls_selection(self):
        rows=[dict(configuration_id=c,parent_id=c,split=s) for c,s in [('a','train'),('b','validation')]]
        labels={c:dict(energy_free_eV=0.,forces_eV_per_A=[[0,0,0]],stress_eV_per_A3=[0]*6) for c in ('a','b')}
        def preds(train_force,val_force):
            return {c:dict(energy_eV=0.,forces_eV_per_A=[[f,0,0]],stress_eV_per_A3=[0]*6) for c,f in [('a',train_force),('b',val_force)]}
        scales=dict(energy_RMSE_scale_eV_per_atom=.01,force_RMSE_scale_eV_per_A=.05,stress_RMSE_scale_eV_per_A3=.001)
        train_best=dict(progress=100,**candidate_metrics(rows,preds(0.,1.),labels,scales))
        val_best=dict(progress=200,**candidate_metrics(rows,preds(100.,.1),labels,scales))
        self.assertIs(choose([train_best,val_best]),val_best)

    def test_rejects_test_rows_even_if_predictions_are_available(self):
        with self.assertRaisesRegex(AssertionError,'train/validation only'):
            candidate_metrics([dict(split='train'),dict(split='validation'),dict(split='test')],{}, {},{})

    def test_original_can_win_and_ties_prefer_smaller_progress(self):
        original=dict(score=1.,progress=0)
        trained=dict(score=2.,progress=100)
        self.assertIs(choose([trained,original]),original)
        self.assertIs(choose([dict(score=1.,progress=200),original]),original)


if __name__=='__main__':unittest.main()
