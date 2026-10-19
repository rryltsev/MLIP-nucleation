"""Integration invariants for the nonmagnetic time-reversal-only adapter."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kplib_grid import reduce_time_reversal,kpoints


class TimeReversalQuadrature(unittest.TestCase):
    def test_shifted_mesh_preserves_even_fourier_integrals(self):
        full=np.array([[(i+.5)/4,(j+.5)/3,(k+.5)/2] for i in range(4) for j in range(3) for k in range(2)])
        reduced,w=reduce_time_reversal(full)
        for vector in ([1,2,3],[4,-2,1],[0,0,0]):
            before=np.cos(2*np.pi*full.dot(vector)).mean()
            after=np.average(np.cos(2*np.pi*np.array(reduced).dot(vector)),weights=w)
            self.assertAlmostEqual(before,after,places=13)
        content=kpoints(reduced,w,'test').splitlines()
        self.assertEqual(content[2],'Reciprocal')
        self.assertEqual(sum(int(l.split()[3]) for l in content[3:]),len(full))

    def test_self_inverse_points_are_not_double_counted(self):
        full=np.array([[i/2,j/2,k/2] for i in range(2) for j in range(2) for k in range(2)])
        reduced,w=reduce_time_reversal(full)
        self.assertEqual(w,[1]*8)
        self.assertEqual(len(reduced),8)

    def test_nonclosed_grid_rejected(self):
        with self.assertRaisesRegex(AssertionError,'not closed'):
            reduce_time_reversal([[.1,0,0],[.2,0,0]])

    def test_periodic_duplicates_rejected(self):
        with self.assertRaisesRegex(AssertionError,'Duplicate'):
            reduce_time_reversal([[0,0,0],[1,0,0]])


if __name__=='__main__':unittest.main()
