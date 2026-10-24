"""Meaningful numerical checks against independent dynamic-programming references."""
import unittest
import numpy as np
from distances import dtw, twed
from run_experiment import make_variant, hac
from ctds import embedding


def reference(x, tx, y, ty, use_twed):
    a = np.full((len(x) + 1, len(y) + 1), np.inf); a[0, 0] = 0
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if not use_twed:
                a[i, j] = (x[i-1]-y[j-1])**2 + min(a[i-1,j],a[i,j-1],a[i-1,j-1])
            else:
                xp, yp = (x[i-2] if i > 1 else 0), (y[j-2] if j > 1 else 0)
                tp, sp = (tx[i-2] if i > 1 else 0), (ty[j-2] if j > 1 else 0)
                delete = a[i-1,j] + abs(x[i-1]-xp) + .1*(tx[i-1]-tp) + .01
                insert = a[i,j-1] + abs(y[j-1]-yp) + .1*(ty[j-1]-sp) + .01
                match = a[i-1,j-1] + abs(x[i-1]-y[j-1])+abs(xp-yp)+.1*(abs(tx[i-1]-ty[j-1])+abs(tp-sp))
                a[i,j] = min(delete, insert, match)
    return a[-1,-1] if use_twed else np.sqrt(a[-1,-1])


class Checks(unittest.TestCase):
    def test_distances(self):
        rng=np.random.default_rng(11)
        for n,m in [(2,3),(7,4),(5,5)]:
            x,y=rng.normal(size=n),rng.normal(size=m)
            tx,ty=np.cumsum(rng.uniform(.01,.2,n)),np.cumsum(rng.uniform(.01,.2,m))
            for func,is_twed in [(dtw,False),(twed,True)]:
                self.assertAlmostEqual(func(x,tx,y,ty),reference(x,tx,y,ty,is_twed),places=10)
                self.assertAlmostEqual(func(x,tx,y,ty),func(y,ty,x,tx),places=10)
                self.assertAlmostEqual(func(x,tx,x,tx),0,places=10)

    def test_timestamp_sensitivity(self):
        x=np.array([0.,1.,0.,-1.]); t=np.arange(4.)/50
        self.assertEqual(dtw(x,t,x,t*2),0)
        self.assertGreater(twed(x,t,x,t*2),0)

    def test_sampling_grid(self):
        t=np.arange(128)/50
        x=np.tile(np.sin(2*np.pi*2*t),(20,1))
        for name,factor in [('50hz',1),('25hz',2),('12p5hz',4)]:
            values,times,lengths,common,rates=make_variant(x,name,42)
            self.assertTrue(np.all(lengths==128//factor))
            np.testing.assert_allclose(np.diff(times[0,:lengths[0]]),factor/50)
            self.assertEqual(common.shape,(20,128))
            if factor==1: np.testing.assert_array_equal(common,x)
        first=make_variant(x,'mixed',42)
        second=make_variant(x,'mixed',42)
        np.testing.assert_array_equal(first[-1],second[-1])

    def test_ctds(self):
        x=np.random.default_rng(2).normal(size=(10,128))
        a=embedding(x,23,psi=8,repetitions=3)
        b=embedding(x,23,psi=8,repetitions=3)
        self.assertEqual(a.shape,(10,48))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(np.linalg.norm(a,axis=1),np.sqrt(2))

    def test_hierarchy_partition(self):
        d=np.array([[0,1,9,10],[1,0,8,9],[9,8,0,1],[10,9,1,0]],dtype=float)
        labels=hac(d,2)
        self.assertEqual(labels[0],labels[1]); self.assertEqual(labels[2],labels[3])
        self.assertNotEqual(labels[0],labels[2])


if __name__=='__main__': unittest.main(verbosity=2)
