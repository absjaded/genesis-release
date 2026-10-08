"""Public-input safety and valid-input parity with unchanged historical code."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from genesis_core import analysis as g


def original():
    spec = importlib.util.spec_from_file_location('genesis_legacy_estimators', ROOT/'tests/originals/gate1_estimators.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = original()
        rng = np.random.default_rng(19)
        cls.x = rng.normal(size=(72,8))
        cls.y = cls.x @ rng.normal(size=(8,10)) + .01*rng.normal(size=(72,10))
        cls.labels = np.repeat(np.arange(3),24)
        cls.train, cls.test = g.stratified_splits(cls.labels,3,51)[0]

    def test_valid_geometry_parity(self):
        dx, dy = g.cosine_distance(g.l2_rows(self.x)), g.cosine_distance(g.l2_rows(self.y))
        self.assertEqual(g.partial_rsa(dx,dy,self.labels), self.old.partial_rsa(dx,dy,self.labels))
        self.assertEqual(g.neighborhood_overlap(dx,dy,5), self.old.neighborhood_overlap(dx,dy,5))
        self.assertEqual(g.spectral_metrics(self.x), self.old.spectral_metrics(self.x))

    def test_ridge_and_retrieval_parity(self):
        new, gallery = g.ridge_prediction(self.x,self.y,self.train,self.test,.1)
        old, old_gallery = self.old.ridge_prediction(self.x,self.y,self.train,self.test,.1)
        np.testing.assert_array_equal(new,old)
        np.testing.assert_array_equal(gallery,old_gallery)
        args=(new,gallery,self.test,self.test,self.labels)
        self.assertEqual(g.retrieval_metrics(*args,candidate_count=16,seed=33),
                         self.old.retrieval_metrics(*args,candidate_count=16,seed=33))

    def test_pls_prediction_parity(self):
        a,b=g.pls_prediction(self.x,self.y,self.train,self.test,2)
        c,d=self.old.pls_prediction(self.x,self.y,self.train,self.test,2)
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(b,d)

    def test_no_heldout_target_fit_leakage(self):
        a,_=g.ridge_prediction(self.x,self.y,self.train,self.test,.1)
        changed=self.y.copy();changed[self.test]+=1e4
        b,_=g.ridge_prediction(self.x,changed,self.train,self.test,.1)
        np.testing.assert_array_equal(a,b)

    def test_duplicate_neighbor_self_exclusion(self):
        x=np.array([[1.,0],[1,0],[0,1],[0,-1]])
        dx,dy=g.cosine_distance(x),g.cosine_distance(x[[0,2,1,3]])
        self.assertEqual(self.old.neighborhood_overlap(dx,dy,1),.25)
        self.assertEqual(g.neighborhood_overlap(dx,dy,1),.75)

    def test_nonfinite_p_rejected(self):
        for value in (np.nan,np.inf,-np.inf):
            with self.subTest(value=value),self.assertRaises(ValueError):g.fwer_p(value,[0,1,2])
        for null in ([],[np.nan],[np.inf]):
            with self.assertRaises(ValueError):g.fwer_p(1,null)
        self.assertEqual(g.fwer_p(2,[0,1,2]),.5)

    def test_categories_not_silently_truncated(self):
        for labels in ([0.2,0.9,1.2,1.9],[False,False,True,True],['a','a','b','b']):
            with self.assertRaises(ValueError):g.require_categories(labels,4)

    def test_nonfinite_complex_and_zero_variance(self):
        for x in (np.full((4,3),np.nan),np.ones((4,3),dtype=complex)):
            with self.assertRaises(ValueError):g.require_matrix(x,'x')
        with self.assertRaises(ValueError):g.spectral_metrics(np.ones((8,3)))
        with self.assertRaises(ValueError):g.fit_standardize(np.ones((8,3)),np.ones((4,3)))

    def test_degenerate_rsa_rejected(self):
        z=np.zeros((72,72))
        with self.assertRaises(ValueError):g.partial_rsa(z,z,self.labels)
        category=(self.labels[:,None]!=self.labels[None,:]).astype(float)
        with self.assertRaises(ValueError):g.partial_rsa(category,category,self.labels)

    def test_split_and_gallery_guards(self):
        for train in (np.array([0,0,1]),np.array([-1,0,1]),np.array([0.,1.,2.]),self.test):
            with self.assertRaises(ValueError):g.ridge_prediction(self.x,self.y,train,self.test,.1)
        prediction,gallery=g.ridge_prediction(self.x,self.y,self.train,self.test,.1)
        with self.assertRaises(ValueError):
            g.retrieval_metrics(prediction,gallery,self.test,self.test,self.labels,candidate_count=16,seed=1,gallery_indices=np.repeat(np.arange(72),2))

    def test_regularization_and_component_guards(self):
        for alpha in (0,-1,np.nan,np.inf):
            with self.assertRaises(ValueError):g.ridge_prediction(self.x,self.y,self.train,self.test,alpha)
        with self.assertRaises(ValueError):g.pls_prediction(self.x,self.y,self.train,self.test,2.5)

    def test_reconstruction_parity_and_invalid_folds(self):
        splits=g.stratified_splits(self.labels,3,9)
        self.assertEqual(g.bidirectional_reconstruction(self.x,self.y,splits),
                         self.old.bidirectional_reconstruction(self.x,self.y,splits))
        with self.assertRaises(ValueError):g.bidirectional_reconstruction(self.x,np.ones_like(self.y),splits)
        bad=[(np.arange(72),test) for _,test in splits]
        with self.assertRaises(ValueError):g.crossfit_center_l2(self.x,bad)
        with self.assertRaises(ValueError):g.bidirectional_reconstruction(self.x,self.y,bad)

    def test_retrieval_ties_follow_conservative_rule(self):
        prediction,gallery=g.ridge_prediction(self.x,self.y,self.train,self.test,.1)
        gallery=gallery.copy();gallery[1]=gallery[0]
        args=(prediction,gallery,self.test,self.test,self.labels)
        self.assertEqual(g.retrieval_metrics(*args,candidate_count=16,seed=12),
                         self.old.retrieval_metrics(*args,candidate_count=16,seed=12))


if __name__=='__main__':unittest.main()
