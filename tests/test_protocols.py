"""Protocol parity on fixtures; not historical feature-matrix reproduction."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import unittest
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from genesis_core import readout_matched as c
from genesis_core import readout_discovery as d
from test_analysis import original


def functions(filename,names,ns):
    nodes=[n for n in ast.parse((ROOT/'tests/originals'/filename).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('missing original function')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),ns)
    return ns


class ProtocolTests(unittest.TestCase):
    def test_discovery_orchestration_ridge_and_pls_parity(self):
        old=original()
        ns=dict(np=np,hashlib=hashlib,json=json,Any=Any,oracle_modules=lambda:(old,None))
        names={'canonical_bytes','sha256_bytes','derived_seed','query_top1_records','stratified_bootstrap_top1','nested_readout_result'}
        functions('mp002d_discovery_analysis.py',names,ns)
        rng=np.random.default_rng(34);x=rng.normal(size=(72,8));y=x@rng.normal(size=(8,10))+.1*rng.normal(size=(72,10));labels=np.repeat(np.arange(3),24)
        config=dict(outer_split_seed=56,outer_folds=3,inner_retrieval_candidates=13,outer_retrieval_candidates=16,
                    ridge_alphas=[.01,.1],pls_components=[2,3],permutations_per_mode=2,bootstrap_resamples=12,bootstrap_confidence=.95)
        for family in ('ridge','pls'):
            with self.subTest(family=family):
                actual=d.nested_readout_result(x,y,labels,config,77,family)
                expected=ns['nested_readout_result'](x,y,labels,config,77,family)
                self.assertEqual(actual,expected)
                self.assertEqual(len(actual['query_top1_records']),72)

    def test_matched_readout_full_protocol_parity(self):
        from types import SimpleNamespace
        old=original()
        ns=dict(np=np,spearmanr=spearmanr,l2_rows=old.l2_rows,upper=old.upper,Any=Any)
        functions('mp002c_core.py',{'standardize_train_test','normalize_targets','retrieval_metrics'},ns)
        core=SimpleNamespace(**{k:ns[k] for k in ('standardize_train_test','normalize_targets','retrieval_metrics')},ALPHAS=c.ALPHAS,OUTER_SEED=c.OUTER_SEED)
        ns2=dict(np=np,StratifiedKFold=StratifiedKFold,core=core)
        functions('mp002c_readout.py',{'kernel_design','predict_design','choose','splits','evaluate'},ns2)
        rng=np.random.default_rng(72);x=rng.normal(size=(64,5));targets=[x@rng.normal(size=(5,7))+.1*rng.normal(size=(64,7))];labels=np.repeat(np.arange(2),32)
        actual=c.evaluate('fixture',x,targets,['target'],labels,[0])
        self.assertEqual(actual,ns2['evaluate']('fixture',x,targets,['target'],labels,[0]))

    def test_candidate_rank_ties(self):
        values=np.ones((8,2));ids=np.arange(8);labels=np.repeat([0,1],4)
        records=d.query_top1_records(values,values,ids,labels,candidate_count=4,seed=17)
        self.assertTrue(all(r['rank']==4 and r['top1']==0 for r in records))

    def test_missing_protocol_inputs_rejected(self):
        with self.assertRaises(ValueError):c.evaluate('bad',np.ones((63,4)),[np.ones((63,4))],['a'],np.repeat([0,1,2],21),[0])


if __name__=='__main__':unittest.main()
