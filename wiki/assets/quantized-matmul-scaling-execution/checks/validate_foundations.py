"""Check new teaching claims against direct arithmetic; no model/backend test."""
from pathlib import Path
import json
import hashlib
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
results = []
def checked(name, condition):
    if not condition:
        raise AssertionError(name)
    results.append(name)

def away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

# Independent decoded arithmetic vs centered integer dot and bias units.
qx = np.array([7, 3]); qw = np.array([6, 2])
x = .5*(qx-5); w = .25*(qw-4); bias = .25
acc = int((qx-5) @ (qw-4)) + 2
checked('bias and integer dot recover decoded floating arithmetic', acc == 10 and np.isclose(.125*acc, x@w+bias))
checked('requantization example and saturation', int(8+away(acc/8)) == 9 and np.clip(8+away(80/8),0,15)==15)
checked('negative and positive tie policies differ', away(-1.5)==-2 and away(2.5)==3 and np.rint(2.5)==2)
step=3/7; zero=2
checked('zero alignment moves min-max endpoint', step*(0-zero)>-1 and step*(zero-zero)==0)
# BN folding includes pre-existing bias and a negative gamma.
rng=np.random.default_rng(20260914)
W=rng.normal(size=(3,4)); X=rng.normal(size=(4,8)); b=rng.normal(size=3)
mu=rng.normal(size=3); var=np.array([.2,1.,2.]); gamma=np.array([-2.,.5,1.]); beta=rng.normal(size=3)
a=gamma/np.sqrt(var+1e-5)
ref=a[:,None]*(W@X+b[:,None]-mu[:,None])+beta[:,None]
fold=(a[:,None]*W)@X+(a*(b-mu)+beta)[:,None]
checked('BN fold with existing bias and negative gamma', np.allclose(ref,fold))
W2=rng.normal(size=(2,3)); scales=np.array([.5,2.,3.]); relu=lambda v:np.maximum(v,0)
checked('CLE across ReLU preserves function with bias', np.allclose(W2@relu(W@X+b[:,None]), (W2*scales)@relu((W/scales[:,None])@X+(b/scales)[:,None])))
checked('ReLU6 fixed threshold is counterexample', min(max(8/2,0),6)!=min(max(8,0),6)/2)
E=rng.normal(size=(3,4)); errors=E@X
corrected=errors-(E@X.mean(axis=1))[:,None]
checked('bias correction removes sample mean, preserves variance', np.allclose(corrected.mean(axis=1),0) and np.allclose(corrected.var(axis=1),errors.var(axis=1)) and np.linalg.norm(corrected)>0)
checked('same-grid residual zero-point correction', .25*((9+6-8)-8)==.25*(9-8)+.25*(6-8))
checked('int16 pair bound with narrow weights', 2*128*127 <= 32767 < 2*128*128)
# Explicit demonstration that STE is not the true local derivative.
f=lambda x: np.round(x/.5)*.5
checked('true quantizer local derivative is zero inside a bin', (f(.7+1e-6)-f(.7-1e-6))/(2e-6)==0)
report={'date':'2026-09-14','status':'passed','checks':results,'count':len(results),'scope':'Teaching arithmetic only; no model or backend execution.'}
(OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','checks':len(results)},ensure_ascii=False))
