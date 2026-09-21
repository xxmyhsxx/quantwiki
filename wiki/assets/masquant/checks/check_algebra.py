"""Teaching identities and counterexamples only; no model/quantization reproduction."""
from pathlib import Path
import numpy as np
import json, hashlib
rng=np.random.default_rng(20260915)
checks={}
def eq(name,x,y):
    err=float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
    assert np.allclose(x,y,rtol=1e-9,atol=1e-9),(name,err)
    checks[name]={'max_abs_error':err}
A=rng.normal(size=(20,6));D=rng.normal(size=(6,5));r=2
lam,P=np.linalg.eigh(A.T@A);T=np.diag(np.sqrt(lam))@P.T
U,s,Vh=np.linalg.svd(T@D,full_matrices=False)
L=np.linalg.solve(T,U[:,:r])@np.diag(s[:r])@Vh[:r]
eq('weighted_svd_tail',np.linalg.norm(A@(D-L))**2,np.sum(s[r:]**2))
eq('whitening',np.linalg.solve(T.T,A.T)@np.linalg.solve(T.T,A.T).T,np.eye(6))
assert np.linalg.matrix_rank(D)==np.linalg.matrix_rank(T@D)
checks['invertible_left_product_preserves_rank']={'rank':int(np.linalg.matrix_rank(D))}
Ae=np.diag([.1,10]);De=np.diag([2.,1.])
eq('ordinary_svd_example_error',np.linalg.norm(Ae@(De-np.diag([2.,0])))**2,100)
eq('weighted_svd_example_error',np.linalg.norm(Ae@(De-np.diag([0.,1.])))**2,.04)
# Fake quant is solely a toy perturbation generator, not an author quantizer.
def q(x):return np.round(x*3)/3
B=rng.normal(size=(6,5));Qa=q(A)
eq('mas_weight_activation_residual',A@(B+D)-(Qa@B+A@L),A@(D-L)+(A-Qa)@B)
X=rng.normal(size=(7,6));W=rng.normal(size=(6,4));parts=[[0,2,4],[1],[3,5]]
eq('all_tokens_all_channel_groups',X@W,sum(X[:,c]@W[c,:] for c in parts))
# CWS must subtract exactly what its separate branch represents before quantization.
Us=rng.normal(size=(6,2));Vs=rng.normal(size=(2,5))
eq('cws_float_identity',A@B,A@(B-Us@Vs)+(A@Us)@Vs)
Bhat=q(B-Us@Vs)+q(Us)@q(Vs);C=q(rng.normal(size=(6,2)))@q(rng.normal(size=(2,5)))
Da=A-Qa;Yhat=Qa@Bhat+q(Da)@C
eq('splitq_three_residual_terms',Yhat-A@B,Qa@(Bhat-B)+Da@(C-B)+(q(Da)-Da)@C)
for rho,expected in [(np.array([10.,10.]),1.),(np.array([1.,10.]),.505)]:
 eq('sqnr_proxy_'+str(rho.tolist()),sum(rho**-2)/(len(rho)*max(rho**-2)),expected)
eq('aux_weight_storage_ratio',.02*2*16/4,.16)
root=Path(__file__).resolve().parents[3]
result={'status':'passed','scope':__doc__,'checks':checks}
Path(__file__).with_name('algebra-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
