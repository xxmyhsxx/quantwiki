"""Teaching checks only: no paper-model experiment or released-kernel test."""
import json
from pathlib import Path
import numpy as np
rng=np.random.default_rng(20260915)
def ortho(n):return np.linalg.qr(rng.normal(size=(n,n)))[0]
def had(n):
 h=np.ones((1,1))
 while len(h)<n:h=np.block([[h,h],[h,-h]])
 return h/np.sqrt(n)
def norm(x):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
def silu(x):return x/(1+np.exp(-x))
def err(a,b):return float(np.max(np.abs(a-b)))
t,d,f,h,dh=5,8,16,2,4
x=rng.normal(size=(t,d));R=ortho(d);H=had(f)
D=np.diag(rng.normal(size=d))
Wu,Wg,Wd=[rng.normal(size=s) for s in [(f,d),(f,d),(d,f)]]
z=silu(norm(x)@D@Wg.T)*(norm(x)@D@Wu.T)
y=x+z@Wd.T
xr=x@R
zr=silu(norm(xr)@(Wg@D@R).T)*(norm(xr)@(Wu@D@R).T)
yr=xr+(zr@H)@(R.T@Wd@H).T
res={'rms_residual_gated_ffn_max_error':err(yr,y@R)}
head=rng.normal(size=(11,d));res['final_head_max_error']=err(yr@(head@R).T,y@head.T)
# Arbitrary post-RoPE Q/K; mixing heads only AFTER separate softmax.
q,k,v=[rng.normal(size=(h,t,dh)) for _ in range(3)]
Hh,Hd=had(h),had(dh)
logit=q@k.transpose(0,2,1)/np.sqrt(dh)
rotlogit=(q@Hd)@(k@Hd).transpose(0,2,1)/np.sqrt(dh)
res['post_rope_qk_max_error']=err(logit,rotlogit)
s=np.exp(logit-logit.max(-1,keepdims=True));s/=s.sum(-1,keepdims=True)
a=(s@v).transpose(1,0,2).reshape(t,d)
valrot=(s@(v@Hd)).transpose(1,0,2).reshape(t,d)
online=valrot@np.kron(Hh,np.eye(dh))
Wo=rng.normal(size=(d,d));Ho=np.kron(Hh,Hd)
res['quarot_head_factorization_max_error']=err(online,a@Ho)
res['quarot_attention_output_max_error']=err(online@(R.T@Wo@Ho).T,a@Wo.T@R)
R2=ortho(dh);B2=np.kron(np.eye(h),R2)
av=(s@(v@R2)).transpose(1,0,2).reshape(t,d)
res['spinquant_value_output_max_error']=err(av@(R.T@Wo@B2).T,a@Wo.T@R)
# Generic smooth objective for Cayley; not SpinQuant's STE objective.
A=rng.normal(size=(d,d));G=R-A;D=-G
Y=.5*(D@R.T-R@D.T)
eta=min(.1,1/(np.linalg.norm(Y,1)+1e-8));I=np.eye(d)
exact=np.linalg.solve(I-eta*Y/2,(I+eta*Y/2)@R)
res['cayley_exact_orthogonality_max_error']=err(exact.T@exact,I)
res['cayley_smooth_loss_before']=float(.5*np.sum((R-A)**2))
res['cayley_smooth_loss_after']=float(.5*np.sum((exact-A)**2))
Z=R+eta*Y@R
for _ in range(5):Z=R+eta/2*Y@(R+Z)
res['cayley_five_step_vs_exact_max_error']=err(Z,exact)
res['cayley_five_step_orthogonality_max_error']=err(Z.T@Z,I)
res['cayley_contraction_factor_bound']=float(eta*np.linalg.norm(Y,1)/2)
# Counterexample: a fixed Hadamard can raise a peak.
res['hadamard_peak_before']=1.0
res['hadamard_peak_after']=float(np.max(np.abs(np.array([1.,1.])@had(2))))
assert all(v<1e-10 for k,v in res.items() if k.endswith('max_error') and 'five_step' not in k)
assert res['cayley_smooth_loss_after']<res['cayley_smooth_loss_before']
assert res['cayley_five_step_vs_exact_max_error']<1e-3
assert res['hadamard_peak_after']>res['hadamard_peak_before']
out={'status':'passed','scope':'Float64 teaching algebra only; not model quantization, optimizer reproduction, or hardware benchmarking.','seed':20260915,'results':res}
Path(__file__).with_name('algebra-check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
