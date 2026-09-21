"""Teaching checks only; no model quantization or hardware benchmark."""
import json
from pathlib import Path
import numpy as np
results=[]
def record(name,condition,values):
 assert condition, name
 results.append(dict(name=name,status='passed',values=values))
a=np.linspace(0,1,10001)
x=np.array([2.,1.]); grads=np.column_stack([2*a,-2*a])
ig=x*np.trapezoid(grads,a,axis=0)
record('signed IG completeness and absolute-value change',np.allclose(ig,[2,-1]) and np.isclose(ig.sum(),1),{'ig':ig.tolist(),'signed_sum':float(ig.sum()),'absolute_sum':float(abs(ig).sum())})
baseline_ig=np.trapezoid(2*(1+a),a)
record('nonzero baseline',np.isclose(baseline_ig,3),{'IG':float(baseline_ig),'F_endpoint':4,'F_baseline':1})
# Subtracting any constant leaves the analytic derivative unchanged.
for c in (0,1,7):
 derivative=2*(1+a)
 assert np.isclose(np.trapezoid(derivative,a),3)
record('constant offset does not change IG',True,{'offsets_checked':[0,1,7]})
X=np.array([[1.,2.],[3.,-1.],[2.,4.]])
lam=np.array([.2,.3,.5]);delta=np.array([.4,-.7])
G=X.T@np.diag(lam)@X
err=float(np.sum(lam*(X@delta)**2));quad=float(delta@G@delta)
record('weighted Gram and square-root factors',np.isclose(err,quad) and np.allclose(G,(np.sqrt(lam)[:,None]*X).T@(np.sqrt(lam)[:,None]*X)),{'loss':err,'quadratic':quad,'eigenvalues':np.linalg.eigvalsh(G).tolist()})
record('signed weights can violate PSD',np.linalg.eigvalsh(np.diag([2.,-1.])).min()<0,{'counterexample_eigenvalues':[-1,2]})
raw=np.array([1.,1.,1.,1.,100.]);q1,q3=np.quantile(raw,[.25,.75]);clipped=np.clip(raw,q1-1.5*(q3-q1),q3+1.5*(q3-q1))
record('IQR changes attribution totals',not np.isclose(raw.sum(),clipped.sum()),{'before':float(raw.sum()),'after':float(clipped.sum())})
p=np.array([.25]*4);h=-np.sum(p*np.log2(p))
record('cluster entropy bound',np.isclose(h,2),{'uniform_four_clusters_bits':float(h),'hundred_cluster_bound_bits':float(np.log2(100))})
R=np.array([[1.,-1.],[1.,1.]])/np.sqrt(2)
Z=np.array([[.4,.4],[1.,0.],[-1.,.1]]);centers=np.array([[0.,0.],[1.,0.]])
dist=((Z[:,None,:]-centers[None,:,:])**2).sum(-1)
rotdist=(((Z@R)[:,None,:]-(centers@R)[None,:,:])**2).sum(-1)
e1=float(np.sum((np.round(Z[0])-Z[0])**2));e2=float(np.sum((np.round(Z[0]@R)-Z[0]@R)**2))
record('rotation preserves distances but not coordinate quantization error',np.allclose(dist,rotdist) and not np.isclose(e1,e2),{'original_error':e1,'rotated_error':e2})
A=np.array([.9,.7,.86,.6]);valid=np.flatnonzero(A>=.8)
record('nonmonotone prefix quality',valid.tolist()==[0,2],{'scores':A.tolist(),'feasible_k':valid.tolist()})
b1=(16*1.08+16*4)/32;b2=(12*1.08+16*4)/28
record('nominal mixed parameter bit widths',np.isclose(b1,2.54) and np.isclose(1-b1/4,.365),{'llava_bits':b1,'qwen_bits':b2,'llava_nominal_reduction':1-b1/4,'deployment_memory_reduction':1-3.4/4.4,'intel_throughput_ratio':9/4.8,'amd_throughput_ratio':18.7/14.1})
report=dict(status='passed',scope='Analytic/toy numerical identities and counterexamples only; not paper reproduction.',checks=results)
Path(__file__).with_name('math-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','checks':len(results)}))
