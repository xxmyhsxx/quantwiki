"""CPU checks of QoQ packing and algebra; not the upstream converter or CUDA."""
import argparse
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20260922)
n,k = 64,256
q = rng.integers(0,16,(n,k),dtype=np.uint8)
reordered = q.reshape(n//32,2,2,8,k//32,2,4,4).transpose(0,4,3,6,1,5,2,7)
reordered = reordered.transpose(0,1,2,3,5,6,7,4)
packed = (reordered[...,0] | (reordered[...,1]<<4)).reshape(n,k//2)
# Independent coordinate oracle: a byte pairs output channels n and n+16.
oracle = np.empty((n//32,k//32,512),dtype=np.uint8)
decoded = np.empty_like(q)
for nt in range(n//32):
    for kt in range(k//32):
        for nc in range(8):
            for kb in range(4):
                for ka in range(2):
                    for nb in range(2):
                        for kc in range(4):
                            j=((((nc*4+kb)*2+ka)*2+nb)*4+kc)
                            ni=nt*32+8*nb+nc; ki=kt*32+16*ka+4*kb+kc
                            oracle[nt,kt,j]=q[ni,ki] | (q[ni+16,ki]<<4)
                            byte=int(packed.reshape(n//32,k//32,512)[nt,kt,j])
                            decoded[ni,ki]=byte&15; decoded[ni+16,ki]=byte>>4
assert np.array_equal(packed,oracle.reshape(n,k//2))
assert np.array_equal(q,decoded)
logical = np.arange((k//128)*n).reshape(k//128,n)
meta=logical.reshape(k//128,n//32,4,8).swapaxes(-2,-1).reshape(k//128,n)
idx=np.array([base+(j%4)*8+j//4 for base in range(0,n,32) for j in range(32)])
assert np.array_equal(meta,logical[:,idx])
# Emulate four byte lanes: multiplication needs no carry across byte boundaries.
for scale in range(1,18):
    codes=[15,3,8,0]
    word=sum(c << (8*i) for i,c in enumerate(codes))
    product=(word*scale)&0xffffffff
    lanes=[(product>>(8*i))&255 for i in range(4)]
    assert lanes==[c*scale for c in codes]
word=15 | (0<<8)
assert ((word*18)>>8)&255 == 1  # carry corrupts the next zero code
# Example: 15*16 - 7*16 = 128 is not a signed INT8 result.
overflow=(15*16-7*16)&255
assert overflow==128 and overflow-256 == -128
# Negative scaled-zero byte must be added, not interpreted as raw zero.
u=np.array([0,4,8,15]); scale=8; z=7
v=u*scale-z*scale
byte_sum=((u*scale)+((-z*scale)&255))&255
signed=np.where(byte_sum>=128,byte_sum-256,byte_sum)
assert np.array_equal(v,signed)
# Per-channel correction uses original X sum, rather than quantized X sum.
x=np.array([[.1,.2,.35,1.0],[.2,-.7,.1,2.]])
sx=np.abs(x).max(axis=1,keepdims=True)/127
qx=np.rint(x/sx); xhat=qx*sx
u=np.array([[1.,5.,2.,7.],[0.,3.,8.,15.]])
s1=np.array([.13,.27]); z=np.array([7.,9.]); zs=s1*z
code=(qx@u.T)*sx*s1-x.sum(axis=1,keepdims=True)*zs
strict=xhat@((u-z[:,None])*s1[:,None]).T
delta=(xhat-x).sum(axis=1,keepdims=True)*zs
assert np.allclose(code-strict,delta,atol=1e-14)
assert np.max(np.abs(delta))>1e-5
result={
    "evidence":"NumPy CPU packing/algebra, no converter/model/GPU execution",
    "seed":20260922,"checks_passed":5,
    "checks":["tile packing against coordinate oracle and roundtrip","metadata channel permutation",
              "byte multiply no-carry and overflow counterexamples","negative scaled-zero addition",
              "original-input-sum correction error identity"],
    "logical_shape":[n,k],"packed_shape":list(packed.shape),
    "input_sum_approximation_max_abs_error":float(np.max(np.abs(delta))),
    "error_identity_residual":float(np.max(np.abs(code-strict-delta)))
}
if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path)
    args=parser.parse_args(); text=json.dumps(result,indent=2,ensure_ascii=False)+"\n"
    if args.output: args.output.write_text(text)
    print(text,end="")
