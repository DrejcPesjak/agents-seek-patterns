import numpy as np, json
poly=np.load('poly.npy'); P=np.load('uv.npy'); M=len(poly)
def dist(P,poly):
    A=poly;B=np.roll(poly,-1,0);best=np.full(len(P),np.inf)
    for i in range(len(A)):
        L2=((B[i]-A[i])**2).sum();t=np.clip(((P-A[i])@(B[i]-A[i]))/L2,0,1);best=np.minimum(best,np.linalg.norm(P-(A[i]+t[:,None]*(B[i]-A[i])),axis=1))
    return best
N=np.random.default_rng(5).random((8000,2))
dd,dn=dist(P,poly),dist(N,poly)
print('data dist mean %.4f p99 %.4f | noise dist mean %.3f, frac noise within 0.02: %.3f'%(dd.mean(),np.percentile(dd,99),dn.mean(),(dn<.02).mean()))
sub={"status":"structure",
 "map":["mod(x3 - x9 + x12 + 2*x25 + 2*x27 - x29, 1)","mod(x14 - 2*x2 - 2*x4 + x15 + x20 + x28, 1)"],
 "drawing":{"points":[[round(float(a),5),round(float(b),5)] for a,b in poly],"segments":[[i,(i+1)%M] for i in range(M)],"triangles":[]},
 "report":("Structure found: a closed 1-D curve shaped like a dagger/sword outline (pommel at top, crossguard at v~0.5, blade tip at bottom ~(0.465,0.17)) "
 "living in two hidden torus coordinates: u = (x3 - x9 + x12 + 2*x25 + 2*x27 - x29) mod 1 and v = (x14 - 2*x2 - 2*x4 + x15 + x20 + x28) mod 1. "
 "All rows lie within ~0.004 (rms; max ~0.02) of the curve, which spans u in [0.32,0.64], v in [0.16,0.76], length ~1.63, populated roughly uniformly; it does not cross the mod-1 border. "
 "The other 18 columns (x1,x5-x8,x10,x11,x13,x16-x19,x21-x24,x26,x30) appear to be pure uniform noise. "
 "How: every column is marginally uniform and all standard tests were null (correlations, PCA, 2-4 column histograms, nearest-neighbour stats, gradient-boosted classifier vs noise, ICA, torus Fourier scans over <=6 columns with small coefficients). "
 "The break came from floating-point fingerprints: 28 columns are exact multiples of 2^-53 with the usual trailing-zero pattern, but x3 and x14 are not, so they were computed (something mod 1). "
 "Regressing the trailing-zero count of x3/x14 on the other columns revealed which columns feed them (x9,x12,x25,x27,x29 and x2,x4,x15,x20,x28); an integer-coefficient Fourier search over those gave the two relations (|mean exp(2 pi i .)| = 0.90 and 0.63 vs ~0.03 noise floor). "
 "The scatter of (u,v) shows the curve; it was fitted with a 180-vertex closed polyline (k-means + TSP ordering + kernel-smoothed principal curve). Uniform noise passed through the same map fills the unit square uniformly.")}
assert all(len(m)<=200 for m in sub["map"])
json.dump(sub,open('../submission.json','w'))
print(len(sub['drawing']['points']),len(sub['report']))
