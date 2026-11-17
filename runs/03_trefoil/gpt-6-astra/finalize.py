import numpy as np,json,time
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
A=np.load('data.npy');R=np.load('revealed.npz');Y=R['Y']+1;T=np.load('trefoil_fit.npz');ix=T['trainix'];jx=T['testix'];B=T['B']
f=open('analysis.log','a')
def log(s):print(s,flush=True);f.write(str(s)+'\n');f.flush()
def feat(t):
 r=2+np.cos(3*t)
 return np.column_stack([np.ones(len(t)),r*np.cos(2*t),r*np.sin(2*t),np.sin(3*t)])
def dfeat(t):
 r=2+np.cos(3*t);dr=-3*np.sin(3*t)
 return np.column_stack([np.zeros(len(t)),dr*np.cos(2*t)-2*r*np.sin(2*t),dr*np.sin(2*t)+2*r*np.cos(2*t),3*np.cos(3*t)])
def segment_dist(Y,P):
 out=np.full(len(Y),np.inf)
 for a,b in zip(P,np.roll(P,-1,axis=0)):
  v=b-a;t=np.clip((Y-a)@v/np.sum(v*v),0,1);out=np.minimum(out,np.linalg.norm(Y-a-t[:,None]*v,axis=1))
 return out
# Validation uses only the affine trefoil parameters fitted to training rows.
P=feat(np.arange(200)*2*np.pi/200)@B
train_d=segment_dist(Y[ix],P);test_d=segment_dist(Y[jx],P)
rng=np.random.default_rng(913);null=rng.uniform(size=(200000,3));null_d=segment_dist(null,P)
# Applying the map to independently permuted CSV columns gives an additional null.
shuffled=np.column_stack([rng.permutation(A[:,j]) for j in range(30)])
shuffleY=(shuffled@R['B'].T)%1;shuffle_d=segment_dist(shuffleY,P)
threshold=float(np.quantile(train_d,.99))
stats={'train_rms':float(np.sqrt(np.mean(train_d**2))),'heldout_rms':float(np.sqrt(np.mean(test_d**2))),'heldout_quantiles':dict(zip(['median','q90','q95','q99','max'],map(float,np.quantile(test_d,[.5,.9,.95,.99,1])))),'noise_median':float(np.median(null_d)),'threshold_train_q99':threshold,'heldout_fraction_below_threshold':float(np.mean(test_d<=threshold)),'uniform_noise_fraction_below_threshold':float(np.mean(null_d<=threshold)),'shuffled_noise_fraction_below_threshold':float(np.mean(shuffle_d<=threshold)),'noise_scaled_heldout_rms':float(np.sqrt(np.mean(test_d**2))*np.sqrt(12))}
log('Held-out validation of 200-segment TRAINING-ONLY trefoil: '+json.dumps(stats,indent=2))
with open('validation.json','w') as out:json.dump(stats,out,indent=2)
fig,ax=plt.subplots(figsize=(9,5))
bins=np.linspace(0,1.9,100)
ax.hist(test_d*np.sqrt(12),bins=bins,density=True,alpha=.8,label='4,000 held-out CSV rows');ax.hist(null_d*np.sqrt(12),bins=bins,density=True,alpha=.6,label='200,000 independent uniform-noise map outputs');ax.set_xlabel('Distance to drawing / noise standard deviation');ax.set_ylabel('Density');ax.legend();fig.tight_layout();fig.savefig('plots/validation_distances.png',dpi=180);plt.close(fig)
# Refit the validated low-parameter model to all 8,000 rows for the final drawing.
phase=np.empty(len(Y));phase[ix]=T['trainphase'];phase[jx]=T['testphase'];n=len(Y)
def fun(z):return (feat(z[12:])@z[:12].reshape(4,3)-Y).ravel()
def jac(z):
 B=z[:12].reshape(4,3);t=z[12:];F=feat(t);v=dfeat(t)@B
 rows=np.repeat(np.arange(3*n),4);cols=np.tile(np.arange(4)*3,3*n)+np.repeat(np.tile(np.arange(3),n),4);vals=np.repeat(F,3,axis=0).ravel()
 return coo_matrix((np.r_[vals,v.ravel()],(np.r_[rows,np.arange(3*n)],np.r_[cols,12+np.repeat(np.arange(n),3)])),shape=(3*n,12+n)).tocsr()
res=least_squares(fun,np.r_[B.ravel(),phase],jac=jac,method='trf',ftol=1e-10,xtol=1e-10,gtol=1e-9,max_nfev=100)
B=res.x[:12].reshape(4,3);phase=res.x[12:];P=feat(np.arange(200)*2*np.pi/200)@B
np.savez('final_trefoil.npz',B=B,phase=phase,points=P)
log('Full-data affine trefoil coefficient matrix:\n'+str(B))
log('Full-data residual Euclidean RMS '+str(np.sqrt(np.mean(fun(res.x)**2)*3)))
log('Map bounds: '+str(np.column_stack([Y.min(0),Y.max(0)])))
# Sampling-distribution diagnostics, without claiming exact uniformity.
counts,_=np.histogram(phase%(2*np.pi),bins=24,range=(0,2*np.pi));log('Latent phase counts in 24 equal bins '+str(counts))
expressions=['mod(x3-x6+2*x8-2*x10-x29,1)','mod(x4-2*x13-x19+2*x22+x30,1)','mod(x2+2*x12+x16+x24+2*x27,1)']
report=(
 'The CSV contains 8000 rows and 30 columns. There is strong, reproducible one-dimensional nonlinear structure: a noisy affine image of the (2,3) torus trefoil knot. '
 'The three map expressions recover it. Each is a five-column integer sum reduced modulo 1; the three column groups are disjoint. '
 'The recovered curve lies strictly inside the unit cube, so it never crosses a wrap border. Its observed coordinate ranges are approximately '
 +str([[round(float(v),4) for v in row] for row in np.column_stack([Y.min(0),Y.max(0)])])+'. '
 'More explicitly, for 0<=t<2*pi define q(t)=((2+cos(3*t))*cos(2*t),(2+cos(3*t))*sin(2*t),sin(3*t)). '
 'The fitted curve is c+q(t)*M, with row-vector convention, c='+str(np.round(B[0],8).tolist())+' and M='+str(np.round(B[1:],8).tolist())+'. '
 'The drawing samples this closed curve at 200 equally spaced values of t and joins consecutive points, including the last to the first. No surface faces are warranted. '
 'Typical perpendicular scatter is consistent with approximately 0.005 noise per recovered coordinate. '
 'Method: inspected only X.csv; checked marginal distributions, correlations, PCA, pairwise mutual information, nearest-neighbor geometry, and real-versus-column-shuffled classifiers. Those basic checks looked like noise. '
 'Searched sparse integer Fourier characters using products of one-, two-, and three-column complex exponentials. The decisive search allowed five participating columns with coefficients +/-1 or +/-2. '
 'The three winning Fourier magnitudes were 0.934, 0.759, 0.731 on 2000 discovery rows and 0.930, 0.757, 0.724 on a disjoint 4000-row test set. '
 'A local-neighbor spectral embedding ordered the recovered loop. A standard affine trefoil fitted on 4000 training rows matched held-out rows as accurately as a flexible Fourier principal curve, supporting a genuine simple knot rather than sample-specific wiggles. '
 f'For the 200-segment training-only drawing, held-out Euclidean RMS distance was {stats["heldout_rms"]:.6f} (or {stats["noise_scaled_heldout_rms"]:.5f} after division by uniform-noise standard deviation 1/sqrt(12)). '
 f'Median held-out distance was {stats["heldout_quantiles"]["median"]:.6f}, versus {stats["noise_median"]:.6f} for 200000 independent uniform-noise outputs. '
 f'At the training 99th-percentile distance cutoff {threshold:.6f}, {100*stats["heldout_fraction_below_threshold"]:.2f}% of held-out rows were accepted, compared with {100*stats["uniform_noise_fraction_below_threshold"]:.3f}% of uniform noise and {100*stats["shuffled_noise_fraction_below_threshold"]:.3f}% of independently column-shuffled CSV rows. '
 'Because the map uses disjoint column groups and each contains a coefficient 1, independent Uniform(0,1) input columns produce exactly independent Uniform(0,1) outputs. '
 'The final drawing refits only the validated affine trefoil to all rows. Full analysis steps are logged in analysis.log; scripts and plots are saved alongside the submission. No other data files or internet sources were used.'
)
submission={'status':'structure','map':expressions,'drawing':{'points':np.round(P,10).tolist(),'segments':[[i,(i+1)%200] for i in range(200)],'triangles':[]},'report':report}
with open('submission.json','w') as out:json.dump(submission,out,indent=2,allow_nan=False);out.write('\n')
# Final scientific views: held-out data and a depth-shaded fitted centerline.
fig,axs=plt.subplots(1,3,figsize=(16,6))
for ax,(j,k) in zip(axs,[(0,1),(0,2),(1,2)]):
 ax.scatter(Y[jx,j],Y[jx,k],s=1,alpha=.25,color='#2478b5');ax.plot(np.r_[P[:,j],P[0,j]],np.r_[P[:,k],P[0,k]],color='#d14435',lw=1.5);ax.set_aspect('equal');ax.set_xlabel(f'u{j+1}');ax.set_ylabel(f'u{k+1}')
fig.suptitle('Recovered trefoil: held-out rows and final 200-point drawing');fig.tight_layout();fig.savefig('plots/final_trefoil_projections.png',dpi=180);plt.close(fig)
# Transform solely for a canonical geometric illustration; the submitted map stays as the three simple wrapped sums.
canonical=(Y-B[0])@np.linalg.inv(B[1:]);q=feat(np.arange(1600)*2*np.pi/1600)[:,1:]
fig,axs=plt.subplots(1,2,figsize=(12,6))
for ax,(az,el) in zip(axs,[(18,72),(40,35)]):
 az,el=np.deg2rad([az,el]);rot=np.array([[np.cos(az),np.sin(az),0],[-np.sin(az)*np.sin(el),np.cos(az)*np.sin(el),np.cos(el)],[np.sin(az)*np.cos(el),-np.cos(az)*np.cos(el),np.sin(el)]])
 z=q@rot.T;segments=np.stack([z,np.roll(z,-1,axis=0)],axis=1);order=np.argsort(segments[:,:,2].mean(1))
 lc=LineCollection(segments[order,:,:2],cmap='viridis',linewidths=4);lc.set_array(segments[order,:,2].mean(1));ax.add_collection(lc);ax.autoscale();ax.set_aspect('equal');ax.axis('off')
fig.suptitle('Trefoil knot in canonical coordinates (color shows depth)');fig.tight_layout();fig.savefig('plots/trefoil_geometry.png',dpi=180);plt.close(fig)
log('Wrote submission.json: structure, 3 expressions, 200 points, 200 segments, no triangles. No wrap crossings.')
