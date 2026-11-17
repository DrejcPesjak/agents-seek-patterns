import numpy as np,json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
X=np.load('working_data.npz')['X'];Y=np.load('coordinates.npy');fit=np.load('graph_fit.npz');P=fit['P'];tr=np.load('graph_train_fit.npz');Pt=tr['p'][:8].reshape(4,2);held=Y[tr['test_indices']]
def distances(Z,P):
 D=P[1:]-P[0];Q=Z[:,None,:]-P[0];T=np.clip(np.einsum('nki,ki->nk',Q,D)/(D*D).sum(1),0,1)
 return np.sqrt(((Q-T[:,:,None]*D)**2).sum(2).min(1))
rng=np.random.default_rng(991);N=200000
# The two maps use disjoint columns, hence independent uniform inputs yield independent uniform outputs exactly.
noise=rng.random((N,2));dn=distances(noise,Pt);dt=distances(held,Pt)
# Separate null preserving every observed marginal, by independently resampling each input column.
V=np.load('fourier_results.npz')['results'][:2];Z=np.column_stack([rng.choice(X[:,i],N) for i in range(30)]);perm=Z@V.T%1;dp=distances(perm,Pt)
stats={'heldout_rows':len(held),'geometry_training_rows':2500,'null_rows':N,'heldout_distance_quantiles':dict(zip(['50%','90%','95%','99%'],np.quantile(dt,[.5,.9,.95,.99]).tolist())),'uniform_null_distance_median':float(np.median(dn)),'marginal_resampling_null_distance_median':float(np.median(dp))}
th=float(np.quantile(dt,.95));stats.update({'threshold_for_95_percent_heldout':th,'uniform_null_fraction_within_threshold':float(np.mean(dn<=th)),'resampled_null_fraction_within_threshold':float(np.mean(dp<=th)),'full_fit_sigma':float(fit['sigma']),'full_fit_vertices':P.tolist(),'segment_lengths':np.linalg.norm(P[1:]-P[0],axis=1).tolist(),'segment_weights':fit['weights'].tolist()})
with open('validation.json','w') as f:json.dump(stats,f,indent=2)
print(json.dumps(stats,indent=2))
fig,ax=plt.subplots(1,3,figsize=(17,5))
for a,Z,title in [(ax[0],held,'Held-out rows and training-fit graph'),(ax[1],noise[:5000],'Independent uniform noise and graph')]:
 a.scatter(*Z.T,s=2,alpha=.35)
 for e in Pt[1:]:a.plot(*np.array([Pt[0],e]).T,color='crimson',lw=2)
 a.set_aspect('equal');a.set(title=title,xlabel='u',ylabel='v')
ax[2].hist(dt,bins=np.linspace(0,.7,140),density=True,alpha=.65,label='Held-out data');ax[2].hist(dn,bins=np.linspace(0,.7,140),density=True,alpha=.65,label='Uniform noise');ax[2].set(xlabel='Distance to training-fit segments',ylabel='Density',yscale='log');ax[2].legend();fig.tight_layout();fig.savefig('06_validation_against_noise.png')
report=f'''There is a noisy one-dimensional Y-shaped graph hidden in two modular linear combinations. Set u=(2*x10+2*x23+x25+x30)%1 and v=(2*x1-2*x6-x11-x24)%1, taking remainders in [0,1). Its junction is approximately ({P[0,0]:.6f},{P[0,1]:.6f}); the three endpoints are {[(round(a,6),round(b,6)) for a,b in P[1:]]}. The drawing contains just these three straight branches, not their filled convex hull. Rows are approximately uniform in arclength along the branches, with transverse noise consistent with standard deviation {float(fit['sigma']):.6f} in these coordinates. The participating columns are x1,x6,x10,x11,x23,x24,x25,x30. This describes structure in a projection; it does not assert that all 30 columns jointly lie on a one-dimensional manifold or prove the other columns independent. Single-column histograms, pairwise dependence checks, correlations (maximum absolute 0.04365), and covariance eigenvalues initially looked like uniform noise. I searched sparse integer Fourier frequencies, splitting discovery and validation rows; only two independent four-column frequencies were strongly concentrated (held-out magnitudes about 0.960 and 0.969). A broader search with coefficients up to 3 found no further strong four-column relation. Plotting the fractional parts revealed the Y. I fitted a mixture of three uniformly sampled line segments with a shared junction and isotropic Gaussian noise. A separate 2500/2500 split assessed geometry: held-out median distance {np.median(dt):.6f}, 95th-percentile distance {th:.6f}. At that threshold, {100*np.mean(dn<=th):.3f}% of 200000 independent uniform-noise projections and {100*np.mean(dp<=th):.3f}% of independently resampled-marginal null projections were close to the drawing. These are internal validation results, not measurements on the grader's hidden rows. Final vertices were refitted using all 5000 rows. Only X.csv and synthetic null samples were used; no external files or internet sources were consulted. See analysis.log, validation.json, and plots 01 through 06 for the audit trail.'''
submission={'status':'structure','map':['(2*x10+2*x23+x25+x30)%1','(2*x1-2*x6-x11-x24)%1'],'drawing':{'points':np.round(P,8).tolist(),'segments':[[0,1],[0,2],[0,3]],'triangles':[]},'report':report}
with open('submission.json','w') as f:json.dump(submission,f,indent=2)
assert set(submission)=={'status','map','drawing','report'}
assert all(len(s)<=200 for s in submission['map'])
assert len(P)<=200 and P.shape[1]==len(submission['map'])
# Evaluate the exact submitted expression syntax against the actual CSV columns.
env={f'x{i+1}':X[:,i] for i in range(30)}
for i,s in enumerate(submission['map']):assert np.allclose(eval(s,{'__builtins__':{}},env),Y[:,i])
with open('analysis.log','a') as f:
 f.write('\nThe first centered-modulo plot cut the object across the boundary at 0.5. Switching to [0,1) fractional parts recovered a contiguous Y.\n')
 f.write('Initial graph optimization failed because trial segments could have zero length; bounding each vertex near its visually established branch solved this. Both final optimization runs converged.\n')
 f.write('Fitted three uniform segments convolved with isotropic Gaussian noise, using 2500 training rows, evaluating 2500 held-out rows, then refitting final vertices on all rows.\n')
 f.write(json.dumps(stats,indent=2)+'\n')
 f.write('Wrote submission.json. Verified required keys, expression length, drawing size and dimensions, and exact expression evaluation.\n')
