import os
os.environ['MPLCONFIGDIR']='/tmp/mpl-cache'
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
X=np.load('analysis_output/X.npy');Y=np.load('analysis_output/recovered_coordinates.npy')
C=np.load('analysis_output/exact_curve_coefficients.npy');center=C[0].copy();Yc=Y-center;C[0]=0
# Parametric curve fitted to just 500 rows. Analytic derivatives permit independent distance checks.
def curve(t,der=0):
 t=np.asarray(t);out=np.zeros(t.shape+(2,))
 if der==0:out+=C[0]
 for k in range(1,4):
  out+=(k**der)*(np.cos(k*t+der*np.pi/2)[...,None]*C[2*k-1]+np.sin(k*t+der*np.pi/2)[...,None]*C[2*k])
 return out

def exact_distance(P):
 t=np.arctan2(P[:,1],P[:,0])
 for _ in range(20):
  q=curve(t);dq=curve(t,1);ddq=curve(t,2);e=q-P
  dt=np.sum(e*dq,axis=1)/(np.sum(dq*dq,axis=1)+np.sum(e*ddq,axis=1))
  t-=np.clip(dt,-.25,.25)
 return np.linalg.norm(curve(t)-P,axis=1)
# Allocate segments by sqrt(curvature)*arc length to balance chord approximation errors.
t=np.linspace(0,2*np.pi,50001);d=curve(t,1);dd=curve(t,2)
weight=np.sqrt(abs(d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/np.linalg.norm(d,axis=1))
s=cumulative_trapezoid(weight,t,initial=0);ts=np.interp(np.linspace(0,s[-1],201)[:-1],s,t);points=curve(ts)
segments=np.c_[np.arange(200),np.roll(np.arange(200),-1)]
def poly_distance(P):
 result=np.full(len(P),np.inf)
 for a,b in segments:
  v=points[b]-points[a];u=np.clip((P-points[a])@v/(v@v),0,1)
  result=np.minimum(result,np.linalg.norm(P-points[a]-u[:,None]*v,axis=1))
 return result
rng=np.random.default_rng(734)
# Actual 50-column uniform control rows, passed through the same modular sums.
N=rng.random((100000,50));W=np.zeros((50,2));W[[3,8,19,25,29,49],0]=[1,-1,1,1,-1,-1];W[[18,28,30,35,37,45],1]=[1,-1,1,1,1,1]
U=(N@W-center+.5)%1-.5
idx=np.random.default_rng(92).choice(len(Y),500,replace=False);held=np.setdiff1d(np.arange(len(Y)),idx)
e=exact_distance(Yc[held]);ep=poly_distance(Yc[held]);en=poly_distance(U)
summary={'fitted_rows':500,'held_out_rows':len(held),'exact_curve_holdout_distance_quantiles':dict(zip(['median','95%','max'],map(float,np.quantile(e,[.5,.95,1])))), 'drawing_holdout_distance_quantiles':dict(zip(['median','95%','max'],map(float,np.quantile(ep,[.5,.95,1])))), 'uniform_noise_rows':len(U),'noise_distance_quantiles':dict(zip(['1%','median','95%'],map(float,np.quantile(en,[.01,.5,.95])))), 'fraction_noise_within_0.0001':float(np.mean(en<.0001)), 'fraction_data_within_0.0001':float(np.mean(ep<.0001)), 'noise_output_std':U.std(0).tolist(),'curve_coefficients':C.tolist(),'coordinate_offsets':center.tolist()}
json.dump(summary,open('analysis_output/validation.json','w'),indent=2)
expr=[f'mod(x4-x9+x20+x26-x30-x50+{.5-center[0]:.14f},1)-0.5',f'mod(x19-x29+x31+x36+x38+x46+{.5-center[1]:.14f},1)-0.5']
report=("The data contain a one-dimensional closed curve hidden by modular mixing. The file actually has 4000 headerless rows and 50 columns; all 50 were authorized for analysis. Two sums of six columns each recover the curve; the remaining 38 columns are not needed for this recovery. The map centers and unwraps both coordinates so the curve never crosses a wrapping border. Its shape is a smooth, slightly asymmetric rounded diamond, with no filled interior and no measurable thickness. More precisely, for 0<=t<2*pi, its coordinates are approximately u=0.24*cos(t)+0.005436206328*cos(2*t)+0.003740070342*sin(2*t)+0.032966152618*cos(3*t)+0.020253455423*sin(3*t), v=0.24*sin(t)-0.014686449995*cos(2*t)+0.009499503341*sin(2*t)+0.012354584070*cos(3*t)-0.013134607360*sin(3*t). Coefficients were fitted on 500 rows; all 3500 other rows lie within %.3g of this curve. Single-column summaries, pairwise dependence tests and PCA looked like uniform noise. A sparse Fourier search through modular combinations involving up to six columns found two independent signals, each with amplitude about 0.5, confirmed on independent rows. A 200-vertex closed polyline approximates the fitted curve; maximum held-out distance to this drawing is %.3g. Of 100000 independent uniform 50-column noise rows passed through the same map, %.4g%% lie within 0.0001 of the drawing, compared with 100%% of held-out data. Scripts, diagnostic plots, coefficients and numerical validation are saved in analysis_output. The drawing contains only the outline; triangles are intentionally empty."%(e.max(),ep.max(),100*np.mean(en<.0001)))
submission={'status':'structure','map':expr,'drawing':{'points':points.tolist(),'segments':segments.tolist(),'triangles':[]},'report':report}
json.dump(submission,open('submission.json','w'),indent=2,allow_nan=False)
# Final plots, generated from all correctly parsed input rows.
fig,axs=plt.subplots(5,10,figsize=(20,10))
for j,ax in enumerate(axs.flat):ax.hist(X[:,j],30,color='steelblue');ax.set_title(f'x{j+1}',fontsize=9);ax.set_xticks([0,1]);ax.set_yticks([])
fig.suptitle('All 50 columns look uniform individually (4,000 rows)');fig.tight_layout();fig.savefig('analysis_output/marginals.png',dpi=130);plt.close(fig)
corr=np.corrcoef(X.T);fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(corr-np.eye(50),vmin=-.06,vmax=.06,cmap='coolwarm');fig.colorbar(im);ax.set_title('Off-diagonal Pearson correlations');fig.savefig('analysis_output/correlations.png',dpi=130);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(16,5))
axs[0].scatter(*Yc.T,s=2,alpha=.4);axs[0].set_title('Recovered data: 4,000 rows')
axs[1].plot(*np.r_[points,points[:1]].T,color='black',lw=1);axs[1].scatter(*Yc[held[::10]].T,s=7,alpha=.6);axs[1].set_title('200-segment drawing + held-out rows')
axs[2].scatter(*U[:5000].T,s=2,alpha=.3,color='gray');axs[2].plot(*np.r_[points,points[:1]].T,color='red',lw=1);axs[2].set_title('Independent uniform-noise control')
for ax in axs:ax.set_aspect('equal');ax.set_xlabel('map output 1');ax.set_ylabel('map output 2');ax.set_xlim(-.51,.51);ax.set_ylim(-.51,.51)
fig.tight_layout();fig.savefig('analysis_output/final_structure.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));ax.hist(ep,bins=40,alpha=.7,label='Held-out data');ax.set_xlabel('Distance to submitted drawing');ax.legend();fig.tight_layout();fig.savefig('analysis_output/heldout_distances.png',dpi=150);plt.close(fig)
with open('analysis_output/work_log.txt','a') as f:
 f.write('\nSecond independent Fourier direction: x19-x29+x31+x36+x38+x46, amplitude .5051 discovery / .4995 validation. Complete unit-coefficient six-variable scan found exactly these two independent directions.\n')
 f.write('Plotted centered modular sums: a smooth thin closed loop. Implicit polynomial degree-six fit has machine-precision residual; degree-five and lower do not.\n')
 f.write('Fitted a parametric circle plus second and third harmonics on 500 random rows. Second harmonics alone cannot fit; adding third harmonics gives machine-precision residuals, including on all 3500 held-out rows. Radial periodic spline independently confirms smoothness.\n')
 f.write('Balanced polyline chord errors using curvature, allocating all 200 allowed vertices to the closed outline; no triangles. Shifted modular cuts away from the shape to avoid artificial seam connections.\n')
 f.write(json.dumps(summary,indent=2)+'\n')
 f.write('Wrote submission.json and final diagnostic plots. No other source files or internet used.\n')
print(json.dumps(summary,indent=2));print('MAP',expr)
