import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['MPLCONFIGDIR']='/tmp/exe2-matplotlib'
import json,ast
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

root=Path('analysis_output');X=pd.read_csv('X.csv').to_numpy();U=np.load(root/'signal.npz')['U'];fit=np.load(root/'graph_fit.npz');V=fit['vertices'];sig=float(fit['sigma'])
segments=[[0,1],[0,2],[0,3]]
maps=['mod(2*x10+2*x23+x25+x30,1)','mod(2*x1-2*x6-x11-x24,1)']

def dist(P):
 D=V[1:]-V[0];t=np.clip((P-V[0])@D.T/(D*D).sum(1),0,1);Q=V[0]+t[:,:,None]*D
 return np.sqrt(((P[:,None,:]-Q)**2).sum(2).min(1))
rng=np.random.default_rng(77338)
# Independent resampling within each relevant original column destroys joint structure and preserves marginals.
nnull=200000
null=np.column_stack([sum(k*rng.choice(X[:,j],nnull) for j,k in [(9,2),(22,2),(24,1),(29,1)])%1,sum(k*rng.choice(X[:,j],nnull) for j,k in [(0,2),(5,-2),(10,-1),(23,-1)])%1])
dtrain=dist(U[:3500]);dtest=dist(U[3500:]);dnull=dist(null)
report=(
 'There is a noisy, one-dimensional Y-shaped graph hidden in two modular linear combinations. '
 'The first map uses x10, x23, x25, x30; the second uses x1, x6, x11, x24. '
 'The drawing gives the junction (point 0) and the three arm endpoints (points 1-3). '
 'The arms are approximately straight, with isotropic Gaussian scatter of standard deviation 0.005 per mapped coordinate. '
 'The mapped observations lie roughly between 0.425 and 0.619 horizontally and between 0.385 and 0.590 vertically; no arm crosses a modular border. '
 'Individual columns look uniform on [0,1]; Pearson correlations, PCA, binned pairwise dependence, and one/two-column Fourier searches did not reveal the structure. '
 'I searched sparse integer Fourier characters, discovered two independent four-column directions with coefficients 1 and 2, and plotted their values modulo one. '
 'On 3500 discovery rows the Fourier magnitudes were 0.96755 and 0.96019; on 1500 untouched validation rows they were 0.96890 and 0.95898. '
 'A three-segment model was fitted using only the discovery rows. Its median held-out distance is 0.00349; 95.6% of validation rows are within 0.01 and 99.73% within 0.015. '
 f'For independently resampled columns preserving the observed marginal distributions, only {100*np.mean(dnull<.01):.3f}% of 200000 null rows are within 0.01. '
 'An expanded four-column Fourier screen with integer coefficients up to magnitude 6 recovered only these directions and their harmonics as strong replicating signals. '
 'This establishes the Y structure; it does not prove the absence of every conceivable additional relationship. '
 'Only X.csv was read as a data source. Analysis code, chronological log, model parameters and plots are saved in analysis_output.'
)
submission={'status':'structure','map':maps,'drawing':{'points':np.round(V,8).tolist(),'segments':segments,'triangles':[]},'report':report}
Path('submission.json').write_text(json.dumps(submission,indent=2)+'\n')
# Check schema, grammar, expression lengths, dimensions, finite values, and numerical map equivalence.
s=json.loads(Path('submission.json').read_text())
assert set(s)=={'status','map','drawing','report'} and s['status']=='structure'
assert 1<=len(maps)<=3 and all(len(m)<=200 for m in maps)
allowed=(ast.Expression,ast.Call,ast.Name,ast.Load,ast.Constant,ast.BinOp,ast.UnaryOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Mod,ast.USub,ast.UAdd)
for m in maps:
 assert all(isinstance(n,allowed) for n in ast.walk(ast.parse(m,mode='eval')))
 vals=eval(m,{'__builtins__':{},'mod':np.mod},{f'x{j+1}':X[:,j] for j in range(30)})
 assert np.allclose(vals,U[:,maps.index(m)],rtol=0,atol=2e-15)
assert len(V)<=200 and V.shape[1]==len(maps) and np.isfinite(V).all()
assert all(0<=i<len(V) for edge in segments for i in edge)
metrics={'discovery_n':3500,'validation_n':1500,'noise_n':nnull,'vertices':V.tolist(),'sigma':sig,'branch_weights':fit['weights'].tolist(),'validation_distance_quantiles':dict(zip(['50%','90%','95%','99%'],np.quantile(dtest,[.5,.9,.95,.99]).tolist())),'within_0.01':{'discovery':float(np.mean(dtrain<.01)),'validation':float(np.mean(dtest<.01)),'independent_column_null':float(np.mean(dnull<.01))},'within_0.015':{'validation':float(np.mean(dtest<.015))}}
(root/'validation_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,10))
def draw(ax):
 for i,j in segments:ax.plot(V[[i,j],0],V[[i,j],1],color='#c53322',lw=2.5,zorder=3)
 ax.scatter(V[:,0],V[:,1],s=35,color='#c53322',zorder=4)
 ax.set_xlabel('u = mod(2*x10 + 2*x23 + x25 + x30, 1)');ax.set_ylabel('v = mod(2*x1 - 2*x6 - x11 - x24, 1)');ax.set_aspect('equal')
ax=axes[0,0];ax.scatter(null[:5000,0],null[:5000,1],s=2,color='#b5bac2',alpha=.3,label='Independent-column null');ax.scatter(U[:,0],U[:,1],s=2,color='#167ab3',alpha=.3,label='Observed rows');draw(ax);ax.set(xlim=(0,1),ylim=(0,1),title='Full modular domain: a small, concentrated graph');ax.legend(loc='lower left',markerscale=3)
ax=axes[0,1];ax.scatter(U[:3500,0],U[:3500,1],s=4,alpha=.28,color='#167ab3');draw(ax);ax.set(xlim=(.415,.625),ylim=(.375,.6),title='Discovery rows with fitted three-arm graph')
for i,p in enumerate(V):ax.annotate(str(i),p,xytext=(5,7),textcoords='offset points',weight='bold',color='#9d2419')
ax=axes[1,0];ax.scatter(U[3500:,0],U[3500:,1],s=5,alpha=.4,color='#e49619');draw(ax);ax.set(xlim=(.415,.625),ylim=(.375,.6),title='1500 held-out rows reproduce the same Y')
ax=axes[1,1]
for values,label,color in [(dtrain,'Discovery','#167ab3'),(dtest,'Held-out','#e49619'),(dnull,'Independent-column null','#777d86')]:
 a=np.sort(values);ax.plot(a,np.arange(1,len(a)+1)/len(a),label=label,color=color,lw=2)
ax.axvline(.01,color='black',ls=':',lw=1);ax.set(xscale='log',xlim=(.0001,1),ylim=(0,1.01),xlabel='Euclidean distance to fitted graph',ylabel='Fraction of rows within this distance',title='Generalization and separation from noise');ax.legend(loc='center right');ax.grid(alpha=.15)
fig.suptitle('Hidden structure in X.csv: a noisy Y in two modular coordinates',fontsize=15)
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(root/'03_final_validation.png',dpi=180);fig.savefig(root/'03_final_validation.svg')
summary=f'''# Findings from X.csv

Two modular coordinates reveal a noisy Y-shaped graph:

- `u = {maps[0]}`
- `v = {maps[1]}`

The junction and three endpoints are given in `submission.json`; there are four vertices and three segments. There are no filled triangles, extra connecting edges, or border crossings.

The graph was fitted on the first 3500 rows and validated on the remaining 1500. The two independent Fourier directions have validation magnitudes 0.96890 and 0.95898. Coordinate noise is approximately Gaussian with standard deviation {sig:.6f}. At distance 0.01 from the graph, the discovery, validation, and independent-column-null coverage are {np.mean(dtrain<.01):.4%}, {np.mean(dtest<.01):.4%}, and {np.mean(dnull<.01):.4%}, respectively.

The expanded Fourier screen found strong replication only for these two directions and their harmonics. This is positive evidence for the Y graph, not an exhaustive proof that every other possible relationship is absent.

## Saved artifacts

- `log.txt`: chronological method and result log, including numerical fitting correction.
- `01_initial.png`: marginal, correlation, PCA diagnostics.
- `02_discovered_projection.png`: the discovered map and discovery/validation shape.
- `03_final_validation.png` and `.svg`: fitted graph and validation against noise.
- `validation_metrics.json`: numerical validation results.
- Python scripts and `.npz` files: reproducible computations and fitted parameters.

Only `X.csv` was used as an input data source; no external files or internet search were used. The null sample is generated by independently resampling each relevant column of X.csv.
'''
(root/'README.md').write_text(summary)
with (root/'log.txt').open('a') as f:
 f.write('Numerical note: initial finite-segment fit encountered an optimizer line-search failure. Replaced unstable CDF subtraction with log-CDF differences and bounded vertices around initial locations; corrected fit converged.\n')
 f.write('Extended Fourier scan completed: only two independent strong directions plus their harmonics; no third coordinate justified by this scan.\n')
 f.write('Final null validation independently resampled relevant original columns, preserving marginals: '+json.dumps(metrics)+'\n')
 f.write('Wrote submission.json and final plots; validated exact top-level keys, allowed expression syntax, expression length, map equivalence, graph dimensions and valid indices.\n')
print(json.dumps(metrics,indent=2));print('submission.json successfully written and validated.')
