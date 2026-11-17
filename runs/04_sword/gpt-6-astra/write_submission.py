import json,ast
import numpy as np
X=np.load('analysis_outputs/X.npy');P=np.load('analysis_outputs/curve.npy')
expressions=['mod(x3-x9+x12+2*x25+2*x27-x29,1)','mod(2*x2+2*x4-x14-x15-x20-x28,1)']
report=('The data contain a noisy, one-dimensional closed outline of a sword, revealed by the two modulo-1 combinations in map. The blade points upward near (0.465,0.83), the curved crossguard spans approximately u=0.325 to 0.635 near v=0.50, and the rounded handle ends near (0.49,0.25). These combinations involve columns x2,x3,x4,x9,x12,x14,x15,x20,x25,x27,x28,x29. The outline stays inside the unit square, so there is no wrap-border crossing. The drawing is a 160-vertex closed polyline following the center of the thin noisy band. '
'I used only X.csv, which actually contains 8000 rows and 30 columns. Histograms, correlations, PCA, pairwise contingency tests, nearest-neighbor comparisons, noise classifiers, and small Fourier searches initially looked like independent uniform noise. The decisive search compared products of complex Fourier features from every three-column integer combination with coefficient magnitudes 1 or 2, finding six-column modular relations. On 3000 separate validation rows the two relations had Fourier concentrations 0.89545 and 0.62564, far above the uniform-noise scale of about 0.018. Plotting the two decoded coordinates exposed the sword. '
'I fit the polyline using only the first 5000 decoded rows, using cluster initialization, a closed traveling-salesman ordering, and a regularized principal-curve fit. On the remaining 3000 rows the median Euclidean distance to the drawing was 0.002711; 98.27% were within 0.01. For 100000 independent uniform points, which have exactly the mapped noise distribution because the expressions use disjoint column sets, the median distance was 0.210471 and 3.318% were within 0.01. Distances here are in the raw map coordinates; both noise outputs have standard deviation 1/sqrt(12). Scripts, numerical progress logs, and diagnostic and validation plots are saved in analysis_outputs and the analysis scripts.')
submission={'status':'structure','map':expressions,'drawing':{'points':np.round(P,8).tolist(),'segments':[[i,(i+1)%len(P)] for i in range(len(P))],'triangles':[]},'report':report}
with open('submission.json','w') as f:json.dump(submission,f,indent=2);f.write('\n')
# Validate the format, expressions, and geometric topology.
with open('submission.json') as f:s=json.load(f)
assert set(s)=={'status','map','drawing','report'}
assert s['status']=='structure' and 1<=len(s['map'])<=3
assert all(len(t)<=200 for t in s['map'])
for expression in s['map']:
 for node in ast.walk(ast.parse(expression,mode='eval')):
  assert isinstance(node,(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Call,ast.Name,ast.Load,ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Mod,ast.USub)),type(node)
  if isinstance(node,ast.Name):assert node.id=='mod' or node.id in ['x'+str(i+1) for i in range(30)]
Y=np.column_stack([eval(t,{'__builtins__':{},'mod':np.mod},{'x'+str(i+1):X[:,i] for i in range(30)}) for t in s['map']])
assert np.allclose(Y,np.load('analysis_outputs/decoded.npy'),atol=1e-12)
assert len(s['drawing']['points'])<=200 and np.array(s['drawing']['points']).shape==(160,2)
assert np.isfinite(P).all()
assert all(0<=i<160 for seg in s['drawing']['segments'] for i in seg)
assert np.array(s['drawing']['segments']).ravel().size==320
assert all(np.count_nonzero(np.array(s['drawing']['segments'])==i)==2 for i in range(160))
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
for i in range(len(P)):
 a,b=P[i],P[(i+1)%len(P)]
 for j in range(i+2,len(P)):
  if i==0 and j==len(P)-1:continue
  c,d=P[j],P[(j+1)%len(P)]
  assert not (cross(b-a,c-a)*cross(b-a,d-a)<0 and cross(d-c,a-c)*cross(d-c,b-c)<0),('intersecting segments',i,j)
print('Validated submission.json: two exact decoding expressions, 160 points, 160 segments, zero triangles, one closed non-self-intersecting outline.')
with open('analysis_outputs/log.txt','a') as f:
 f.write('\nFINAL DISCOVERY: two mixed-coefficient six-column Fourier relations generalize, revealing a sword outline. Earlier negative results were interim results only.\n')
 f.write('Maps: '+str(expressions)+'\n')
 f.write('Eight-column scan stopped after the successful mixed-six discovery; it is not claimed complete.\n')
 f.write('submission.json written and validated. Final drawing uses first 5000 rows only; held-out 3000-row geometric validation is reported above.\n')
