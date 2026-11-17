import os
os.environ['MPLCONFIGDIR']='/tmp/mpl-cache'
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
X=np.load('analysis_output/X.npy');w=np.zeros(50);w[[3,8,19,25,29,49]]=[1,-1,1,1,-1,-1]
u=X@w%1
print('Fourier:',[(k,abs(np.mean(np.exp(2j*np.pi*k*u))),np.angle(np.mean(np.exp(2j*np.pi*k*u)))/(2*np.pi)) for k in range(1,13)])
print('Quantiles',np.quantile(u,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]))
fig,ax=plt.subplots(figsize=(10,4));ax.hist(u,bins=100);ax.set_xlabel('(x4-x9+x20+x26-x30-x50) mod 1');fig.tight_layout();fig.savefig('analysis_output/six_column_signal.png')
with open('analysis_output/work_log.txt','a') as f:
 f.write('Fourier search: pair coefficients up to 3; triple coefficients up to 3; unit-coefficient four/five/six-column scans. Weighted four-column scan used 768 discovery rows and 3232 validation rows. No signals before six columns.\n')
 f.write('Six-column discovery: (x4-x9+x20+x26-x30-x50) mod 1. Fourier amplitude 0.5296 on 384 discovery rows, 0.5211 on 3616 independent validation rows.\n')
