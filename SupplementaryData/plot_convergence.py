import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from espei.analysis import truncate_arrays

def conv(i):
    if i<10:
       return 'VV000'+str(i)
    else:
       return 'VV00'+str(i)

tr_0 = np.load('trace_prop.npy')
ln_0 = np.load('prob_prop.npy')
#trp,lnp=truncate_arrays(tr_0,ln_0)
trace, lnprob = truncate_arrays(tr_0,ln_0)
np.save("trace_370.npy",trace)
#tr_2 = np.load('trace_prop_trunc.npy')
#tr3 = np.load('trace_prop3.npy')
#ln3= np.load('prob_prop3.npy')
#tr_3, ln_3 = truncate_arrays(tr3,ln3)
#trace = np.append(trace,tr_2,axis=1)
#trace=np.append(trace,tr_3,axis=1)
print (np.shape(trace), np.shape(lnprob))
if not os.path.exists('ParamsTriangularPropagatedCu'):
   os.mkdir('ParamsTriangularPropagatedCu')
pars = \
{"VV0000":"G(CUMG2,CU:MG;0)",
"VV0001": "L(FCC_A1,CU,MG;1) 'V1'",
"VV0002": "L(FCC_A1,CU,MG;0) V2",
"VV0004": "L(HCP_A3,CU,MG;0) 'V4'",
"VV0003": "L(HCP_A3,CU,MG;1) 'V3'", 
"VV0006": "G(LAVES_C15,CU:MG;0) 'V6'",
"VV0007": "G(LAVES_C15,MG:CU;0) 'V7'",
"VV0008": "G(LAVES_C15,MG:MG;0) 'V8'",
"VV0009": "L(LAVES_C15,CU:CU,MG;0) 'V9'",
"VV0010": "L(LAVES_C15,CU,MG:MG;0) 'V10'",
"VV0011": "L(LIQUID,CU,MG;3) 'V11'",
"VV0012": "L(LIQUID,CU,MG;2) 'V12'",
"VV0013": "L(LIQUID,CU,MG;1) 'V13'",
"VV0014": "L(LIQUID,CU,MG;0) 'V14'",
"VV0005": "G(LAVES_C15,CU:CU;0)",
"VV0015": "CU(A)",
"VV0016": "CU(B)",
"VV0017": "CU(D)",
"VV0018": "CU(E)",
"VV0019": "CU(Ssolid)",
"VV0020": "CU(Sliquid)",
"VV0021": "CU(H298_liquid)",
"VV0022": "CU(Tb)"}

num_walkers = trace.shape[0]
#print (num_walkers, ' walkers')
iterations = trace.shape[1]
#print (trace.shape[1], ' iterations')
num_parameters = trace.shape[2]
for parameter in range(num_parameters):
    ax = plt.figure().gca()
    ax.set_xlabel('Iterations')
    ax.set_ylabel(pars[conv(parameter)])
    for w in range(num_walkers):
        if w<20:
           ax.plot(trace[w,:,parameter],alpha=0.01, color='blue')
        elif 20<=w<=80:
           ax.plot(trace[w,:,parameter],alpha=0.05, color='blue')
        else:
           ax.plot(trace[w,:,parameter],alpha=0.15, color='blue')
    init_par = trace[0,0,parameter]
    for i in range(iterations):
        plt.scatter([i],[np.percentile(trace[:,i,parameter],97.5)], color='orange')
        plt.scatter([i],[np.percentile(trace[:,i,parameter],2.5)], color='orange')
        plt.scatter([i],[np.percentile(trace[:,i,parameter],50)], color='orange')
    lb_par = init_par - 5.0*abs(init_par)
    ub_par = lb_par + 10.0*abs(init_par)
    #print (lb_par,init_par,ub_par)
    ax.hlines(init_par,0,len(trace[0,:,0]),color='black')
    ax.hlines(lb_par,0,len(trace[0,:,0]),color='black',linestyle=':')
    ax.hlines(ub_par,0,len(trace[0,:,0]),color='black',linestyle=':')
    plt.ylim(lb_par + 2.0*abs(init_par), ub_par - 2.0*abs(init_par))
    plt.tight_layout()
    plt.savefig('ParamsTriangularPropagatedCu/Parameter{}.png'.format(parameter))
    #print ("Finished {}".format(parameter))
    plt.close() 
    for i in range(iterations):
        plt.scatter([i],[abs(np.max(trace[:,i,parameter])-np.min(trace[:,i,parameter]))], color='blue')

    plt.xlabel('Iterations')
    plt.ylabel(pars[conv(parameter)])
    plt.savefig('ParamsTriangularPropagatedCu/ParameterDiff{}.png'.format(parameter))
    plt.close()
