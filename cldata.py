# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:16:14 2019

@author: sounderb
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
import math
import time as time

t1 = time.perf_counter()

cldata = {'Depth': [0.25, 0.75, 1.25,2,3], 'Cl': [11.19401,4.969,2.048,0.2,0.01]}
cldata = pd.DataFrame.from_dict(cldata)

cl = cldata['Cl'] 
mask = np.array(np.isfinite(cl)) #bool series to filter NaN values
age = 4
dcinit = 0.01 #initial estimate for the optimization

fig,ax1 = plt.subplots(1,1,figsize = (6,3))

cl1 = cl.first_valid_index() #index of surface chloride
cl = np.array(cl)
dep = np.array(cldata['Depth'])

def dccalc(dcinit,optimize=True):
    cxtp = (cl[cl1] * (1-(erf((dep - dep[cl1]) / (2 * math.sqrt(dcinit*age))))))
    if(optimize==True):
        return np.array(np.nansum((cxtp - cl)**2))
    else:
        print("Estimate",np.round(dcinit,decimals=4),np.around(cxtp[pd.notna(cl)],decimals=1),np.around(np.nansum((cxtp - cl)**2),decimals=2))
        return cxtp

#print("Chloride Profile",np.around(cl[pd.notna(cl)],2))
result = least_squares(fun=dccalc,x0=dcinit,args=[True])
print("Diffusion Coeff:",np.around(result.x[0],decimals=3),"Dist Unit^2 / year")
print(result.message)
print("Success:",result.success)
print("SSE:",result.fun)

t2 = time.perf_counter()
print("Solved in...",round(t2-t1,3),"sec")

ax1.plot(cl[mask],dep[mask],'ro',linestyle='-',label='Actual',alpha=0.8)
ax1.plot(dccalc(result.x,False),dep[mask],'go',linestyle='--',label='Estimated',alpha=0.8)
ax1.set_ylim([5,0])
ax1.grid()
ax1.legend()
plt.show()

t3 = time.perf_counter()
print("Plotted in...",round(t3-t2,3),"sec")
