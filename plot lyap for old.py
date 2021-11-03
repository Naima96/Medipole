# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:48:36 2021

@author: al-abiad
"""
ML_phonew_low=np.array([1.324,1.34,1.32])
AP_phonew_low=np.array([1.683,1.49,1.3])
V_phonew_low=np.array([1.68,1.49,1.3])

arr_AP=np.array([1.14,1.09,1.17])

arr_ML=np.array([1.28,1.22,1.21])

arr_V=np.array([1.46,1.213,1.1])

x=np.array([0,3,6])
labels2 = ['S1', 'S2','S3']

fig, ax = plt.subplots(1,1,sharey=True)
rects1 = ax.bar(x - width/2, arr_ML,width, label='ML')
rects2 = ax.bar(x + width/2,  arr_AP, width, label='AP',color='darkorange')
rects3 = ax.bar(x + 1.5*width, arr_V, width, label='V',color='r')
autolabel(rects1,ax,0)
autolabel(rects2,ax,0.0)
autolabel(rects3,ax,0.3)
plt.legend()
ax.set_xticks(x)
ax.set_xticklabels(labels2)


arr_ML=np.array([1.21,1.2,1.13])

arr_AP=np.array([1.13,1.12,1.08])

arr_V=np.array([1.36,1.11,1.14])

arr_MLm=np.array([0.1,0.1,0.1])

arr_APp=np.array([0.08,0.08,0.1])

arr_Vv=np.array([0.16,0.09,0.09])

x=np.array([0,3,6])
labels2 = ['L', 'C','H']

fig, ax = plt.subplots(1,1,sharey=True)
rects1 = ax.bar(x - width/2, arr_ML,width, yerr=arr_MLm,label='ML')
rects2 = ax.bar(x + width/2,  arr_AP, width, label='AP',yerr=arr_APp,color='darkorange')
rects3 = ax.bar(x + 1.5*width, arr_V, width, label='V',yerr=arr_MLm,color='r')
autolabel(rects1,ax,0)
autolabel(rects2,ax,0.0)
autolabel(rects3,ax,0.3)
plt.legend()
ax.set_xticks(x)
ax.set_xticklabels(labels2)

