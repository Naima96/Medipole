# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:15:26 2021

@author: al-abiad
"""

import pyCompare
import numpy as np
import pandas as pd


pyCompare.blandAltman(GU_P,Phon)


pyCompare.blandAltman(GU_H,hand)

pyCompare.blandAltman(Cov_P,Cov_GU)


pyCompare.blandAltman(gaitup_hand_summary,hand_cov)

pyCompare.blandAltman(fractal_foot2,fractal_waist2)

pyCompare.blandAltman(gaitup_foot_fractal_hand,
                      hand_fractal)





judges=np.hstack([["gaitup"]*21,["Phone"]*21])
Measurement=np.hstack([gaitup_phone_cov,phone_cov])
Exam=np.hstack([np.arange(0,21),np.arange(0,21)])
x={}
x["judges"]=judges
x["Measurement"]=Measurement
x["Exam"]=Exam



data=pd.DataFrame(x)
icc2 = pg.intraclass_corr(data=data, 
                         targets='Exam',
                         raters='judges', 
                         ratings='Measurement')


judges=np.hstack([["gaitup"]*21,["Phone"]*21])
Measurement=np.hstack([gaitup_phone_SD,phone_SD])
Exam=np.hstack([np.arange(0,21),np.arange(0,21)])
x={}
x["judges"]=judges
x["Measurement"]=Measurement
x["Exam"]=Exam



data=pd.DataFrame(x)
icc = pg.intraclass_corr(data=data, 
                         targets='Exam',
                         raters='judges', 
                         ratings='Measurement')


judges=np.hstack([["gaitup"]*321,["Phone"]*321])
Measurement=np.hstack([Stride_time_gaitup,Stride_time_phone])
Exam=np.hstack([np.arange(0,321),np.arange(0,321)])
x={}
x["judges"]=judges
x["Measurement"]=Measurement
x["Exam"]=Exam



data=pd.DataFrame(x)
icc = pg.intraclass_corr(data=data, 
                         targets='Exam',
                         raters='judges', 
                         ratings='Measurement')


judges=np.hstack([["gaitup"]*280,["Phone"]*280])
Measurement=np.hstack([Stridetime_Gaitup_hand,hand_stride_time])
Exam=np.hstack([np.arange(0,280),np.arange(0,280)])
x={}
x["judges"]=judges
x["Measurement"]=Measurement
x["Exam"]=Exam



data=pd.DataFrame(x)
icc = pg.intraclass_corr(data=data, 
                         targets='Exam',
                         raters='judges', 
                         ratings='Measurement')


judges=np.hstack([["gaitup"]*10307,["Phone"]*10307])
Measurement=np.hstack([Stride_time_inst_phone,Stride_time_inst_phone_gaitup])
Exam=np.hstack([np.arange(0,10307),np.arange(0,10307)])
x={}
x["judges"]=judges
x["Measurement"]=Measurement
x["Exam"]=Exam



data=pd.DataFrame(x)
icc = pg.intraclass_corr(data=data, 
                         targets='Exam',
                         raters='judges', 
                         ratings='Measurement')


RMSE=1000*((np.mean((Stride_time_inst_phone-Stride_time_inst_phone_gaitup)**2))**(0.5))


judges=np.hstack([["gaitup"]*8887,["Phone"]*8887])
Measurement=np.hstack([Hand_stride_time_inst,Gaitup_hand_stride_time_inst])
Exam=np.hstack([np.arange(0,8887),np.arange(0,8887)])
x={}
x["judges"]=judges
x["Measurement"]=Measurement
x["Exam"]=Exam



data=pd.DataFrame(x)
icc = pg.intraclass_corr(data=data, 
                         targets='Exam',
                         raters='judges', 
                         ratings='Measurement')


RMSE=1000*((np.mean((Hand_stride_time_inst-Gaitup_hand_stride_time_inst)**2))**(0.5))








import matplotlib.pyplot as plt
import numpy as np

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    
    print(1.96*sd)
plt.figure()   
bland_altman_plot(gaitup_foot_fractal_hand, hand_fractal)
