# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:41:14 2021

@author: al-abiad
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from CPclass import phone as CP

def calculate_norm_accandgyro(gyro=None,acc=None):
    matrix1=acc
    x=matrix1.iloc[:,0].values**2
    y=matrix1.iloc[:,1].values**2
    z=matrix1.iloc[:,2].values**2
    m=x+y+z
    mm=np.array([np.sqrt(i) for i in m])
    acc_magnitude=mm
    
    matrix2=gyro
    x=matrix2.iloc[:,0].values**2
    y=matrix2.iloc[:,1].values**2
    z=matrix2.iloc[:,2].values**2
    m=x+y+z
    mm=np.array([np.sqrt(i) for i in m])
    gyro_magnitude=mm
    
    return(acc_magnitude,gyro_magnitude)

def detectstartofwalk(sig1,thresh=3): 
    i=0
    N_wf=128
    condition=True    
    while condition==True:
        mag=sig1[i:i+N_wf]
        mag=mag-np.mean(mag)
        ener=np.max(mag**2)
        if ener>thresh:
            while condition==True:
                mag=sig1[i:i+N_wf//10]
                ener=np.max(mag**2)
                if ener>thresh:
                    startp=i
                    print(i)
                    condition=False
                i=i+1
        i=i+1
        
    return(startp)

#gait up
path="d:\\Users\\al-abiad\\Desktop\\Enguerran_test\\Jeunes\\Test jeune 1\\Gaitup"

#hand
filename=os.path.join(path,"hand.csv" ) 
col_list = ["Time", "Gyro X","Gyro Y","Gyro Z","Accel X", "Accel Y","Accel Z"]
df_h=pd.read_csv(filename, delimiter=",",skiprows=[0],usecols=col_list)
df_h = df_h.iloc[1:]
df_h=df_h.astype('float32')
acc_h=df_h.iloc[:,4:7]
gyro_h=df_h.iloc[:,1:4]

acc_h_magnitude,gyro_h_magnitude=calculate_norm_accandgyro(gyro=gyro_h,acc=acc_h)

plt.plot(acc_h_magnitude)

s_h=detectstartofwalk(acc_h_magnitude,thresh=0.22)

#feet
filename=os.path.join(path,"left_foot.csv" ) 
col_list = ["Time", "Gyro X","Gyro Y","Gyro Z","Accel X", "Accel Y","Accel Z"]
df_lf=pd.read_csv(filename, delimiter=",",skiprows=[0],usecols=col_list)
df_lf = df_lf.iloc[1:]
df_lf=df_lf.astype('float32')

acc_lf=df_lf.iloc[:,4:7]
gyro_lf=df_lf.iloc[:,1:4]

acc_lf_magnitude,gyro_lf_magnitude=calculate_norm_accandgyro(gyro=gyro_lf,acc=acc_lf)

plt.plot(acc_lf_magnitude)

s_lf=detectstartofwalk(acc_lf_magnitude,thresh=5)

filename=os.path.join(path,"right_foot.csv" ) 
col_list = ["Time", "Gyro X","Gyro Y","Gyro Z","Accel X", "Accel Y","Accel Z"]
df_rf=pd.read_csv(filename, delimiter=",",skiprows=[0],usecols=col_list)
df_rf = df_rf.iloc[1:]
df_rf=df_rf.astype('float32')

acc_rf=df_rf.iloc[:,4:7]
gyro_rf=df_rf.iloc[:,1:4]

acc_rf_magnitude,gyro_rf_magnitude=calculate_norm_accandgyro(gyro=gyro_rf,acc=acc_rf)

plt.plot(acc_rf_magnitude)

s_rf=detectstartofwalk(acc_rf_magnitude,thresh=5)

s_f=np.minimum(s_rf,s_lf)



df_lf=df_lf.iloc[s_f:,:].copy()
df_lf=df_lf.reset_index(drop=True)
df_rf=df_rf.iloc[s_f:,:].copy()
df_rf=df_rf.reset_index(drop=True)
df_h=df_h.iloc[s_h:,:].copy()
df_h=df_h.reset_index(drop=True)


acc_h=df_h.iloc[:,4:7]
gyro_h=df_h.iloc[:,1:4]
acc_h_magnitude,gyro_h_magnitude=calculate_norm_accandgyro(gyro=gyro_h,acc=acc_h)
plt.plot(gyro_h_magnitude)

acc_rf=df_rf.iloc[:,4:7]
gyro_rf=df_rf.iloc[:,1:4]
acc_rf_magnitude,gyro_rf_magnitude=calculate_norm_accandgyro(gyro=gyro_rf,acc=acc_rf)

acc_lf=df_lf.iloc[:,4:7]
gyro_lf=df_lf.iloc[:,1:4]

acc_lf_magnitude,gyro_lf_magnitude=calculate_norm_accandgyro(gyro=gyro_lf,acc=acc_lf)

plt.plot(acc_lf_magnitude)
plt.plot(acc_rf_magnitude)    

#excel right HS
path="d:\\Users\\al-abiad\\Desktop\\Enguerran_test\\Jeunes\\Test jeune 1\\Gaitup"

filename=os.path.join(path,"J1_right_noturn.xlsx" ) 
sheet_right = pd.read_excel(filename)

start_time_r=sheet_right['Unnamed: 1'].iloc[12] 
stop_time_r=sheet_right['Unnamed: 3'].iloc[12]

HS_r=sheet_right['Unnamed: 3'].iloc[24:]
HS_r=HS_r.iloc[:HS_r.last_valid_index()-24]+start_time_r


#excel left HS
filename=os.path.join(path,"J1_left_noturn.xlsx" ) 
sheet_left = pd.read_excel(filename)

start_time_l=sheet_left['Unnamed: 1'].iloc[12] 
stop_time_l=sheet_left['Unnamed: 3'].iloc[12]

HS_l=sheet_left['Unnamed: 3'].iloc[24:]
HS_l=HS_l.iloc[:HS_l.last_valid_index()-24]+start_time_l

#index of HS_l

# it is okay we can choose any foot
time=df_lf['Time'].values

HS_l_index=[]
for hs in HS_l.values:
    t1=np.where(time>hs)[0][0]
    t2=t1-1
    if np.abs(hs-time[t1])<=np.abs(hs-time[t2]):
        HS_l_index.append(t1)
    else:
        HS_l_index.append(t2)
    
HS_r_index=[]
for hs in HS_r.values:
    t1=np.where(time>hs)[0][0]
    t2=t1-1
    if np.abs(hs-time[t1])<=np.abs(hs-time[t2]):
        HS_r_index.append(t1)
    else:
        HS_r_index.append(t2)

plt.plot(gyro_h_magnitude)
plt.scatter(HS_l_index,gyro_h_magnitude[HS_l_index])
plt.scatter(HS_r_index,gyro_h_magnitude[HS_r_index])

#phone
path="d:\\Users\\al-abiad\\Desktop\\Enguerran_test\\Jeunes\\Test jeune 1\\telephone\\02-03-2021\\14-40-13-417"
CP_data=CP(path,app="geoloc")
CP_data.interpolategyrnacc(fs=128)
CP_data.rad2deg()
CP_data.filter_data(acc=CP_data.acc_interp,gyro=CP_data.gyro_interp,N=10,fc=3,fs=128)
CP_data.calculate_norm_accandgyro(gyro=CP_data.gyro_filtered,acc=CP_data.acc_filtered)
plt.plot(CP_data.acc_magnitude)
s_w=detectstartofwalk(CP_data.acc_magnitude,thresh=2)

#crop
CP_data.acc_magnitude=CP_data.acc_magnitude[s_w:]

plt.plot(CP_data.acc_magnitude)

plt.scatter(HS_l_index,CP_data.acc_magnitude[HS_l_index])


plt.scatter(HS_r_index,CP_data.acc_magnitude[HS_r_index])















