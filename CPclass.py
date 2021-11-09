# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:56:34 2020

@author: al-abiad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal,fft
from scipy.signal import find_peaks
import sys
import os
from scipy import stats
from pickle import load
import joblib
class phone(object):
    '''
    create phone object interpolated and filtered and aligned with treadmill data 
    '''
    def __init__(self, filename="",app="sensor play",t="calib",acc=[],gyro=[],allsensors=False):
        """
        create phone object 
        :param str filename: .csv file 
        """
        #---READ csv FILE---
        if(app=="manual_entry"):
            
            self.acc_interp=acc
            self.gyro_interp=gyro
            
        if(app=="PhysicsToolbox"):
            path="d:\\Users\\al-abiad\\Desktop\\montreal gaitup result\\Projet MBAM\\Data PhysicsToolboxSuite"
            filename=os.path.join(path,"2019-12-0315.48.10 S02 POST MAX.csv ")
            df=pd.read_csv(filename, delimiter=";",decimal=',')
            time=pd.to_datetime(df['time'],format="%H:%M:%S:%f")#change string to datetime
            timestamp= np.array([i.timestamp()for i in time])
            timestamp=(timestamp-timestamp[0]).astype('float64')
            acceleration= df.iloc[:,1:4]
            acceleration.index=timestamp
            acceleration = acceleration[~acceleration.index.duplicated()]
        
        if (app=="geoloc"):
            path=filename
            filename=os.path.join(path,"ACC.txt" )
            df=pd.read_csv(filename, delimiter=",")
            time_acc=df["Time [s]"].values
            time_start_acc=time_acc[0]
            acce_uncalib=df.iloc[:,1:4]
            acce_calib=df.iloc[:,4:7]
            acce_uncalib.index=time_acc
            acce_calib.index=time_acc

            
            filename=os.path.join(path,"GYRO.txt" )
            df=pd.read_csv(filename, delimiter=",")
            time_gyro=df["Time(s)"].values
            time_start_gyro=time_gyro[0]
            gyro_uncalib=df.iloc[:,1:4]
            gyro_calib=df.iloc[:,4:7]
            
            gyro_uncalib.index=time_gyro
            gyro_calib.index=time_gyro
            
            time_start=np.minimum(time_start_gyro,time_start_acc)
            
            
            

            acce_calib = acce_calib[~acce_calib.index.duplicated()]
            acce_uncalib = acce_uncalib[~acce_uncalib.index.duplicated()]

            gyro_calib = gyro_calib[~gyro_calib.index.duplicated()]
            gyro_uncalib = gyro_uncalib[~gyro_uncalib.index.duplicated()]
            
            
            acce_calib=acce_calib[acce_calib.index>time_start]
            acce_uncalib=acce_uncalib[acce_uncalib.index>time_start]
            
            gyro_calib=gyro_calib[gyro_calib.index>time_start]
            gyro_uncalib=gyro_uncalib[gyro_uncalib.index>time_start]
            
            acce_calib.index=acce_calib.index-acce_calib.index[0]
            
            acce_uncalib.index=acce_uncalib.index-acce_uncalib.index[0]
            
            gyro_calib.index=gyro_calib.index-gyro_calib.index[0]
            gyro_uncalib.index=gyro_uncalib.index-gyro_uncalib.index[0]
            
            filename=os.path.join(path,"MAG.txt" )
            df=pd.read_csv(filename, delimiter=",")
            time=df["Time(s)"].values
            time=time-time[0]
            mag_uncalib=df.iloc[:,1:4]
            mag_calib=df.iloc[:,4:7]
            mag_uncalib.index=time
            mag_calib.index=time
            mag_calib = mag_calib[~mag_calib.index.duplicated()]
            mag_uncalib = mag_uncalib[~mag_uncalib.index.duplicated()]
            
            if t=="calib":
                self.acc_rawdata=acce_calib
                self.gyro_rawdata=gyro_calib
        elif filename!="":
            df=pd.read_csv(filename, delimiter=",")   
            if (app=="sensor play"):
                time=pd.to_datetime(df['Timestamp'])#change string to datetime
                timestamp= np.array([datetime.timestamp(i)for i in time])#change to timestamp
                acceleration= df.iloc[:,1:4] #unit is g
                gyroscope=df.iloc[:,7:10] #unit is rad /sec
                Altitude=df.iloc[:,10:13]
                Quaternion=df.iloc[:,23:26]
            elif(app=="sensor record"):
                timestamp=df['time']
                acceleration= df.iloc[:,1:4] #unit is g
                gyroscope=df.iloc[:,4:7] #unit is rad /sec
            acceleration.index=timestamp[:]-timestamp[0]
            gyroscope.index=timestamp[:]-timestamp[0]
            
            if allsensors:
                Altitude.index=timestamp[:]-timestamp[0]
                Quaternion.index=timestamp[:]-timestamp[0]
            
            acceleration = acceleration[~acceleration.index.duplicated()]
            gyroscope = gyroscope[~gyroscope.index.duplicated()]
            
            if allsensors:
                Altitude = Altitude[~Altitude.index.duplicated()]
                Quaternion = Quaternion[~Quaternion.index.duplicated()]
            #index.microsecond .second .minute .hour to give time specifically
            self.acc_rawdata=acceleration
            self.gyro_rawdata=gyroscope
            
            if allsensors:
                self.altitude_rawdata=Altitude
                self.quaternion=Quaternion
        
    def interpolategyrnacc(self,fs=100,allsensors=False):
        """
        interpolate to fs 100 Hz
        :param int fs: sampling frequency (default 100Hz)
        """
        #---Interpolate acceleration and gyroscope---
        matrix1=self.acc_rawdata
        matrix2=self.gyro_rawdata
        if allsensors:
            matrix3=self.altitude_rawdata
            matrix4=self.quaternion
        t_t=np.linspace(0, matrix1.index[len(matrix1)-1], num=np.int((matrix1.index[len(matrix1)-1])*fs), endpoint=True,dtype=np.float32)
        matrix1=matrix1.reindex(matrix1.index.union(t_t))
        matrix2=matrix2.reindex(matrix2.index.union(t_t))
        
        if allsensors:
            matrix3=matrix3.reindex(matrix3.index.union(t_t))
            matrix4=matrix4.reindex(matrix4.index.union(t_t))
        
        matrix1=matrix1.interpolate(method='linear', limit_direction='both', axis=0)
        matrix2=matrix2.interpolate(method='linear', limit_direction='both', axis=0)
        
        if allsensors:
            matrix3=matrix3.interpolate(method='linear', limit_direction='both', axis=0)
            matrix4=matrix4.interpolate(method='linear', limit_direction='both', axis=0)
        
        matrix1=matrix1[matrix1.index.isin(pd.Index(t_t))]
        matrix2=matrix2[matrix2.index.isin(pd.Index(t_t))]
        
        if allsensors:
            matrix3=matrix3[matrix3.index.isin(pd.Index(t_t))]
            matrix4=matrix4[matrix4.index.isin(pd.Index(t_t))]
        
        matrix1.index=np.around(matrix1.index.values.astype('float64'),decimals=4)
        matrix2.index=np.around(matrix2.index.values.astype('float64'),decimals=4)
        
        if allsensors:
            matrix3.index=np.around(matrix3.index.values.astype('float64'),decimals=4)
            matrix4.index=np.around(matrix4.index.values.astype('float64'),decimals=4)
        
        self.acc_interp=matrix1
        self.gyro_interp=matrix2
        
        if allsensors:
            self.altitude_interp=matrix3
            self.quaternion_interp=matrix4
    
    def gtom2s(self):
        """
        Change unit of accelerometer signal into m/s^2
        """
        self.acc_interp= self.acc_interp.mul(9.8)
        return()
    
    def rad2deg(self):
        """
        Change unit of gyroscope signal into deg/sec
        """
        c=180/np.pi
        self.gyro_interp= self.gyro_interp.mul(c)
        return()
    
    def deg2rad(self):
        """
        Change unit of gyroscope signal into deg/sec
        """
        c=np.pi/180
        self.gyro_interp= self.gyro_interp.mul(c)
        return()
        
    
    
    def plot_phases(self,TM,weight,zoom=True,events=1,part="pocket"):
        """
        plot the data into three parts depending on the cellphone holding position
        :param object TM: treadmill object
        :param int weight: weight of person
        :param bool zoom: zoom on the plot (default True)
        :param int fs: sampling frequency (default 600Hz)
        :param int start: start of walking after the jump (default 30sec)
        """
        plt.rcParams.update({'font.size': 22})
        
        if events==1:
                    if part=="hand":
                        
                        self.filter_data(acc=self.acc_hand,gyro=self.gyro_hand)#,N=10,fc=3,fs=100)#change
                        self.calculate_norm_accandgyro(gyro=self.gyro_filtered,acc=self.acc_filtered)
                        self.find_dominantfreq(fs=100,N_wf=512,overlap=512)
                        self.detect_mode()
                        self.detect_steps(N_wf=512,overlap=512,remove=True)
                        allsteps=self.allsteps
                        
                        fig, ax = plt.subplots(2)
                        color = 'tab:blue'
                        ax[0].title.set_text("Smartphone gyroscope")
                        #--- plot hand gyroscope---
                #        ax[0].set_xlabel('time (s)')
                        ax[0].set_ylabel('T_acc m/s\u00b2', color=color)
                        TM.calculate_sumforceandmoment(phase="hand")
                        TM.calculate_normforceandmomemt()
                        TM.calculategaitevents(fs=100,weight=weight,P=0.5,phase="hand")
                        ax[0].plot(TM.force_norm/weight, color=color,alpha=0.5)
                        ax[0].tick_params(axis='y', labelcolor=color)
                     
                        Rcycle=np.concatenate(TM.right_cycle).astype('int')[::2]
#                        Lcycle=np.concatenate(self.TM_data.left_cycle).astype('int')//(fss//100)
                        ax[0].scatter(Rcycle,TM.force_norm[Rcycle]/weight,color='k',alpha=0.5)
                        
                        if zoom:
                            ax[0].set(xlim=(4200, 5000))
                        
                            
                        ax2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
                        color = 'tab:orange'
                        ax2.set_ylabel('S_gyro rad/s', color=color)  # we already handled the x-label with ax1
                        self.filter_data(acc=self.acc_hand,gyro=self.gyro_hand,N=10,fc=3,fs=100)
                        self.calculate_norm_accandgyro(gyro=self.gyro_filtered,acc=self.acc_filtered)
                        ax2.plot(self.gyro_magnitude, color=color)
                        ax2.tick_params(axis='y', labelcolor=color)
                        ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax[0].get_yticks())))
                #        fig.tight_layout() 
                        ax2.grid(None)
                        ax2.set_xticklabels([])
                        ax2.scatter(allsteps,self.gyro_magnitude[allsteps],color='k')
                        
                        #--- plot hand acceleration---
                        
                        ax[1].title.set_text("Smartphone accelerometer")
                        color = 'tab:blue'
                        ax[1].set_xlabel('Time')
                        ax[1].set_ylabel('T_acc m/s\u00b2', color=color)
                        ax[1].plot(TM.force_norm/weight, color=color,alpha=0.5)
                        ax[1].tick_params(axis='y', labelcolor=color)
                        ax[1].scatter(Rcycle,TM.force_norm[Rcycle]/weight,color='k',alpha=0.5)
                        if zoom:
                            ax[1].set(xlim=(4200, 5000))
                            
                        ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
                        
                        color = 'tab:orange'
                        ax2.set_ylabel('S_acc m/s\u00b2', color=color)  # we already handled the x-label with ax1
                        ax2.plot(self.acc_magnitude, color=color)
                        ax2.tick_params(axis='y', labelcolor=color)
                        ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax[0].get_yticks())))
                #        fig.tight_layout() 
                        ax2.grid(None)
                        ax2.scatter(allsteps,self.acc_magnitude[allsteps],color='k')
                        
                        ax2.set_xticklabels([])
                        
                        
                        plt.figure()
                        plt.plot(self.mode['texting'])
                        plt.plot(self.mode['swinging'])
                        
                    if part=="waist":
                        self.filter_data(acc=self.acc_waist,gyro=self.gyro_waist,N=10,fc=3,fs=100)#change
                        self.calculate_norm_accandgyro(gyro=self.gyro_filtered,acc=self.acc_filtered)
                        
                        self.peakdet_m2(acc=True,plot_peak=True)#change
                        TM.calculategaitevents(fs=100,weight=weight,P=0.5,phase="waist")
                        allsteps=self.peakandvalley['peak_index'].astype('int')
                        
                        fig, ax = plt.subplots(2)
                        color = 'tab:blue'
                        
                        ax[0].title.set_text("Smartphone gyroscope")
                        
                        #--- plot waist gyroscope---
                #        ax[0].set_xlabel('time (s)')
                        ax[0].set_ylabel('T_acc m/s\u00b2', color=color)
                        TM.calculate_sumforceandmoment(phase="waist")
                        TM.calculate_normforceandmomemt()
                        ax[0].plot(TM.force_norm/weight, color=color,alpha=0.5)
                        ax[0].tick_params(axis='y', labelcolor=color)
                        
                        Rcycle=np.concatenate(TM.right_cycle).astype('int')[::2]
#                        Lcycle=np.concatenate(self.TM_data.left_cycle).astype('int')//(fss//100)
                        ax[0].scatter(Rcycle,TM.force_norm[Rcycle]/weight,color='k',alpha=0.5)
                        
                        if zoom:
                            ax[0].set(xlim=(4200, 5000))
                        ax2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
                        
                        color = 'tab:orange'
                        ax2.set_ylabel('S_gyro rad/s', color=color)  # we already handled the x-label with ax1
                        self.filter_data(acc=self.acc_waist,gyro=self.gyro_waist,N=10,fc=3,fs=100)
                        self.calculate_norm_accandgyro(gyro=self.gyro_filtered,acc=self.acc_filtered)
                        ax2.plot(self.gyro_magnitude, color=color)
                        ax2.tick_params(axis='y', labelcolor=color)
                        ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax[0].get_yticks())))
                #        fig.tight_layout() 
                        ax2.grid(None)
                        ax2.set_xticklabels([])
                        ax2.scatter(allsteps,self.gyro_magnitude[allsteps],color='k')
                        ax2.set_xticklabels([])
                        #--- plot waist acceleration---
                        ax[1].title.set_text("Smartphone accelerometer")
                        color = 'tab:blue'
                        ax[1].set_xlabel('Time')
                        ax[1].set_ylabel('T_acc m/s\u00b2', color=color)
                        ax[1].plot(TM.force_norm/weight, color=color,alpha=0.5)
                        ax[1].tick_params(axis='y', labelcolor=color)
                        ax[1].scatter(Rcycle,TM.force_norm[Rcycle]/weight,color='k',alpha=0.5)
                        ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
                        if zoom:
                            ax[1].set(xlim=(4200, 5000))
                        color = 'tab:orange'
                        ax2.set_ylabel('S_acc m/s\u00b2', color=color)  # we already handled the x-label with ax1
                        ax2.plot(self.acc_magnitude, color=color)
                        ax2.tick_params(axis='y', labelcolor=color)
                        ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax[0].get_yticks())))
                #        fig.tight_layout() 
                        ax2.grid(None)
                        
                        ax2.scatter(allsteps,self.acc_magnitude[allsteps],color='k')
                        ax2.set_xticklabels([])
                        
                    if part=="pocket":
                        
                        self.filter_data(acc=self.acc_pocket,gyro=self.gyro_pocket,N=10,fc=3,fs=100)#change
                        self.calculate_norm_accandgyro(gyro=self.gyro_filtered,acc=self.acc_filtered)
                        
                        self.peakdet_m2(acc=False,plot_peak=False)#change
                        allsteps=self.peakandvalley['peak_index'].astype('int')
                        
                        fig, ax = plt.subplots(2)
                        ax[0].title.set_text("Smartphone gyroscope")
                        color = 'tab:blue'
                        TM.calculategaitevents(fs=100,weight=weight,P=0.5,phase="pocket")
                        Rcycle=np.concatenate(TM.right_cycle).astype('int')[::2]
#                        Lcycle=np.concatenate(self.TM_data.left_cycle).astype('int')//(fss//100)
                        
                        
                        #--- plot pocket gyroscope---
                #        ax[0].set_xlabel('time (s)')
                        ax[0].set_ylabel('T_acc m/s\u00b2', color=color)
                        TM.calculate_sumforceandmoment(phase="pocket")
                        TM.calculate_normforceandmomemt()
                        ax[0].plot(TM.force_norm/weight, color=color,alpha=0.5)
                        ax[0].tick_params(axis='y', labelcolor=color)
                        ax[0].scatter(Rcycle,TM.force_norm[Rcycle]/weight,color='k',alpha=0.5)
                        
                        if zoom:
                            ax[0].set(xlim=(6200, 7000))
                        ax2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
                        
                        
                        color = 'tab:orange'
                        ax2.set_ylabel('S_gyro rad/s', color=color)  # we already handled the x-label with ax1
                        self.filter_data(acc=self.acc_pocket,gyro=self.gyro_pocket,N=10,fc=3,fs=100)
                        self.calculate_norm_accandgyro(gyro=self.gyro_filtered,acc=self.acc_filtered)
                        ax2.plot(self.gyro_magnitude, color=color)
                        ax2.tick_params(axis='y', labelcolor=color)
                        ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax[0].get_yticks())))
                        ax2.scatter(allsteps,self.gyro_magnitude[allsteps],color='k')
                #        fig.tight_layout() 
                        ax2.grid(None)
                        ax2.set_xticklabels([])
                        #--- plot pocket acceleration---
                        ax[1].title.set_text("Smartphone accelerometer")
                        color = 'tab:blue'
                        ax[1].set_xlabel('Time')
                        ax[1].set_ylabel('T_acc m/s\u00b2', color=color)
                        ax[1].plot(TM.force_norm/weight, color=color,alpha=0.5)
                        ax[1].tick_params(axis='y', labelcolor=color)
                        ax[1].scatter(Rcycle,TM.force_norm[Rcycle]/weight,color='k',alpha=0.5)
                        ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
                        
                        if zoom:
                            ax[1].set(xlim=(6200, 7000))
                        color = 'tab:orange'
                        ax2.set_ylabel('S_acc m/s\u00b2', color=color)  # we already handled the x-label with ax1
                        ax2.plot(self.acc_magnitude, color=color)
                        ax2.tick_params(axis='y', labelcolor=color)
                        ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax[0].get_yticks())))
                #        fig.tight_layout() 
                        ax2.grid(None)
                        ax2.scatter(allsteps,self.acc_magnitude[allsteps],color='k')
                        ax2.set_xticklabels([])
        
        return()
        
    def filter_data(self,gyro=[0],acc=[0],altitude=[0],quaternion=[0],N=4,fc=15,fs=100):
        """
        filter the gyro and acc data by a low pass butterworth filter using interpolated data or aligned data
        :param int gryo: gyroscope signal dataframe
        :param int acc: acceleration signal dataframe
        :param int fc: cut off frequency (default 15Hz)
        :param int fs: sampling frequency (default 600Hz)
        :param int N: order of filter (default 4) 
        returns attributes acc_filtered and gyro_filtered
        """
        Wn =fc/(fs/2) # Cutoff frequency normalized 
        B, A = signal.butter(N, Wn,'low', output='ba') 
        
        if len(acc)!=1:
            y=acc.copy()
            for col in y:
                y[col]=signal.filtfilt(B, A, y[col].values) # fix
            self.acc_filtered=y
            
        if len(gyro)!=1:
            z=gyro.copy()
            for col in z:
                z[col]=signal.filtfilt(B, A, z[col].values) # fix
            self.gyro_filtered=z
            
        if len(altitude)!=1:
            x=altitude.copy()
            for col in x:
                x[col]=signal.filtfilt(B, A, x[col].values) # fix      
            self.altitude_filtered=x
            
        if len(quaternion)!=1:
            w=quaternion.copy()
            for col in w:
                w[col]=signal.filtfilt(B, A, w[col].values) # fix
            self.quaternion_filtered=w
        return()
    
    def calculate_norm_accandgyro(self,gyro=None,acc=None):
        """
        calculate the norm of acc and gyro from the  acc data and gyro data adds the attributes
        acc_magnitude and gyro_magnitude.
        :param dataframe gryo: gyroscope signal dataframe
        :param dataframe acc: acceleration signal dataframe
        """
        matrix1=acc
        x=matrix1.iloc[:,0].values**2
        y=matrix1.iloc[:,1].values**2
        z=matrix1.iloc[:,2].values**2
        m=x+y+z
    #    m=np.sqrt(m)
        mm=np.array([np.sqrt(i) for i in m])
        self.acc_magnitude=mm
        
        matrix2=gyro
        x=matrix2.iloc[:,0].values**2
        y=matrix2.iloc[:,1].values**2
        z=matrix2.iloc[:,2].values**2
        m=x+y+z
    #    m=np.sqrt(m)
        mm=np.array([np.sqrt(i) for i in m])
        self.gyro_magnitude=mm
        
    def align_phonetreadmill_correlation(self,normforce_tread,plot=True,weight=60,remove_initiationtermination=True,period=60,fs=100,mlag=0,term_period=0):
        """
        align the treadmill downsampled norm force with cellphone data. adds the attribute acc_aligned, gyro_aligned.
        The aligned data is a cropped interpolated data
        :param array normforce_tread: norm of force from treadmill
        :param bool plot: plot the alignment (default true)
        :param int weight: weight of person
        :param bool remove_initiationtermination: remove initiation and termination phase from phone dataframe
        :param int period: initiation period to remove
        :param int fs: sampling frequency
        :param int mlag: manual lag detected visually
        :param int term_period: termination period to remove
        """
        magtread=normforce_tread/weight
        magphone=self.acc_magnitude
        if mlag==0:
            lag=np.argmax(signal.correlate(magtread,magphone))
            lagInd=np.arange(-np.max([len(magtread),len(magphone)]),np.max([len(magtread),len(magphone)]))
            lag=lagInd[lag]
    #        if lag>0:
    #            lag=np.argmax(signal.correlate(magphone,magtread))
    #            lagInd=np.arange(-np.max([len(magtread),len(magphone)]),np.max([len(magtread),len(magphone)]))
    #            lag=lagInd[lag]
            
            self.lag=lag
        else:
            lag=mlag
    
        y=np.arange(lag,len(magphone)+lag)


#        x=pd.Index(x)
#        self.acc_aligned=self.acc_interp.copy()
#        self.acc_aligned.set_index(x)
#        self.gyro_aligned=self.gyro_interp.copy()
#        self.gyro_aligned.set_index(x)
        self.lag=lag
        if lag<0:
            #go back to this
            self.acc_aligned=self.acc_interp.iloc[-lag:len(normforce_tread)-lag,:].copy()
#            self.acc_aligned=self.acc_interp.set_index(self.acc_interp.index-self.acc_interp.index[0])
            self.gyro_aligned=self.gyro_interp.iloc[-lag:len(normforce_tread)-lag,:].copy()
#            self.gyro_aligned=self.gyro_interp.set_index(self.gyro_interp.index-self.gyro_interp.index[0])
            self.lag=lag
            self.acc_magnitude=self.acc_magnitude[-lag:len(normforce_tread)-lag]
            self.gyro_magnitude=self.gyro_magnitude[-lag:len(normforce_tread)-lag]
            
        else:

            self.acc_aligned=self.acc_interp.copy()
            z=pd.DataFrame(np.zeros((lag,3)),columns=self.acc_aligned.columns)
            self.acc_aligned=z.append(self.acc_aligned,ignore_index=True)
#            self.acc_aligned=self.acc_interp.set_index(self.acc_interp.index-self.acc_interp.index[0])
            self.gyro_aligned=self.gyro_interp.copy()
            x=pd.DataFrame(np.zeros((lag,3)),columns=self.gyro_aligned.columns)
            self.gyro_aligned=x.append(self.gyro_aligned,ignore_index=True)
#            self.gyro_aligned=self.gyro_interp.set_index(self.gyro_interp.index-self.gyro_interp.index[0])
            self.lag=lag
#            self.acc_magnitude=self.acc_magnitude[lag:len(normforce_tread)+lag]
#            self.gyro_magnitude=self.gyro_magnitude[lag:len(normforce_tread)+lag]
            if len(self.acc_aligned)<len(magtread):
                length=len(magtread)-len(self.acc_aligned)
                z=pd.DataFrame(np.zeros((length,3)),columns=self.acc_aligned.columns)
                self.acc_aligned=self.acc_aligned.append(z,ignore_index=True)
                x=pd.DataFrame(np.zeros((length,3)),columns=self.gyro_aligned.columns)
                self.gyro_aligned=self.gyro_aligned.append(x,ignore_index=True)
        
        if plot:
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('treadmill force/mass', color=color)
            ax1.plot(magtread, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            
            color = 'tab:orange'
            ax2.set_ylabel('cellphone acceleration', color=color)  # we already handled the x-label with ax1
            if lag>0:
                ax2.plot(y,self.acc_magnitude-np.mean(self.acc_magnitude), color=color)
            else:
                ax2.plot(self.acc_magnitude-np.mean(self.acc_magnitude), color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
#            fig.tight_layout() 
            ax2.grid(None)
            
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('treadmill force/mass', color=color)
            ax1.plot(magtread-np.mean(magtread), color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            
            color = 'tab:orange'
            ax2.set_ylabel('cellphone angular velocity', color=color)  # we already handled the x-label with ax1
            if lag>0:
                ax2.plot(y,self.gyro_magnitude-np.mean(self.gyro_magnitude), color=color)
            else:
                ax2.plot(self.gyro_magnitude-np.mean(self.gyro_magnitude), color=color)
            ax2.tick_params(axis='y', labelcolor=color)
#            fig.tight_layout() 
            ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
            ax2.grid(None)
            
        
        if remove_initiationtermination:
            self.acc_aligned=self.acc_aligned.iloc[period*fs:len(self.acc_aligned)-term_period*fs,:]
#            self.acc_aligned=self.acc_aligned.reset_index(drop=True)
            self.gyro_aligned=self.gyro_aligned.iloc[period*fs:len(self.gyro_aligned)-term_period*fs,:]
#            self.gyro_aligned=self.gyro_aligned.reset_index(drop=True)
            
#            self.acc_magnitude=self.acc_magnitude[period*fs:len(self.acc_magnitude)-period*fs]
#            self.gyro_magnitude=self.gyro_magnitude[period*fs:len(self.gyro_magnitude)-period*fs]
            
    def removeinitiationtermination(self,fs=100,init_period=30,term_period=0):
        """
        :param int init_period: initiation period sec (default 30sec)
        :param int fs: sampling frequency (default 600Hz)
        :param int term_period: termination period sec (default 30sec)
        """
        self.acc_aligned=self.acc_interp.iloc[init_period*fs:len(self.acc_interp)-term_period*fs,:]
#            self.acc_aligned=self.acc_aligned.reset_index(drop=True)
        self.gyro_aligned=self.gyro_interp.iloc[init_period*fs:len(self.gyro_interp)-term_period*fs,:]
        
        return()
            
        
            
    def alignpeakstreadmill(self,normforce_tread,plot=True,weight=60):
        """
        very bad method dont try it unless necessary 
        """
        magtread=normforce_tread
        magphone=self.acc_magnitude
        
        peakstread, _ = find_peaks(magtread, distance=300, height=1600)
        peaksphone, _ = find_peaks(magphone, distance=600, height=35)
        
        magphone=self.acc_magnitude[peaksphone[0]:len(magphone)]
#        self.gyro_magnitude=self.gyro_magnitude[peaksphone[0]:peaksphone[len(peaksphone)-1]+1]
        normforce_tread=normforce_tread[peakstread[0]:peakstread[len(peakstread)-1]+1]
    
        if plot==True:
            plt.plot(magphone)
            plt.plot(peaksphone-peaksphone[0],magphone[peaksphone-peaksphone[0]], "H",color='k')
            plt.plot(normforce_tread)
            plt.plot(peakstread-peakstread[0],normforce_tread[peakstread-peakstread[0]], "H",color='k')
        return() 
        
    def crop_phase(self,fs=100,start=30,period=210):
        """
        crop the data into three parts depending on the cellphone holding position
        :param int fs: sampling frequency (default 100Hz)
        :param int start: start of walking after the jump (default 30sec)
        
        """
        warm_up=30
#        period=210
        change=30
        #---gyroscope signal---
        #120 sec waist
        self.gyro_waist=self.gyro_aligned.iloc[(start+warm_up)*fs:(start+period+warm_up)*fs,:].copy()
        self.gyro_waist=self.gyro_waist.reset_index(drop=True)
        #120 sec pocket
        self.gyro_pocket=self.gyro_aligned.iloc[(start+period+warm_up+change)*fs:(start+2*period+warm_up+change)*fs,:].copy()
        self.gyro_pocket=self.gyro_pocket.reset_index(drop=True)
        #120 sec hand
        self.gyro_hand=self.gyro_aligned.iloc[(start+2*period+warm_up+2*change)*fs:(start+3*period+warm_up+2*change)*fs,:].copy()
        self.gyro_hand=self.gyro_hand.reset_index(drop=True)
        
        #---acceleration signal---
        #120 sec waist
        self.acc_waist=self.acc_aligned.iloc[(start+warm_up)*fs:(start+period+warm_up)*fs,:].copy()
        self.acc_waist=self.acc_waist.reset_index(drop=True)
        #120 sec pocket
        self.acc_pocket=self.acc_aligned.iloc[(start+period+warm_up+change)*fs:(start+2*period+warm_up+change)*fs,:].copy()
        self.acc_pocket=self.acc_pocket.reset_index(drop=True)
        #120 sec hand
        self.acc_hand=self.acc_aligned.iloc[(start+2*period+warm_up+2*change)*fs:(start+3*period+warm_up+2*change)*fs,:].copy()
        self.acc_hand=self.acc_hand.reset_index(drop=True)
        
        
    def crop_medipole(self,fs=100,phases=[],turn_time=0):
        ind=0
        walkingperiodsgyro=[]
        walkingperiodsacc=[]
        turn_time=turn_time*100
        for (start,stop) in phases:
            if ind==0:
                print("ind =0")
                df=self.gyro_interp.iloc[(start)*fs:(stop)*fs,:].copy()
                df=df.reset_index(drop=True)
                walkingperiodsgyro.append(df)
                
                df=self.acc_interp.iloc[(start)*fs:(stop)*fs,:].copy()
                df=df.reset_index(drop=True)
                walkingperiodsacc.append(df)
            else:
                print("ind =1")
                print(start+turn_time)
                print(start)
                df=self.gyro_interp.iloc[(start+turn_time)*fs:(stop)*fs,:].copy()
                df=df.reset_index(drop=True)
                walkingperiodsgyro.append(df)
                
                df=self.acc_interp.iloc[(start+turn_time)*fs:(stop)*fs,:].copy()
                df=df.reset_index(drop=True)
                walkingperiodsacc.append(df)
            ind=1
                
        return(walkingperiodsacc,walkingperiodsgyro)


    def manual_crop(self,ind_start=0,ind_stop=0):
        print("croping")
        
        if ind_stop==0:
            ind_stop=len(self.acc_interp)-1
            
        self.acc_interp=self.acc_interp.iloc[ind_start:ind_stop,:].copy()
        self.gyro_interp=self.gyro_interp.iloc[ind_start:ind_stop,:].copy()
        
        
            
    def save_file(self):
        """
        save the full data file after preprocessing to csv

        """
        
        self.acc_aligned.to_csv('acc_hand.csv',index=True)
        self.gyro_aligned.to_csv('gyro_hand.csv',index=True)      
        
    def peakdet_m1(self, delta=0, x = None,acc=True,gyro=False,plot=True):
        """
        Converted from MATLAB script at http://billauer.co.il/peakdet.html
        
        Currently returns two lists of tuples, but maybe arrays would be better
        
        function [maxtab, mintab]=peakdet(v, delta, x)
        %PEAKDET Detect peaks in a vector
        %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
        %        maxima and minima ("peaks") in the vector V.
        %        MAXTAB and MINTAB consists of two columns. Column 1
        %        contains indices in V, and column 2 the found values.
        %      
        %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
        %        in MAXTAB and MINTAB are replaced with the corresponding
        %        X-values.
        %
        %        A point is considered a maximum peak if it has the maximal
        %        value, and was preceded (to the left) by a value lower by
        %        DELTA.
        
        % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
        % This function is released to the public domain; Any use is allowed.
        
        """
        if acc:
            v=self.acc_magnitude
        if gyro:
            v=self.gyro_magnitude
        maxtab = []
        mintab = []
        
        delta=np.abs(np.amax(v))/5 
        if x is None:
            x = np.arange(len(v))
        
        v = np.asarray(v)
        
        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')
        
        if not np.isscalar(delta):
            sys.exit('Input argument delta must be a scalar')
        
        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        
        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN
        
        lookformax = True
        
        for i in np.arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]
            
            if lookformax:
                if this < mx-delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn+delta:
                    mintab.append((mnpos, mn))
                    mx = this
                    mxpos = x[i]
                    lookformax = True
        peak_index=np.vstack(maxtab)[:,0].astype('int')
        valley_index=np.vstack(mintab)[:,0].astype('int')
        self.peakandvalley={'peak_index':peak_index,'valley_index':valley_index}
        
        if plot:
            plt.figure()
            x=v
            plt.plot(x)
            plt.plot(peak_index, x[peak_index], "x")
           
    
        return()
    
    def peakdet_m2(self,acc=None,plot_valley=False,plot_peak=True,detect_turn=False):
        """
        peak detection using python built in method
        :param bool acc: use acceleration data if false use gyroscope data
        :param bool plot_valley: plot valley in signal norm
        :param bool plot_peak:plot peak in signal norm
        """
        if acc:
            magmatrix=self.acc_magnitude
            delta_acc=np.abs(np.amax((magmatrix)))/7
            delta_acc=1
            peak_index,peak_properties= find_peaks(magmatrix,distance=30,height=(None,None),prominence=(delta_acc,None),width=(None,None),threshold=(None,None)) # hand low speed prominence 1
            valley_index,valley_properties= find_peaks(-magmatrix,distance=30,height=(None,None),prominence=(delta_acc,None),width=(None,None),threshold=(None,None))#width=(7,None)
#            matrix=self.acc_aligned
        else:
            magmatrix=self.gyro_magnitude
            #pocket
            if detect_turn:
                peak_index,peak_properties= find_peaks(magmatrix,distance=500,height=(1.5,None),prominence=(1.5,None),width=(None,None),threshold=(None,3))#slow height(100,None) # comf height (120,None) # high speed height (150)
                
                delta_gyr=np.abs(np.amax(-(magmatrix-np.mean(magmatrix))))/10
                valley_index,valley_properties= find_peaks(-magmatrix,distance=30,height=(None,None),prominence=(None,None),width=(None,None),threshold=(None,delta_gyr))

            else:
                delta_gyr=np.abs(np.amax((magmatrix-np.mean(magmatrix))))/10
                peak_index,peak_properties= find_peaks(magmatrix,distance=80,height=(None,None),prominence=(None,None),width=(None,None),threshold=(None,delta_gyr))#slow height(100,None) # comf height (120,None) # high speed height (150)
                delta_gyr=np.abs(np.amax(-(magmatrix-np.mean(magmatrix))))/10
                valley_index,valley_properties= find_peaks(-magmatrix,distance=30,height=(None,None),prominence=(None,None),width=(None,None),threshold=(None,delta_gyr))
#            matrix=self.gyro_aligned
        #---indexes of peaks and vallyes---    
      
        #---time of peaks and valleys
#        peak_time=[matrix.index.values[i] for i in peak_index]
#        valley_time=[matrix.index.values[i] for i in valley_index]
        #---value of peaks and valleys
        peak_value=[magmatrix[i] for i in peak_index]
        valley_value=[magmatrix[i] for i in valley_index]
        
        if plot_peak:
            plt.figure()
            x=magmatrix
            plt.plot(x)
            plt.plot(peak_index, x[peak_index], "x")
            plt.vlines(x=peak_index, ymin=x[peak_index] - peak_properties["prominences"],ymax = x[peak_index], color = "C1")
            plt.hlines(y=peak_properties["width_heights"], xmin=peak_properties["left_ips"],xmax=peak_properties["right_ips"], color = "C1")
        if plot_valley:
            plt.figure()
            x=magmatrix
            plt.plot(x)
            plt.plot(valley_index, x[valley_index], "x")
#            plt.vlines(x=peak_index, ymin=x[peak_index] - peak_properties["prominences"],ymax = x[peak_index], color = "C1")
#            plt.hlines(y=peak_properties["width_heights"], xmin=peak_properties["left_ips"],xmax=peak_properties["right_ips"], color = "C1")
        
        peakandvalley={'peak_index':peak_index,'peak_properties':peak_properties,'valley_index':valley_index,'valley_properties':valley_properties}
        self.peakandvalley=peakandvalley
        
        return()
        
    def computeVarStride(self,fs=100,remove_outliers=True,N=1,use_smartstep=False,manual_peaks=[],use_peaks=True,pocket=True,remove_step=0,round_data=True):
        """
        compute stride time 
        :param int fs: sampling frequency
        :param bool remove_outlier: whether to remove outliers
        :param int N: Nxstandard deviation 
        :param bool use_peaks: peaks are used as a mark that step happened
        :param bool pocket: whether one stride time is calculated 
        """
        #note: we remove two steps from beginging and end
        cycle_tempparam = {}
        if use_smartstep:
            peaks=manual_peaks
        else:
            if use_peaks:
                peaks=np.array(self.peakandvalley['peak_index'])
                if remove_step!=0:
                    peaks=peaks[remove_step:len(peaks)-remove_step]
            else:
                peaks=np.array(self.peakandvalley['valley_index'])

        if pocket:
            #---if cellphone is in the pocket---
            stride_time=np.diff(peaks)/fs
            
            list_stride_time=[]
            list_step_time=[]
            for i in range(0,len(peaks)-1):
                stride_time=peaks[i+1]-peaks[i]
                stride_time=stride_time/fs
                list_stride_time.append([peaks[i],peaks[i+1],stride_time])
            list_stride_time=np.vstack(list_stride_time)

            if remove_outliers:
                list_stride_time=np.vstack([i for i in list_stride_time if i[2]>=0.8 and i[2]<=2])
                mean=np.mean(list_stride_time[:,2])
                cut_off=N*np.std(list_stride_time[:,2])
                lower, upper =  mean- cut_off, mean + cut_off
                list_stride_time=np.vstack([i for i in list_stride_time if i[2]>=lower and i[2]<=upper])

            cycle_tempparam['stridetime']=list_stride_time[:,2]
            #---stride time std and cov
            cycle_tempparam['stridetime_std']=np.around(np.std(cycle_tempparam['stridetime']),decimals=3)
            cycle_tempparam['stridetime_Cov']=np.around(np.std(cycle_tempparam['stridetime']*100)/np.mean(cycle_tempparam['stridetime']),decimals=3)
            cycle_tempparam["detailed_stridetime"]=list_stride_time
        else:
            #---if cellphone is in the hand or waist we can detect leading and contralateral foot stride time---
            stride_time_leading=np.diff(peaks[::2])/fs
            stride_time_contralateral=np.diff(peaks[1::2])/fs

            list_stride_time=[]
            list_step_time=[]
            for i in range(0,len(peaks)-2):
                step_time=peaks[i+1]-peaks[i]
                step_time=step_time/fs
                list_step_time.append([peaks[i],peaks[i+1],step_time])
                
                stride_time=peaks[i+2]-peaks[i]
                stride_time=stride_time/fs
                list_stride_time.append([peaks[i],peaks[i+2],stride_time])
            
            list_step_time=np.vstack(list_step_time)
            list_stride_time=np.vstack(list_stride_time)
            
            if remove_outliers:
                stride_time_leading=np.array([i for i in stride_time_leading if i >= 0.8 and i <= 2])
                stride_time_contralateral=np.array([i for i in stride_time_contralateral if i >= 0.8 and i <= 2])
                #---stride time leading foot---
                mean=np.mean(stride_time_leading)
                cut_off=N*np.std(stride_time_leading)
                lower, upper =  mean- cut_off, mean + cut_off
                cycle_tempparam['stride_time_leading'] = np.array([i for i in stride_time_leading if i > lower and i < upper])
                #---stride time contralateral foot---
                mean=np.mean(stride_time_contralateral)
                cut_off=N*np.std(stride_time_contralateral)
                lower, upper =  mean- cut_off, mean + cut_off
                cycle_tempparam['stride_time_contralateral'] = np.array([i for i in stride_time_contralateral if i > lower and i < upper])
                
                #---step time---
                mean=np.mean(list_step_time[:,2])
                cut_off=N*np.std(list_step_time[:,2])
                lower, upper =  mean- cut_off, mean + cut_off
                cycle_tempparam['steptime'] = np.array([i for i in list_step_time[:,2] if i > lower and i < upper])

                list_stride_time=np.vstack([i for i in list_stride_time if i[2]>=0.8 and i[2]<=2])
                mean=np.mean(list_stride_time[:,2])
                cut_off=N*np.std(list_stride_time[:,2])
                lower, upper =  mean- cut_off, mean + cut_off
                list_stride_time=np.vstack([i for i in list_stride_time if i[2]>=lower and i[2]<=upper])

            else:
                cycle_tempparam['stride_time_leading']=stride_time_leading
                cycle_tempparam['stride_time_contralateral']=stride_time_contralateral
                cycle_tempparam['steptime']=step_time
                
            #---merge left right stride cycle
            
            rl_stride=list_stride_time[:,2]
            
            if len(rl_stride)>1:
                cycle_tempparam['stridetime']=rl_stride
                cycle_tempparam["detailed_stridetime"]=list_stride_time
                cycle_tempparam["detailed_steptime"]=list_step_time
                
                cycle_tempparam['stride_time_leading_std']=np.around(np.std(cycle_tempparam['stride_time_leading']),decimals=3)
                cycle_tempparam['stride_time_leading_Cov']=np.around(np.std(cycle_tempparam['stride_time_leading']*100)/np.mean(cycle_tempparam['stride_time_leading']),decimals=3)
                
                cycle_tempparam['stride_time_contralateral_std']=np.around(np.std(cycle_tempparam['stride_time_contralateral']),decimals=3)
                cycle_tempparam['stride_time_contralateral_Cov']=np.around(np.std(cycle_tempparam['stride_time_contralateral']*100)/np.mean(cycle_tempparam['stride_time_contralateral']),decimals=3)
                
                cycle_tempparam['stridetime_std']=np.around(np.std(cycle_tempparam['stridetime']),decimals=3)
                cycle_tempparam['stridetime_Cov']=np.around(np.std(cycle_tempparam['stridetime']*100)/np.mean(cycle_tempparam['stridetime']),decimals=3)
                
                cycle_tempparam['steptime_std']=np.around(np.std(cycle_tempparam['steptime']),decimals=3)
                cycle_tempparam['steptime_Cov']=np.around(np.std(cycle_tempparam['steptime']*100)/np.mean(cycle_tempparam['steptime']),decimals=3)
                
                
            else:
                print("most strides have been filtered because of misdetection ")
        self.cycle_temp=cycle_tempparam
        
    
    def compute_meanstridecurve(self,mark=True,TM_cycle=0,remove_outliers=True,N=2,plot=True):
        """
        compute mean stride time 
        :param bool mark: use events detected from treadmill
        :param array TM_cycle: events detected from treadmill
        :param bool remove_outliers: whether to remove outliers
        :param int N:  Nxstandard deviation  outlier 
        :param bool plot: whether to plot curve
        """
        signal_acc=self.acc_magnitude
        signal_gyro=self.gyro_magnitude
        
        if mark:
            strides_cycles=[]
            for i in range(0,len(TM_cycle)-1):
                strides_cycles.append([TM_cycle[i],TM_cycle[i+1]])
            strides_cycles=np.vstack(strides_cycles)
            
            stride_peak=strides_cycles[:,1]-strides_cycles[:,0]
            strides=[]
            if remove_outliers:
                mean=np.mean(stride_peak)
                cut_off=N*np.std(stride_peak)
                lower, upper =  mean- cut_off, mean + cut_off
                
                for i in range(0,len(stride_peak)):
                    if stride_peak[i]<upper and stride_peak[i]>lower:
                        strides.append((strides_cycles[i,0],strides_cycles[i,1]))
                strides=np.vstack(strides)
        else:
        
            peak_index=self.peakandvalley['peak_index']
            peak_index=peak_index[::2]
       
            stride_peak=np.diff(peak_index)
            strides=[]
            if remove_outliers:
                print("we are in remove outliers")
                mean=np.mean(stride_peak)
                cut_off=N*np.std(stride_peak)
                lower, upper =  mean- cut_off, mean + cut_off
                
                for i in range(0,len(stride_peak)):
                    if stride_peak[i]<upper and stride_peak[i]>lower:
                        strides.append((peak_index[i],peak_index[i+1]))
                strides=np.vstack(strides)
                print("the strides are created")
                self.cpstrides=strides
           
        Total_strides_acc=[]
        Total_strides_gyro=[]
        
        for i in range(0,len(strides)):
            
            crop_signal_acc=signal_acc[strides[i,0]:strides[i,1]]
            crop_signal_gyro=signal_gyro[strides[i,0]:strides[i,1]]
            
            normalize_signal_acc=np.zeros((100))
            normalize_signal_gyro=np.zeros((100))
            for j in range (0,100):
                normalize_signal_acc[j]=crop_signal_acc[np.round(j*len(crop_signal_acc)/100).astype('int')]
                normalize_signal_gyro[j]=crop_signal_gyro[np.round(j*len(crop_signal_gyro)/100).astype('int')]
                
            Total_strides_acc.append(normalize_signal_acc)
            Total_strides_gyro.append(normalize_signal_gyro)
            
        if plot:
            plt.figure(figsize=(20,10))
            mean_acc=np.mean(Total_strides_acc,axis=0)
            std_acc=np.std(Total_strides_acc,axis=0)
            plt.plot(mean_acc)
            plt.fill_between(np.arange(0,100),y1=mean_acc+std_acc,y2=mean_acc-std_acc,alpha=0.5)
            plt.xlabel('Gait cycle %',fontsize=60,weight='bold')
            plt.ylabel('[m/s^2]',fontsize=60,weight='bold')
            
            
            plt.figure(figsize=(20,10))
            mean_gyro=np.mean(Total_strides_gyro,axis=0)
            std_gyro=np.std(Total_strides_gyro,axis=0)
            plt.plot(mean_gyro)
            plt.fill_between(np.arange(0,100),y1=mean_gyro+std_gyro,y2=mean_gyro-std_gyro,alpha=0.5)
            plt.xlabel('Gait cycle %',fontsize=60,weight='bold')
            plt.ylabel('[deg/s]',fontsize=60,weight='bold')
            
        self.acc_strides=Total_strides_acc
        self.gyro_strides=Total_strides_gyro
        
 
        
        
        
           
        
    def plot_results(self,varStride=True,savefigure=False,pocket=True):
        """
        :param bool varstride: plot mean and standard deviation lines
        :param bool savefigure: save figure
        :param bool pocket: whether to plot one side 
        """
        #---plot stride time--
        mydict=self.cycle_temp
        if pocket:
            fig=plt.figure()
            plt.plot(mydict['stridetime'])
            plt.ylabel('Time(s)')
            plt.xlabel('Stride number')
            plt.text(0.8, 0.8,("Stride std:%s ms, Stride cov:%s%%"%(mydict['stridetime_std']*1000,mydict['stridetime_Cov'])),size=20,transform=fig.transFigure,ha="center", va="top", bbox=dict(facecolor='red', alpha=0.5))
            plt.axhline(y=np.mean(mydict['stridetime']),label='Average Stride Time',color='k')
            plt.axhline(y=np.mean(mydict['stridetime'])-mydict['stridetime_std'],linestyle='--',color='k',linewidth=0.5)
            plt.axhline(y=np.mean(mydict['stridetime'])+mydict['stridetime_std'],linestyle='--',color='k',linewidth=0.5)
            plt.title('pocket stride time')
        else:
            fig, ax=plt.subplots(2,figsize=(25,15))
            ax[0].plot(mydict['stridetime'])
            ax[1].plot(mydict['steptime'])
            font = {'family' : 'cursive',
                    'style':'normal',
                    'weight': 2,
                    'size'   : 20}
            plt.rc('font', **font)
            plt.subplots_adjust(hspace=0.5)
#            ax[0].set_xlabel('Number of stride')
            ax[0].set_ylabel('Time(s)')
            ax[1].set_xlabel('Number of steps')
            ax[1].set_ylabel('Time(s)')
            ax[0].set_title('Stride time')
            ax[1].set_title('Step time')
            plt.text(0.5, 0.5,("Stride std:%s ms, Stride cov:%s%%, Step std:%s ms, Step cov:%s%%"%(mydict['stridetime_std']*1000,mydict['stridetime_Cov'],mydict['steptime_std']*1000,mydict['steptime_Cov'])),transform=fig.transFigure,size=20,ha="center", va="center", bbox=dict(facecolor='red', alpha=0.5))
            if varStride:
                ax[0].axhline(y=np.mean(mydict['stridetime']),label='Average Stride Time',color='k')
                ax[0].axhline(y=np.mean(mydict['stridetime'])-mydict['stridetime_std'],linestyle='--',color='k',linewidth=0.5)
                ax[0].axhline(y=np.mean(mydict['stridetime'])+mydict['stridetime_std'],linestyle='--',color='k',linewidth=0.5)
                ax[1].axhline(y=np.mean(mydict['steptime']),label='Average Step Time',color='k')
                ax[1].axhline(y=np.mean(mydict['steptime'])-mydict['steptime_std'],linestyle='--',color='k',linewidth=0.5)
                ax[1].axhline(y=np.mean(mydict['steptime'])+mydict['steptime_std'],linestyle='--',color='k',linewidth=0.5)
                ax[0].legend()
                ax[1].legend()
                
            
        if savefigure:
            x=os.path.dirname(self.filename)
            y=os.path.join(x,'Stride and step time variability.png')
            fig.savefig(y, format='png')
        return()
        
        
    def read_gaitupexcel(self,path,period=30,remove_init=True,phase=True):
        sheet = pd.read_excel(path)
        HS_L=sheet['Unnamed: 3'].iloc[24:len(sheet)] #change to 20 in case of old excel file
        HS_R=sheet['Unnamed: 4'].iloc[24:len(sheet)]
        gct_L=sheet['Unnamed: 5'].iloc[24:len(sheet)]
        gct_R=sheet['Unnamed: 6'].iloc[24:len(sheet)]
        x=pd.DataFrame(columns=['HS_L','HS_R','gct_L','gct_R'])
        x['HS_L']=HS_L.values
        x['HS_R']=HS_R.values
        x['gct_L']=gct_L.values
        x['gct_R']=gct_R.values
        start=x['HS_L'].values[0]
        end=x['HS_L'].values[len(x)-3]
        gaitup_param = {}
        if phase:
            #waist
            period=210
            warmup=29
            change=30
            start=start+warmup
            end=start+period
            t=np.where(x['HS_L'].values>start)[0][0]
            tt=np.where(x['HS_L'].values>end)[0][0]
            if remove_init:
                y=x.iloc[t:tt,:].copy()
            gaitup_param['waist_Lstd_stride']=np.std(y['gct_L'].values)*1000
            gaitup_param['waist_Rstd_stride']=np.std(y['gct_R'].values)*1000
            gaitup_param['waist_Lstd_Cov']=np.std(y['gct_L'].values)*100/np.mean(y['gct_L'].values)
            gaitup_param['waist_Rstd_Cov']=np.std(y['gct_R'].values)*100/np.mean(y['gct_R'].values)
            start=end+change
            end=start+period
            t=np.where(x['HS_L'].values>start)[0][0]
            tt=np.where(x['HS_L'].values>end)[0][0]
            if remove_init:
                y=x.iloc[t:tt,:].copy()
            gaitup_param['pocket_Lstd_stride']=np.std(y['gct_L'].values)*1000
            gaitup_param['pocket_Rstd_stride']=np.std(y['gct_R'].values)*1000
            gaitup_param['pocket_Lstd_Cov']=np.std(y['gct_L'].values)*100/np.mean(y['gct_L'].values)
            gaitup_param['pocket_Rstd_Cov']=np.std(y['gct_R'].values)*100/np.mean(y['gct_R'].values)
            start=end+change
            end=x['HS_L'].values[len(x)-2]
            t=np.where(x['HS_L'].values>start)[0][0]
            tt=np.where(x['HS_L'].values>end)[0][0]
            if remove_init:
                y=x.iloc[t:tt,:].copy()
            gaitup_param['hand_Lstd_stride']=np.std(y['gct_L'].values)*1000
            gaitup_param['hand_Rstd_stride']=np.std(y['gct_R'].values)*1000
            gaitup_param['hand_Lstd_Cov']=np.std(y['gct_L'].values)*100/np.mean(y['gct_L'].values)
            gaitup_param['hand_Rstd_Cov']=np.std(y['gct_R'].values)*100/np.mean(y['gct_R'].values)  
        self.gaitup=gaitup_param
        return(x,self.gaitup)
        
    def detectstartofwalk(self,sig1,thresh=3): 
        i=0
        N_wf=128
        condition=True    
        while condition==True:
#            mag=sig1.iloc[i:i+N_wf]
            mag=sig1[i:i+N_wf]
            mag=mag-np.mean(mag)
            ener=np.max(mag**2)
            if ener>thresh:
                while condition==True:
#                    mag=sig1.iloc[i:i+N_wf//4]
                    mag=sig1[i:i+N_wf//4]
                    ener=np.max(mag**2)
                    if ener>thresh:
                        startp=i
                        condition=False
                    i=i+N_wf//4
            i=i+N_wf
            
        # startp=startp/100
        
        return(startp)
        
    def align_gaitup_phone(self,gaitup,pocket=False):
        """
        """
        align_stride={}
        if not pocket:
            x1=self.cycle_temp['stride_time_contralateral']
            x2=self.cycle_temp['stride_time_leading']
            y=gaitup['gct_L'].values
            lag=np.argmax(signal.correlate(y,x1))
            lagInd=np.arange(-np.max([len(y),len(x1)]),np.max([len(y),len(x1)]))
            lag=lagInd[lag]
    
            if lag<0:
                x1_new=x1[-lag:len(x1)]
                l=np.amin([len(x1_new),len(y)])
                x1_new=x1_new[0:l]
                y1_new=y[0:l]
            else:
                y1_new=y[lag:len(y)]
                l=np.amin([len(y1_new),len(x1)])
                x1_new=x1[0:l]
                y1_new=y1_new[0:l]
            print(lag)
            lag=np.argmax(signal.correlate(y,x2))
            lagInd=np.arange(-np.max([len(y),len(x2)]),np.max([len(y),len(x2)]))
            lag=lagInd[lag]
            
            if lag<0:
                x2_new=x2[-lag:len(x2)]
                l=np.amin([len(x2_new),len(y)])
                x2_new=x2_new[0:l]
                y2_new=y[0:l]
            else:
                y2_new=y[lag:len(y)]
                l=np.amin([len(y2_new),len(x2)])
                x2_new=x2[0:l]
                y2_new=y2_new[0:l]
            print(lag)
            c1=np.sum(((x1_new-y1_new)**2))
            c2=np.sum(((x2_new-y2_new)**2))
            
            if c2>c1:
                align_stride['gct_L']=[x1_new,y1_new]
                one=True
                plt.figure()
                plt.plot(x1_new)
                plt.plot(y1_new)
            else:
                align_stride['gct_L']=[x2_new,y2_new]
                one=False
                plt.figure()
                plt.plot(x2_new)
                plt.plot(y2_new) 
            y=gaitup['gct_R'].values
            if one==True:
                rx=x2
            else:
                rx=x1
            
            lag=np.argmax(signal.correlate(y,rx))
            lagInd=np.arange(-np.max([len(y),len(rx)]),np.max([len(y),len(rx)]))
            lag=lagInd[lag]
            
            print(lag)
            if lag<0:
                x1_new=rx[-lag:len(rx)]
                l=np.amin([len(x1_new),len(y)])
                x1_new=x1_new[0:l]
                y1_new=y[0:l]
            else:
                y1_new=y[lag:len(y)]
                l=np.amin([len(y1_new),len(rx)])
                x1_new=rx[0:l]
                y1_new=y1_new[0:l]
            
            align_stride['gct_R']=[x1_new,y1_new]
            
            self.align_stride=align_stride
            plt.figure()
            plt.plot(x1_new)
            plt.plot(y1_new)
        else:
            y=self.cycle_temp['stridetime']
            x1=gaitup['gct_L'].values
            x2=gaitup['gct_R'].values
            
            lag=np.argmax(signal.correlate(y,x1))
            lagInd=np.arange(-np.max([len(y),len(x1)]),np.max([len(y),len(x1)]))
            lag=lagInd[lag]
    
            if lag<0:
                x1_new=x1[-lag:len(x1)]
                l=np.amin([len(x1_new),len(y)])
                x1_new=x1_new[0:l]
                y1_new=y[0:l]
            else:
                y1_new=y[lag:len(y)]
                l=np.amin([len(y1_new),len(x1)])
                x1_new=x1[0:l]
                y1_new=y1_new[0:l]
            print(lag)
            lag=np.argmax(signal.correlate(y,x2))
            lagInd=np.arange(-np.max([len(y),len(x2)]),np.max([len(y),len(x2)]))
            lag=lagInd[lag]
            
            if lag<0:
                x2_new=x2[-lag:len(x2)]
                l=np.amin([len(x2_new),len(y)])
                x2_new=x2_new[0:l]
                y2_new=y[0:l]
            else:
                y2_new=y[lag:len(y)]
                l=np.amin([len(y2_new),len(x2)])
                x2_new=x2[0:l]
                y2_new=y2_new[0:l]
            print(lag)
            c1=np.sum(((x1_new-y1_new)**2))
            c2=np.sum(((x2_new-y2_new)**2))
            
            align_stride={}
            
            if c2>c1:
                align_stride['gct_L']=[y1_new,x1_new]
                one=True
                plt.figure()
                plt.plot(x1_new)
                plt.plot(y1_new)
            else:
                align_stride['gct_R']=[y2_new,x2_new]
                one=False
                plt.figure()
                plt.plot(x2_new)
                plt.plot(y2_new)
                
            self.align_stride=align_stride
            
        return(align_stride)
        
    def plot_gaitupresult(self):
        
        g=self.align_stride
        
        for i in [*g]:
        #---left stride---
            if i=='gct_L':
                left=g['gct_L']
                fig, ax=plt.subplots(2,figsize=(30,15),sharey=True,sharex=True)
                ax[0].plot(left[0])
                ax[1].plot(left[1])
                plt.subplots_adjust(hspace=0.5)
                ax[0].set_ylabel('Time(s)')
                ax[1].set_xlabel('Number of stride')
                ax[1].set_ylabel('Time(s)')
                ax[0].set_title('phone Left stride time')
                ax[1].set_title('Gaitup Left stride time')
                plt.text(0.5, 0.5,("Pstd:%s ms, Pcov:%s%%, Gstd:%s ms, Gcov:%s%%"%(np.round(np.std(left[0])*1000,decimals=2),np.round(np.std(left[0])*100/np.mean(left[0]),decimals=2),np.round(np.std(left[1])*1000,decimals=2),np.round(np.std(left[1])*100/np.mean(left[1]),decimals=2))),transform=fig.transFigure,size=20,ha="center", va="center", bbox=dict(facecolor='red', alpha=0.5))
            if i=='gct_R':
                Right=g['gct_R']
                fig, ax=plt.subplots(2,figsize=(30,15),sharey=True,sharex=True)
                ax[0].plot(Right[0])
                ax[1].plot(Right[1])
                plt.subplots_adjust(hspace=0.5)
                ax[0].set_ylabel('Time(s)')
                ax[1].set_xlabel('Number of stride')
                ax[1].set_ylabel('Time(s)')
                ax[0].set_title('phone Right stride time')
                ax[1].set_title('Gaitup Right stride time')
                plt.text(0.5, 0.5,("Pstd:%s ms, Pcov:%s%%, Gstd:%s ms, Gcov:%s%%"%(np.round(np.std(Right[0])*1000,decimals=2),np.round(np.std(Right[0])*100/np.mean(Right[0]),decimals=2),np.round(np.std(Right[1])*1000,decimals=2),np.round(np.std(Right[1])*100/np.mean(Right[1]),decimals=2))),transform=fig.transFigure,size=20,ha="center", va="center", bbox=dict(facecolor='red', alpha=0.5))
        
        

    def find_dominantfreq(self,fs=100,N_wf=256,overlap=128):
        """
        Find features in data stepfreq, meandominantfreq, Variance, Energy, VarXmedian.
        :param int N_wf: windowsize number of samples
        :param int overlap: overlap between windows
        """
        signals=[self.acc_magnitude,self.gyro_magnitude]
        f0v=8
        fmin=0.5
        fmax=4
        Nf=int(f0v*(N_wf/2))
        Ns=10
        self.acc_features=pd.DataFrame(columns=['stepfreq','meandominantfreq','Variance','Energy','VarXmedian'])
        self.gyro_features=pd.DataFrame(columns=['stepfreq','meandominantfreq','Variance','Energy','VarXmedian'])
        s=0
        for sig in signals:
            dom_freq=[]
            step_freq=[]
            var=[]
            ener=[]
            VarXMedian=[]
            for i in range(0,len(sig)-N_wf,overlap):
                mag=sig[i:i+N_wf]
                VarXMedian.append(np.var(mag)*np.median(mag))
                mag=mag-np.mean(mag)
#                mx=np.amax(mag)
#                mn=np.amin(mag)
#                mag=(mag-mn)/(mx-mn)
                
                var.append(np.var(mag))
#                print(var)
                ener.append(np.max(mag**2))
                domfreq=np.zeros((1,3))
                fourcoef=np.zeros((1,3), dtype=complex)
                
                freq=np.arange(0,(2*Nf))/(2*Nf)*(fs)
                lowind=np.where(freq>fmin)[0][0]
                upind=np.max(np.where(freq<fmax))
                
                haming= np.hamming(N_wf)
                
                furval=fft(mag*haming,n=2*Nf)
                
                fourcoef[0,0]=(furval[lowind+np.argmax(np.abs(furval[lowind:upind]))])
                ind=lowind+np.argmax(np.abs(furval[lowind:upind]))
                idx=np.where(furval==fourcoef[0,0])[0][0]
                domfreq[0,0]=freq[idx]
                furval[np.maximum(1,ind-Ns):(ind+Ns)]=0 #         furval[np.maximum(1,ind-Ns):(ind+Ns)+lowind+1]=0
                
                fourcoef[0,1]=(furval[lowind+np.argmax(np.abs(furval[lowind:upind]))])
                ind=lowind+np.argmax(np.abs(furval[lowind:upind]))
                idx=np.where(furval==fourcoef[0,1])[0][0]
                domfreq[0,1]=freq[idx]
                furval[np.maximum(1,ind-Ns):(ind+Ns)]=0
                
                fourcoef[0,2]=(furval[lowind+np.argmax(np.abs(furval[lowind:upind]))])
                ind=lowind+np.argmax(np.abs(furval[lowind:upind]))
                idx=np.where(furval==fourcoef[0,2])[0][0]
                domfreq[0,2]=freq[idx]
                
                stepfreq=domfreq[0,0]
                    
#                if domfreq[0,0]<1.2:
#                    stepfreq=domfreq[0,1]
                    
                step_freq.append(stepfreq)
                dom_freq.append(domfreq)
#            plt.plot(mag)
            if s==0:
                
                self.acc_features['stepfreq']=step_freq
                self.acc_features['meandominantfreq']=np.mean(np.vstack(dom_freq),axis=1)
                self.acc_features['Variance']=var
                self.acc_features['Energy']=ener
                self.acc_features['VarXmedian']=VarXMedian
                self.dom_freqacc=dom_freq
                
            if s==1:
                
                self.gyro_features['stepfreq']=step_freq
                self.gyro_features['meandominantfreq']=np.mean(np.vstack(dom_freq),axis=1)
                self.gyro_features['Variance']=var
                self.gyro_features['Energy']=ener
                self.gyro_features['VarXmedian']=VarXMedian
                self.dom_freqgyro=dom_freq
            
            s=s+1
            
            
    def detect_mode(self):
        mode=pd.DataFrame(columns=['periodicityacc','periodicitygyro','texting','undefined'])
        mode['periodicityacc']= (self.acc_features['stepfreq'].values>0.5) & (self.acc_features['stepfreq'].values<2.5)
        mode['periodicitygyro']= (self.gyro_features['stepfreq'].values>0.5) & (self.gyro_features['stepfreq'].values<2.5)
        
        mode['undefined']=self.gyro_features['Energy'].values>(15000*(np.pi/180))**2
        
        mode['static']=(self.gyro_features['Energy'].values<(60*(np.pi/180))**2) & (self.acc_features['Energy'].values<5)
        mode['texting']=(self.gyro_features['VarXmedian'].values<0.37) &~ mode['static'] & ~ mode['undefined']& mode['periodicityacc']
        mode['swinging']= ~ mode['static'] & ~ mode['undefined'] & ~ mode['texting']& mode['periodicitygyro']
        
        self.mode=mode
        
    def detect_steps(self,remove=True,hst=1.5,lst=0.8,N_wf=512,overlap=512,fs=100):
        cycle_tempparam = {}
        irregular=0
        swing_mode=self.mode['swinging'].values
        text_mode=self.mode['texting'].values
        fcg=self.gyro_features['stepfreq'].values
        fca=self.acc_features['stepfreq'].values
        
        allsteps=np.zeros([1])
        step_timee=np.zeros([1])
        
        stride_time_leadingg=np.zeros([1])
        
        stride_time_contralaterall=np.zeros([1])
        kk=0
        debug=[]
        i=1
        for i in range(0,len(self.gyro_magnitude)-N_wf,overlap):
            if swing_mode[kk]==True:
                
                x=self.gyro_magnitude[i:i+N_wf]-np.mean(self.gyro_magnitude[i:i+N_wf])
                Wn =(fcg[kk]+1)/(fs/2) # Cutoff frequency normalized 
                B, A = signal.butter(10, Wn,'low', output='ba')
                x=signal.filtfilt(B, A, x)
                
                p=-np.mean(np.sort(x)[:150])
                
                delta_gyr=np.abs(np.amax(-x))/10
                
                valley_index,_= find_peaks(-x,distance=np.int(100/fcg[kk])-10,height=(0,None),prominence=(None,None),width=(None,None),threshold=(None,delta_gyr)) # hand low speed prominence 1
                
                valley_index=valley_index+i
        #        valley_index=np.expand_dims(valley_index,axis=1)
                allsteps=np.concatenate((allsteps,valley_index))
                
            
            elif text_mode[kk]:
                
                y=self.acc_magnitude[i:i+N_wf]-np.mean(self.acc_magnitude[i:i+N_wf])
        #                mx=np.amax(y)
        #                mn=np.amin(y)
        #                y=(y-mn)/(mx-mn)
                Wn =(fca[kk]+1)/(fs/2) # Cutoff frequency normalized 
                B, A = signal.butter(10, Wn,'low', output='ba')
                y=signal.filtfilt(B, A, y)
                p=-np.mean(np.sort(-y)[:150])
                delta_acc=np.abs(np.amax(y))/10
                peak_index,_= find_peaks(y,distance=np.int(100/fca[kk])-10,height=(0,None),prominence=(None,None),width=(None,None),threshold=(None,delta_acc)) # hand low speed prominence 1
                peak_index=peak_index+i
#                peak_index=np.expand_dims(peak_index,axis=1)
                
                allsteps=np.concatenate((allsteps,peak_index))
            else:
                irregular=irregular+1
                print(irregular)
            kk=kk+1
        
        allsteps=np.delete(allsteps,[0]).astype('int')   
        step_timee=np.diff(np.transpose(allsteps))/fs
        stride_time_contralaterall=np.diff(np.transpose(allsteps[1::2]))/fs
        stride_time_leadingg=np.diff(np.transpose(allsteps[::2]))/fs
        
        Percentageirregular=irregular*100/kk
        print(kk)
                    
            
        if remove:
            N=3
            hst=1.5
            lst=0.8
            stride_time_contralaterall = np.delete(stride_time_contralaterall, np.where((stride_time_contralaterall>hst)|(stride_time_contralaterall<lst)))
            stride_time_leadingg = np.delete(stride_time_leadingg, np.where((stride_time_leadingg>hst)|(stride_time_leadingg<lst)))
            
            mean=np.mean(stride_time_leadingg)
            cut_off=N*np.std(stride_time_leadingg)
            lower, upper =  mean- cut_off, mean + cut_off
            stride_time_leadingg = np.array([i for i in stride_time_leadingg if i > lower and i < upper])
            #---stride time contralateral foot---
            mean=np.mean(stride_time_contralaterall)
            cut_off=N*np.std(stride_time_contralaterall)
            lower, upper =  mean- cut_off, mean + cut_off
            stride_time_contralaterall = np.array([i for i in stride_time_contralaterall if i > lower and i < upper])
            #---step time---
            mean=np.mean(step_timee)
            cut_off=N*np.std(step_timee)
            lower, upper =  mean- cut_off, mean + cut_off
            step_timee = np.array([i for i in step_timee if i > lower and i < upper])
            
            
        cycle_tempparam['stride_time_leading']=stride_time_leadingg
        cycle_tempparam['stride_time_contralateral']=stride_time_contralaterall
        cycle_tempparam['steptime']=step_timee
        
        plt.plot(self.gyro_magnitude)
        plt.scatter(allsteps,self.gyro_magnitude[allsteps])
#        ---merge left right stride cycle
        rl_stride=[]
        j=0
        
        ls=stride_time_leadingg
        rs=stride_time_contralaterall
        
        while j<np.minimum(len(ls),len(rs)):
    
            rl_stride.append(ls[j])
            rl_stride.append(rs[j])
            j=j+1
        
        rl_stride=np.vstack(rl_stride)
        cycle_tempparam['stridetime']=rl_stride
        
        cycle_tempparam['stride_time_leading_std']=np.around(np.std(cycle_tempparam['stride_time_leading']),decimals=3)
        cycle_tempparam['stride_time_leading_Cov']=np.around(np.std(cycle_tempparam['stride_time_leading']*100)/np.mean(cycle_tempparam['stride_time_leading']),decimals=3)
        
        cycle_tempparam['stride_time_contralateral_std']=np.around(np.std(cycle_tempparam['stride_time_contralateral']),decimals=3)
        cycle_tempparam['stride_time_contralateral_Cov']=np.around(np.std(cycle_tempparam['stride_time_contralateral']*100)/np.mean(cycle_tempparam['stride_time_contralateral']),decimals=3)
        
        cycle_tempparam['stridetime_std']=np.around(np.std(cycle_tempparam['stridetime']),decimals=3)
        cycle_tempparam['stridetime_Cov']=np.around(np.std(cycle_tempparam['stridetime']*100)/np.mean(cycle_tempparam['stridetime']),decimals=3)
        
        
        cycle_tempparam['steptime_std']=np.around(np.std(cycle_tempparam['steptime']),decimals=3)
        cycle_tempparam['steptime_Cov']=np.around(np.std(cycle_tempparam['steptime']*100)/np.mean(cycle_tempparam['steptime']),decimals=3)
            
        cycle_tempparam['irregular']=Percentageirregular
        print("Percentageirregular")
        print(Percentageirregular)
        
        self.cycle_temp=cycle_tempparam
        
        self.allsteps=allsteps
        
        
    def normal_signal(self,threeD_acc,threeD_gyro,peaks=[],remove_outliers=True,N=2,Numberofstrides=140,plot=True):
        
        signal_acc=self.acc_magnitude
        signal_gyro=self.gyro_magnitude
        
        
        peak_index=peaks
        peak_index=peak_index[::2]
       
        stride_peak=np.diff(peak_index)
        strides=[]
        
        if remove_outliers:
            print("we are in remove outliers")
            mean=np.mean(stride_peak)
            cut_off=N*np.std(stride_peak)
            lower, upper =  mean- cut_off, mean + cut_off
            
            for i in range(0,len(stride_peak)):
                if stride_peak[i]<upper and stride_peak[i]>lower:
                    strides.append((peak_index[i],peak_index[i+1]))
            strides=np.vstack(strides)
            print("the strides are created")
        else:
            for i in range(0,len(stride_peak)):
                strides.append((peak_index[i],peak_index[i+1]))
            strides=np.vstack(strides)
        self.stridess=strides
        
        crop_signal_acc=signal_acc[strides[0,0]:strides[Numberofstrides,1]]
        crop_signal_gyro=signal_gyro[strides[0,0]:strides[Numberofstrides,1]]
        
        threeD_acc=threeD_acc.iloc[strides[0,0]:strides[Numberofstrides,1],:]
        threeD_gyro=threeD_gyro.iloc[strides[0,0]:strides[Numberofstrides,1],:]
        
        
        
        normalize_signal_acc=np.zeros((100*Numberofstrides))
        normalize_signal_gyro=np.zeros((100*Numberofstrides))
        
        normalize_threesignal_acc=np.zeros((100*Numberofstrides,3))
        normalize_threesignal_gyro=np.zeros((100*Numberofstrides,3))
        
        
        
        for j in range (0,100*Numberofstrides):
            normalize_signal_acc[j]=crop_signal_acc[np.round(j*len(crop_signal_acc)/(100*Numberofstrides)).astype('int')]
            normalize_signal_gyro[j]=crop_signal_gyro[np.round(j*len(crop_signal_gyro)/(100*Numberofstrides)).astype('int')]
            
            normalize_threesignal_acc[j,0]=threeD_acc.iloc[np.round(j*len(threeD_acc)/(100*Numberofstrides)).astype('int'),0]
            normalize_threesignal_acc[j,1]=threeD_acc.iloc[np.round(j*len(threeD_acc)/(100*Numberofstrides)).astype('int'),1]
            normalize_threesignal_acc[j,2]=threeD_acc.iloc[np.round(j*len(threeD_acc)/(100*Numberofstrides)).astype('int'),2]
            

            
            normalize_threesignal_gyro[j,0]=threeD_gyro.iloc[np.round(j*len(threeD_gyro)/(100*Numberofstrides)).astype('int'),0]
            normalize_threesignal_gyro[j,1]=threeD_gyro.iloc[np.round(j*len(threeD_gyro)/(100*Numberofstrides)).astype('int'),1]
            normalize_threesignal_gyro[j,2]=threeD_gyro.iloc[np.round(j*len(threeD_gyro)/(100*Numberofstrides)).astype('int'),2]
            
        
        Total_strides_acc=normalize_signal_acc
        Total_strides_gyro=normalize_signal_gyro
        
        normalize_threeD_acc=pd.DataFrame(data=normalize_threesignal_acc,columns=threeD_acc.columns)
        normalize_threeD_gyro=pd.DataFrame(data=normalize_threesignal_gyro,columns=threeD_acc.columns)
        
        #to remove the mean
        # normalize_threeD_acc.iloc[:,0]=normalize_threeD_acc.iloc[:,0]-np.mean(normalize_threeD_acc.iloc[:,0])
        # normalize_threeD_acc.iloc[:,1]=normalize_threeD_acc.iloc[:,1]-np.mean(normalize_threeD_acc.iloc[:,1])
        # normalize_threeD_acc.iloc[:,2]=normalize_threeD_acc.iloc[:,2]-np.mean(normalize_threeD_acc.iloc[:,2])
        
        
        if plot:
            plt.figure(figsize=(20,10))
            plt.plot(Total_strides_acc)
            plt.xlabel('samples',fontsize=60,weight='bold')
            plt.ylabel('[m/s^2]',fontsize=60,weight='bold')
            
            
            plt.figure(figsize=(20,10))
            plt.plot(Total_strides_gyro)
            plt.xlabel('samples',fontsize=60,weight='bold')
            plt.ylabel('[deg/s]',fontsize=60,weight='bold')
        
        
        self.norma_acc_strides=Total_strides_acc
        self.norma_gyro_strides=Total_strides_gyro  
        
        self.norma_threeD_acc=normalize_threeD_acc
        self.norma_threeD_gyro=normalize_threeD_gyro
        
        
    def findMiddle(self,input_list):
        middle = float(len(input_list))/2
        if middle % 2 != 0:
            return int(middle - .5)
        else:
            return int(middle)
        
    def calc_rms(self,data):
        return np.sqrt(np.mean(data ** 2))
    
    def calc_mean_abs_deviation(self,data):
        return stats.median_abs_deviation(data)
    
    def calc_data_entropy(self,data):
        value, counts = np.unique(data, return_counts=True)
        return stats.entropy(counts)
    
    def calc_index_max(self,data):
        data = list(data)
        return data.index(min(data))
    
    def calc_index_min(self,data):
        data = list(data)
        return data.index(max(data))
    
    def calc_energy(self,data):
        squares = data ** 2
        return squares.sum()
    
    def calc_sma(self,data):
        absolute = list(map(abs, data))
        return sum(absolute)
    
    def _dominant_frequency(self,signal_x, sampling_rate=100):
    
        signal_x = signal_x-np.mean(signal_x)
        dim = signal_x.shape
        
        #valerie from matlab
        nfft=1024
        freq_hat = np.fft.fftfreq(nfft) * sampling_rate
        
        #freq are freq from 0 to 50 hz
        freq = freq_hat[0:nfft // 2]
        
        #valerie from matlab
        fmin=0.5
        fmax=4 
        lowind=np.where(freq>fmin)[0][0]
        upind=np.max(np.where(freq<fmax))
    
        # fourier transform
        #valerie add hamming
        # nfft is used for padding
        haming= np.hamming(dim[0])
        sp_hat = np.fft.fft(signal_x*haming, nfft)
        furval = sp_hat[0:nfft // 2] * np.conjugate(sp_hat[0:nfft // 2])
        furval=furval/sum(furval)
    
        #from the internet
        # cutoff is 12 
        cutoff=12.0
        idx1 = freq <= cutoff
        idx_cutoff = np.argwhere(idx1)
        #all freq less than cutoff
        freq = freq[idx_cutoff]
        #keep values less than cutoff
        sp = furval[idx_cutoff]
        #normalise
        sp_norm = sp / sum(sp)
        furval=furval/sum(furval)
    
        max_freq = freq[sp_norm.argmax()][0] # Feature 1: dominant freq 
        max_freq_val = sp_norm.max().real    #Feature 2: magnitude of dominant freq
        idx2 = (freq > max_freq - 0.5) * (freq < max_freq + 0.5)  
        idx_freq_range = np.where(idx2)[0]
        dom_freq_ratio = sp_norm[idx_freq_range].real.sum() #Feature 3: dominant frequency ratio
    
        # Calculate Feature 4: spectral flatness 
        spectral_flatness = 10.0*np.log10(stats.mstats.gmean(sp_norm)/np.mean(sp_norm))
    
        # Calculate Feature 5: Estimate spectral entropy
        spectral_entropy_estimate = 0
        for isess in range(len(sp_norm)):
            if sp_norm[isess] != 0:
                logps = np.log2(sp_norm[isess])
            else:
                logps = 0
            spectral_entropy_estimate = spectral_entropy_estimate - logps * sp_norm[isess]
    
        spectral_entropy_estimate = spectral_entropy_estimate / np.log2(len(sp_norm))
    
        # from valerie matlab
        Ns=10
        domfreq=np.zeros((1,3))
        fourcoef=np.zeros((1,3), dtype=complex)
        fourcoef[0,0]=(furval[lowind+np.argmax(np.abs(furval[lowind:upind]))])
        ind=lowind+np.argmax(np.abs(furval[lowind:upind]))
        idx=np.where(furval==fourcoef[0,0])[0][0]
        domfreq[0,0]=freq[idx] # it is equal to the maximum frequency ==Max_freq 
        furval[np.maximum(1,ind-Ns):(ind+Ns)]=0 
        fourcoef[0,1]=(furval[lowind+np.argmax(np.abs(furval[lowind:upind]))])
        ind=lowind+np.argmax(np.abs(furval[lowind:upind]))
        idx=np.where(furval==fourcoef[0,1])[0][0]
        domfreq[0,1]=freq[idx] #second dominant freq
        furval[np.maximum(1,ind-Ns):(ind+Ns)]=0
        fourcoef[0,2]=(furval[lowind+np.argmax(np.abs(furval[lowind:upind]))])
        ind=lowind+np.argmax(np.abs(furval[lowind:upind]))
        idx=np.where(furval==fourcoef[0,2])[0][0]
        domfreq[0,2]=freq[idx] #Third dominant freq
    
        return max_freq,max_freq_val,dom_freq_ratio,spectral_flatness[0].real,spectral_entropy_estimate[0].real,domfreq
     
    def calculate_features(self,signals,signals_unfiltered):

        window_size80= 80
        window_size5= 5
        window_size30= 30
        window_size20= 20
        window_size50= 50
        window_size60= 60
        window_size70= 70
        window_slide_step=1
        window_freq=128
    
        acc_features=pd.DataFrame()
        gyro_features=pd.DataFrame()
    
        s=0
    
    
        # signals=[acc_mag_filtered,gyro_mag_filtered]
    
        # signals_unfiltered=[acc_mag_unfiltered,gyro_mag_unfiltered]
    
        ########Feature Calculation#######
        for sig in signals:
            print("calculating both signals")
            acc_minindex_win30=[]
            acc_skew_win5=[]
            acc_median_win80=[]
            acc_valleyprom_win80=[]
            acc_peakprom_win80=[]
            acc_peakprom_win50=[]
            acc_domfreq1=[]
            acc_kurt_win30=[]
    
            gyro_maxindex_win60=[]
            gyro_skew_win70=[]
            gyro_skew_win30=[]
            gyro_maxfreqvalue=[]
            gyro_SMA_win80=[]
            gyro_var_win20=[]
            gyro_maxvalue_win70=[]
            gyro_valleyprom_win80=[]
    
    
            for i in range(window_freq//2,len(sig)-window_freq//2,window_slide_step):
    
                mag30=sig[i-(window_size30//2):i+1+(window_size30//2)]
                mag30_nomean=mag30-np.mean(mag30)
    
                mag5=sig[i-(window_size5//2):i+1+(window_size5//2)]
    
    
                mag80=sig[i-(window_size80//2):i+1+(window_size80//2)]
                mag80_nomean=mag80-np.mean(mag80)
    
                mag50=sig[i-(window_size50//2):i+1+(window_size50//2)]
                mag50_nomean=mag50-np.mean(mag50)
    
                mag60=sig[i-(window_size60//2):i+1+(window_size60//2)]
    
    
                mag70=sig[i-(window_size70//2):i+1+(window_size70//2)]
    
    
                mag20=sig[i-(window_size20//2):i+1+(window_size20//2)]
                mag20_nomean=mag20-np.mean(mag20)
    
    
                mag_freq=signals_unfiltered[s][i-window_freq//2:i+(window_freq//2)+1]
    
    
                #===============================================
    
                if s==0:
    #                 print("calculating acceleration")
    
                    acc_minindex_win30.append(self.calc_index_min(mag30_nomean))
    
                    acc_skew_win5.append(stats.skew(mag5))
                    acc_median_win80.append(np.median(mag80))
                    acc_kurt_win30.append(stats.kurtosis(mag30))
    
                    peak_index,peak_properties= find_peaks(mag80_nomean,prominence=(None,None)) 
                    if peak_index.size>0:
                        ind_mid=self.findMiddle(peak_index)
                        acc_peakprom_win80.append(peak_properties["prominences"][ind_mid])
                    else:
                        acc_peakprom_win80.append(50)
    
                    peak_index,peak_properties= find_peaks(mag50_nomean,prominence=(None,None)) 
                    if peak_index.size>0:
                        ind_mid=self.findMiddle(peak_index)
                        acc_peakprom_win50.append(peak_properties["prominences"][ind_mid])
                    else:
                        acc_peakprom_win50.append(50)
    
                    valley_index,valley_properties= find_peaks(-mag80_nomean,prominence=(None,None))
                    if valley_index.size>0:
                        ind_mid=self.findMiddle(valley_index)
                        acc_valleyprom_win80.append(valley_properties["prominences"][ind_mid])
                    else:
                        acc_valleyprom_win80.append(50)
    
                    _,_,_,_,_,domfreq=self._dominant_frequency(mag_freq)
    
    
                    acc_domfreq1.append(domfreq[0,0])
    
    
    
                if s==1:
    #                 print("gyro")
                    gyro_skew_win70.append(stats.skew(mag70))
                    gyro_skew_win30.append(stats.skew(mag30))
                    gyro_maxindex_win60.append(self.calc_index_max(mag60))
                    gyro_SMA_win80.append(self.calc_sma(mag80_nomean))
                    gyro_var_win20.append(np.var(mag20_nomean))
                    gyro_maxvalue_win70.append(np.amax(mag70))
                    valley_index,valley_properties= find_peaks(-mag80_nomean,prominence=(None,None))
    
                    if valley_index.size>0:
                        ind_mid=self.findMiddle(valley_index)
                        gyro_valleyprom_win80.append(valley_properties["prominences"][ind_mid])
                    else:
                        gyro_valleyprom_win80.append(50)
    
                    _,max_fr_val,_,_,_,_=self._dominant_frequency(mag_freq)
    
                    gyro_maxfreqvalue.append(max_fr_val)
    
            if s==0: 
                print("acceleration features")
                acc_features['acc_indMin_win30']=acc_minindex_win30
                acc_features['acc_skew_win5']=acc_skew_win5
                acc_features['acc_Median_win80']=acc_median_win80
                acc_features['acc_peak_prominences80']=acc_peakprom_win80
                acc_features['acc_peak_prominences50']=acc_peakprom_win50
                acc_features['acc_valley_prominences80']=acc_valleyprom_win80
                acc_features['acc_domfreq1']=acc_domfreq1
                acc_features['acc_kurt_win30']=acc_kurt_win30
                acc_features=acc_features
    
            if s==1:
                print("gyroscope features")
                gyro_features['gyro_skew_win70']=gyro_skew_win70
                gyro_features['gyro_skew_win30']=gyro_skew_win30
                gyro_features['gyro_indMax_win60']=gyro_maxindex_win60
                gyro_features['gyro_SMA_win80']=gyro_SMA_win80
                gyro_features['gyro_var_win20']=gyro_var_win20
                gyro_features['gyro_Max_win70']=gyro_maxvalue_win70
                gyro_features['gyro_valley_prominences80']=gyro_valleyprom_win80
                gyro_features['gyro_max_freq_val']=gyro_maxfreqvalue
                gyro_features=gyro_features
            s=s+1
            
        return gyro_features,acc_features
        
    def predict_steps(self,gyro_features,acc_features):

        feat_gyro=['gyro_indMax_win60',
         'gyro_skew_win70',
         'gyro_max_freq_val',
         'gyro_SMA_win80',
         'gyro_skew_win30',
         'gyro_var_win20',
         'gyro_Max_win70',
         'gyro_valley_prominences80']
    
        feat_acc=['acc_indMin_win30', 'acc_skew_win5', 'acc_Median_win80',
                'acc_kurt_win30',
               'acc_valley_prominences80', 'acc_peak_prominences80',
               'acc_peak_prominences50',  'acc_domfreq1']
    
        joblib_file = "d:\\Users\\al-abiad\\Desktop\\experiments\\treadmill\\machine_learning_features\\joblib_Model_accv3.pkl" 
        Model_acc = joblib.load(joblib_file)
        joblib_file = "d:\\Users\\al-abiad\\Desktop\\experiments\\treadmill\\machine_learning_features\\joblib_Model_gyrov3.pkl" 
        Model_gyro = joblib.load(joblib_file)
        steps_in_window=[]
        maxprobacc=0.5
        maxprobgyro=0.5
        index=0
        steps_in_window=[]
        step_type=[]
        maxprobacc=0.5
        maxprobgyro=0.5

        index=0
        while index< len(gyro_features):
    
    
            featgyro=gyro_features.iloc[index,:][feat_gyro]
            featacc=acc_features.iloc[index,:][feat_acc]
    
    
    
            y_predprobacc= Model_acc.predict_proba([featacc])
    
            y_predprobgyro= Model_gyro.predict_proba([featgyro])
    
            if (y_predprobgyro[0][1]>maxprobgyro):
                steps_in_window.append(1)
                step_type.append("gyro")
                print("Model is gyro")
                index=index+1
                distance=0
                while distance<80 and index<len(gyro_features):
                    print("entered loop")
                    featgyro=gyro_features.iloc[index,:][feat_gyro]
                    y_predprobgyro= Model_gyro.predict_proba([featgyro])
    
                    if(y_predprobgyro[0][1]>maxprobgyro):
                        print("step is gyro")
                        steps_in_window.append(1)
                        step_type.append("gyro")
                        distance=0
                    else:
                        distance=distance+1
                        steps_in_window.append(0)
                    index=index+1
    
                if index<len(gyro_features):
                    print("we entered condition")
                    index=index-50
                    steps_in_window=steps_in_window[:-50]
    
    
            elif (y_predprobacc[0][1]>maxprobacc):
                steps_in_window.append(1)
                step_type.append("acc")
                print("Model is acc")
                index=index+1
                distance=0
                print(index)
                while distance<80 and index<len(gyro_features):
                    featacc=acc_features.iloc[index,:][feat_acc]
                    y_predprobacc= Model_acc.predict_proba([featacc])
                    print(index)
                    if(y_predprobacc[0][1]>maxprobacc):
                        print("step is acc")
                        steps_in_window.append(1)
                        step_type.append("acc")
                        distance=0
                    else:
                        distance=distance+1
                        steps_in_window.append(0)
                    index=index+1
    
                if index<len(gyro_features):
                    print("we entered condition")
                    index=index-50
                    steps_in_window=steps_in_window[:-50]
    
    
            else:
                steps_in_window.append(0)
                print("irregular or static")
                index=index+1
                
        return(steps_in_window,step_type)
    
    def inverse_window_step(self,windows):
        steps=[]
        ii=0
        for wind in windows:
    
            if wind==1:
                steps.append(64+ii)
            ii=ii+1
        return steps
    

        
        
            
                
        
if __name__=="__main__":
    plt.close('all')
    
    #---TEST THREAMILL OBJECT---
    if True:
        path="d:\\Users\\al-abiad\\Desktop\\zahertest\\handnoor.csv"
        Test = phone(path,app="T")
#        Test.interpolategyrnacc()
        xx=Test.acc_rawdata
        yy=Test.gyro_rawdata
        plt.figure()
        plt.plot(xx)
        plt.figure()
        plt.plot(yy)
#        Test.filter_data(acc=Test.acc_interp,gyro=Test.gyro_interp)#,N=10,fc=3,fs=100)
#        z=Test.gyro_filtered
#        w=Test.acc_filtered
#        plt.figure()
#        plt.plot(z)
#        plt.figure()
#        plt.plot(w)
        Test.calculate_norm_accandgyro(gyro=Test.gyro_rawdata,acc=Test.acc_rawdata)
        nx=Test.acc_magnitude
        ny=Test.gyro_magnitude
        plt.figure()
        plt.plot(ny)
        plt.title("gyroscope signal")
        plt.xlabel("number of samples")
        plt.ylabel("[rad/s]")
#        Test.gtom2s()
        
#        Test.calculate_norm()
        

       
    
    #---USE IT---
        
    
        
        
        
        
        
        
        