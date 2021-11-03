# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:51:20 2021

@author: al-abiad
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:38:51 2021

@author: al-abiad
"""
import matplotlib.pyplot as plt
import os
from os import walk
from scipy import signal,fft
from scipy.signal import hilbert
import tkinter as tk
import pygubu
from tkinter import messagebox
from CPclass import phone as CP
from scipy.signal import find_peaks
import numpy as np

from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)# Implement the default Matplotlib key bindings.
from matplotlib.figure import Figure

import pandas as pd

from openpyxl import load_workbook
from openpyxl.styles import Font, Color
from openpyxl.styles import colors

#nonlinear
from nolitsa import data, delay,dimension
import nolds as nld

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
pickable_artists = []


class MyApplication:
    
    def __init__(self):
        
        self.walking_period=[]

        #1: Create a builder
        self.builder = builder = pygubu.Builder()

        #2: Load an ui file
        builder.add_from_file(os.path.join(CURRENT_DIR, 'notebook23.ui'))
        
        #3: Create the toplevel widget.
        self.mainwindow = builder.get_object('mainwindow')

        # Container for the matplotlib canvas and toolbar classes
        self.fcontainer = builder.get_object('fcontainer')
        self.fcontainer2 = builder.get_object('fcontainer2')
        self.fcontainer3 = builder.get_object('fcontainer3')
        self.fcontainer4 = builder.get_object('fcontainer4')
        self.fcontainer_gaitupfeet= builder.get_object('fcontainer_gaitupfeet')
        self.fcontainer_gaituphand= builder.get_object('fcontainer_gaituphand')
        self.fcontainer_phonewaist= builder.get_object('fcontainer_phonewaist')
        self.fcontainer_allsignals=builder.get_object('fcontainer_allsignals')
        self.fcontainer_signalwithstep=builder.get_object('fcontainer_signalwithstep')
        #set filepathchooser
        self.filepath = builder.get_object('filepath')
        self.filepath_export=builder.get_object('path_choose_Export')
        
        #set combo_box
        # self.accunit_combo = builder.get_object('combobox_1')
        # options = ['Select an option', 'm/s^2', 'g']
        # self.accunit_combo.config(values=options)
        
        
        # self.gyrounit_combo = builder.get_object('combobox_2')
        # options = ['Select an option', 'rad/s', 'deg/s']
        # self.gyrounit_combo.config(values=options)
        
        self.plot_combo = builder.get_object('combobox_4')
        options = ['Select an option', 'Acceleration Norm', 'Gyroscope norm' ]
        self.plot_combo.config(values=options)
        
        #get labels
        self.lbl_files=builder.get_object('lbl_files')
        self.lbl_SDstride=builder.get_object('SD_stride')
        self.lbl_CVstride=builder.get_object('CV_stride')
        self.lbl_Nstride=builder.get_object('N_stride')
        self.lbl_Nphase=builder.get_object('N_phase')
        self.lbl_lyapx=builder.get_object('lbl_lyapx')
        self.lbl_lyapy=builder.get_object('lbl_lyapy')
        self.lbl_lyapz=builder.get_object('lbl_lyapz')
        self.lbl_lyapr=builder.get_object('lbl_lyapr')
        
        #get entries
        self.ent_str_walk = self.builder.get_object('str_walk')
        self.ent_stp_walk = self.builder.get_object('stp_walk')
        self.ent_turn = self.builder.get_object('step_turn')
        self.ent_embdim = self.builder.get_object('ent_embdim')
        self.ent_timedelay = self.builder.get_object('ent_timedelay')
        self.ent_Nbstridelyap = self.builder.get_object('ent_Nbstridelyap')
        self.ent_turntime=self.builder.get_object('ent_turntime')
        self.ent_str_gaitup_feet=self.builder.get_object('str_gaitup_feet')
        self.ent_str_phone_waist=self.builder.get_object('str_phone_waist')
        self.ent_str_gaitup_hand=self.builder.get_object('str_gaitup_hand')
        
        
        
        #get treeview
        self.treeview=self.builder.get_object('myetv')
        self.treeview_res=self.builder.get_object('Result_treeview')
        self.treeview_stat=self.builder.get_object('stat_treeview')
        
        
        #buttons
        self.btn_cal_norm=self.builder.get_object('btn_cal_norm')
        self.btn_plot=self.builder.get_object('btn_plot')
        self.btn_detect_start_stop=self.builder.get_object('btn_detect_start_stop')
        self.btn_detect_step=self.builder.get_object('btn_detect_step')
        self.btn_detect_turn=self.builder.get_object('btn_detect_turn')
        self.btn_cal_lyap=self.builder.get_object('btn_cal_lyap')
        self.btn_cal_time_delay=self.builder.get_object('btn_cal_time_delay')
        self.btn_cal_emb_dim=self.builder.get_object("btn_cal_emb_dim")

        # Connect button callback
        builder.connect_callbacks(self)
        
        


        
    def on_click_canvas(self,event,mode="detect_turns"):
#        print('-----')
#        print('button:', event.button)
#        print('xdata, ydata:', event.xdata, event.ydata)
#        print('x, y:', event.x, event.y)
#        print('canvas:', event.canvas)
        print('inaxes:', event.inaxes)
        
        if self.toolbar.mode =="":
            print("toolbar is not selected")
            if mode=="detect_walking_period":
                print("detecting walking period")
                if event.inaxes is not None and not hasattr(event, 'already_picked'):
                    remove = [artist for artist in pickable_artists if artist.contains(event)[0]]
                    print("--remove--")
                    print(remove)
                    print("---pickable_artists---")
                    print(pickable_artists)
                    x=event.xdata.astype('int')
                    if not remove:
                        # add a pt
                        pt = self.a.scatter(x, 10,color='red',marker="|",s=10000000000,zorder=2,picker=5)
                        pickable_artists.append(pt)
                        self.walking_period.append(x)
                        print(self.walking_period)
                    else:
                        #remove a point
                        pickable_artists.remove(remove[0])
                        self.walking_period=[i for i in self.walking_period if i>x+50 or i<x-50]
                        for artist in remove:
                            artist.remove()
                    self.canvas.draw()
                else:
                    print ('Clicked ouside axes bounds but inside plot window')
                
            elif mode=="detect_turns":

                if event.inaxes is not None and not hasattr(event, 'already_picked'):
                    
                    remove = [artist for artist in pickable_artists if artist.contains(event)[0]]
                    print("--remove--")
                    print(remove)
                    print("---pickable_artists---")
                    print(pickable_artists)
                    x=event.xdata.astype('int')
                    
                    if not remove:
                        # add a pt
                        pt = self.a.scatter(x, 10,color='red',marker="|",s=10000000000,zorder=2,picker=5)
                        pickable_artists.append(pt)
                        self.peaks2=np.append(self.peaks2,x)
                        self.refresh_treeview()
                    else:
                        self.peaks2=[i for i in self.peaks2 if i>x+50 or i<x-50]
                        self.refresh_treeview()
                        pickable_artists.remove(remove[0])
                        for artist in remove:
                            artist.remove()
                    self.canvas.draw()
                else:
                    print ('Clicked ouside axes bounds but inside plot window')
        else:
            print("toolbar is selected")
            
            
    def refresh_treeview(self):
        turn=int(self.ent_turn.get())
        self.phase=[]
        self.peaks2=np.sort(self.peaks2)
        for i in range(0,len(self.peaks2)-1):
            self.phase.append((i+1,self.peaks2[i],self.peaks2[i+1],turn))
            
        if (len(self.treeview.get_children())!=0):
            self.treeview.delete(*self.treeview.get_children())
            
        for d in self.phase:
            self.treeview.insert('', tk.END, values=d)
        
            
    def undo_turn(self,event):
        self.turns.pop()
        
        
    def on_path_changed(self, event=None):
        # Get the path choosed by the user
        print("you are in on path changed")
        self.lyap_NBstrides=0
        self.preprocessed=False
        self.btn_cal_norm["state"] = "disabled"
        self.btn_plot["state"] = "disabled"
        self.btn_detect_start_stop["state"] = "disabled"
        self.btn_detect_step["state"] = "disabled"
        self.btn_detect_turn["state"] = "disabled"
        
        self.path = self.filepath.cget('path')
        self.export_path=self.path
        
        phone_dir=os.path.join(self.path,"telephone" )
        for root, dirs, files in os.walk(phone_dir):
            for file in files:
                if file.endswith(".txt"):
                    self.phone_direct=root
                break
        #phone directory is self.phone_direct
        print(self.phone_direct)
        
        #gaitup directory
        gaitup_dir=os.path.join(self.path,"Gaitup" )
        
        # show paths in label
        _, _, self.files = next(walk(self.phone_direct))
        self.lbl_files.configure(text = 'The files are %s ,%s and %s'%(self.files[0],self.files[1],self.files[2]))
        
        
    def on_path_changed_export(self, event=None):
        
        self.export_path= self.filepath_export.cget('path')
        print("export path is changed")

        
    def Interpolate_and_filter(self,event=None):
        #acceleration signal
        self.CP_data=CP(self.phone_direct,app="geoloc")
        self.CP_data.interpolategyrnacc()
        self.CP_data.rad2deg()
        self.CP_data.filter_data(acc=self.CP_data.acc_interp,gyro=self.CP_data.gyro_interp,N=10,fc=3,fs=100)
        self.btn_cal_norm["state"] = "normal"

        
    def calculate_norm(self, event=None):
        self.CP_data.calculate_norm_accandgyro(gyro=self.CP_data.gyro_filtered,acc=self.CP_data.acc_filtered)
        self.btn_plot["state"] = "normal"
        self.btn_detect_start_stop["state"] = "normal"
        self.plot_graph(plot_name="x")


    
    def Detect_startandend(self,event=None):
        
        walk_end=int(len(self.CP_data.acc_magnitude)-self.CP_data.detectstartofwalk(sig1=self.CP_data.acc_magnitude[::-1])*100)//100
        print(walk_end)
        walk_start=self.CP_data.detectstartofwalk(sig1=self.CP_data.acc_magnitude)
        print(walk_start)
        self.ent_str_walk.delete(0,tk.END)
        self.ent_str_walk.insert(0,walk_start)
        self.ent_stp_walk.delete(0,tk.END)
        self.ent_stp_walk.insert(0,walk_end)
        self.btn_detect_turn["state"] = "normal"
        
    def Detect_turns(self,method="gyro_peak",event=None):
        method="gyro_peak"
        if method=="gyro_peak":
            self.phase=[]
            str_walk=int(float(self.ent_str_walk.get()))*100
            stp_walk=int(self.ent_stp_walk.get())*100
            turn=int(self.ent_turn.get())
            
            self.CP_data.peakdet_m2(acc=False,plot_peak=True,detect_turn=True)
            self.peaks=self.CP_data.peakandvalley["peak_index"]
    
            self.peaks=self.peaks[self.peaks>str_walk]
            self.peaks=self.peaks[self.peaks<stp_walk]
            
            self.peaks2 = [str_walk, *self.peaks, stp_walk]
    
            for i in range(0,len(self.peaks2)-1):
                self.phase.append((i+1,self.peaks2[i],self.peaks2[i+1],turn))
             
            self.peaks2=np.array(self.peaks2)
            
            if (len(self.treeview.get_children())!=0):
                self.treeview.delete(*self.treeview.get_children())
                
            for d in self.phase:
                self.treeview.insert('', tk.END, values=d)
    
            self.btn_detect_step["state"] = "normal"
            
            print(len(self.treeview.get_children()))
            
            for p in self.phase:
                pt = self.a.scatter(p[1], 10,color='red',marker="|",s=10000000000,zorder=2,picker=5)
                pickable_artists.append(pt)
                
                pt = self.a.scatter(p[2], 10,color='red',marker="|",s=10000000000,zorder=2,picker=5)
                pickable_artists.append(pt)
                
            self.canvas.draw()
        elif method=="extract gaitup":
            print("extracting from gaitup")
            
            
            
        elif method=="gyro_turn_period":
            print("gyro_turn_period_hilbert")
            self.CP_data.filter_data(acc=self.CP_data.acc_interp,gyro=self.CP_data.gyro_interp,N=10,fc=3,fs=100)
            self.CP_data.calculate_norm_accandgyro(gyro=self.CP_data.gyro_filtered,acc=self.CP_data.acc_filtered)
            
            str_walk=int(float(self.ent_str_walk.get()))*100
            stp_walk=int(self.ent_stp_walk.get())*100
            # low_idx, high_idx =self.hl_envelopes_idx(self.CP_data.gyro_magnitude, 
            #                                           dmin=1, dmax=1, split=False)
            
            

            # plt.figure()
            # plt.plot(high_idx,self.CP_data.gyro_magnitude[high_idx])
            # plt.plot(self.CP_data.gyro_magnitude)
            # plt.plot(amplitude_envelope_acc)

            #########################################################
            turns=self.window_rms(self.CP_data.gyro_magnitude, window_size=300)
            peak_index,_= find_peaks(turns[str_walk:stp_walk],distance=500)
            
            print(peak_index)
            gyro_max=np.mean(turns[str_walk:stp_walk][peak_index])
            
            # gyro_max=np.mean(turns[str_walk:stp_walk])
            print(gyro_max)
            gyro_binary=np.where(turns>gyro_max)[0]
            ranges=self.ranges(gyro_binary)
            turn_array=np.zeros(len(self.CP_data.gyro_magnitude))

            for (start,end) in ranges:
                if start+50<end:
                    turn_array[start:end]=1
            
            #########################################################
            # walking_zone=[(self.walking_period[0],self.walking_period[1])]
            # accc,gyroo=self.CP_data.crop_medipole(fs=1,phases=walking_zone,turn_time=0)
            
            # self.CP_data.filter_data(acc=accc[0],gyro=gyroo[0],N=10,fc=12,fs=100)
            # self.CP_data.calculate_norm_accandgyro(gyro=self.CP_data.gyro_filtered,acc=self.CP_data.acc_filtered)

            # gyro_mag=self.CP_data.gyro_magnitude
            # acc_mag=self.CP_data.acc_magnitude
            # gyro_max=np.amax(gyro_mag)
            # acc_mean=np.mean(acc_mag)
            
            # self.CP_data.filter_data(acc=self.CP_data.acc_interp,gyro=self.CP_data.gyro_interp,N=10,fc=12,fs=100)
            # self.CP_data.calculate_norm_accandgyro(gyro=self.CP_data.gyro_filtered,acc=self.CP_data.acc_filtered)
            
            # gyro_mag=self.CP_data.gyro_magnitude
            # acc_mag=self.CP_data.acc_magnitude
            
            # gyro_binary=np.where(self.CP_data.gyro_magnitude>gyro_max)[0]
            # # acc_binary=[self.CP_data.acc_magnitude<acc_mean]
            # # gyro_binary=np.transpose(np.array(gyro_binary).astype("int"))
            
            # ranges=self.ranges(gyro_binary)
            
            # turn_array=np.zeros(len(gyro_mag))

            # for (start,end) in ranges:
            #     if start+50<end:
            #         turn_array[start:end]=1
            
            plt.figure()
            plt.plot(self.CP_data.gyro_magnitude)
            plt.plot(turn_array*100)
            plt.plot(turns)
            
            self.turns=turn_array
            
            turn_array=turn_array+np.mean(self.CP_data.acc_magnitude)
            

            self.a.plot(turn_array, color='red')

 
            self.canvas.draw()
                    
                    
                    
            
    def window_rms(self,a, window_size):
        a2 = np.power(a,2)
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(a2, window, 'same'))
            
            


    def hl_envelopes_idx(self,s, dmin=1, dmax=1, split=False):
        """
        Input :
        s: 1d-array, data signal from which to extract high and low envelopes
        dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
        split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
        Output :
        lmin,lmax : high/low envelope idx of input signal s
        """
    
        # locals min      
        lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
        # locals max
        lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
        
    
        if split:
            # s_mid is zero if s centered around x-axis or more generally mean of signal
            s_mid = np.mean(s) 
            # pre-sorting of locals min based on relative position with respect to s_mid 
            lmin = lmin[s[lmin]<s_mid]
            # pre-sorting of local max based on relative position with respect to s_mid 
            lmax = lmax[s[lmax]>s_mid]
    
    
        # global max of dmax-chunks of locals max 
        lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
        # global min of dmin-chunks of locals min 
        lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
        
        return lmin,lmax
            
            
    def ranges(self,nums,d=200):
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+d < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        return list(zip(edges, edges))
                
            
        
    def Save_and_Export(self,event=None):
        
        self.exported_results.to_csv(self.export_path+"\\_exported_results.csv",index=False,header=True)
        self.exported_results_res.to_csv(self.export_path+"\\_exported_stride_results.csv",index=False,header=True)
        print("saved")
        
        
    def Detect_steps(self,event=None):
        phases_adj=[]
        turn_strides=[]
        phases=[]
        tot_peaks=[]
        
        for line in self.treeview.get_children():
            phases.append(self.treeview.item(line)['values'])
        
        
        if (len(self.treeview_res.get_children())!=0):
            self.treeview_res.delete(*self.treeview_res.get_children())
        
        if (len(self.treeview_stat.get_children())!=0):
            self.treeview_stat.delete(*self.treeview_stat.get_children())
            
        for (x,start,stop,z) in phases:
            start=int(start)
            stop=int(stop)
            z=int(z)
            phases_adj.append((start,stop))
            turn_strides.append(z)
            

        accc,gyroo=self.CP_data.crop_medipole(fs=1,phases=phases_adj,turn_time=int((self.ent_turntime.get())))
    
        
        df = pd.concat(accc,ignore_index=True)
        plt.figure()
        plt.plot(df)
                
        for i in range(0,len(accc)):
            self.CP_data.filter_data(acc=accc[i],gyro=gyroo[i],N=10,fc=3,fs=100)
            
            self.CP_data.calculate_norm_accandgyro(gyro=self.CP_data.gyro_filtered,acc=self.CP_data.acc_filtered)
    
            self.CP_data.peakdet_m2(acc=True,plot_peak=False,detect_turn=False)
            
            
            cum_ind=0
            for kk in range(0,i):
                cum_ind=cum_ind+len(accc[kk])
            
            #remove 2 steps from beginging and end 
            peaks=self.CP_data.peakandvalley["peak_index"]+cum_ind
            
            tot_peaks.append(peaks)
            
            
            self.CP_data.computeVarStride(fs=100,remove_outliers=True,N=3,use_peaks=True,
                                          pocket=False,remove_step=turn_strides[i])
            
#            self.CP_data.plot_results(pocket=False)
        
            result=self.CP_data.cycle_temp
            
            # Numberofsteps.append(len(result["steptime"])+1)
            
            result_adj=[]
            for k in range(0,len(self.CP_data.cycle_temp['stridetime'])):
                result_adj.append((i,k+1,self.CP_data.cycle_temp['stridetime'][k][0]))
                
            for d in result_adj:
                self.treeview_res.insert('', tk.END, values=d)
             
            d=((i,np.around(np.mean(self.CP_data.cycle_temp['stridetime']),decimals=3),self.CP_data.cycle_temp['stridetime_std'],self.CP_data.cycle_temp['stridetime_Cov'],len(result["stridetime"])))   
            self.treeview_stat.insert('',tk.END,values=d)
            
        self.tot_peaks=np.hstack(tot_peaks)

        print(self.treeview_stat.get_children())
        exported_list=[]
        
        for line in self.treeview_stat.get_children():
            exported_list.append(self.treeview_stat.item(line)['values'])
            
        self.exported_results=pd.DataFrame(exported_list, columns=["Phase", "Mean stride duration", "standard deviation of stride duration","Coefficient of variance of stride duration", "Number of strides"])
        
        # print(self.exported_results)
        
        exported_list=[]
        
        for line in self.treeview_res.get_children():
            exported_list.append(self.treeview_res.item(line)['values'])
            
        # print(exported_list)
            
        self.exported_results_res=pd.DataFrame(exported_list, columns=["Phase","stride number", "stride duration"])
        
        # print(self.exported_results)
        
        self.exported_results_res['stride duration']=pd.to_numeric(self.exported_results_res['stride duration'], downcast='float')
        
        self.lbl_SDstride.configure(text ='%.2f milliseconds'%(np.round(np.std(self.exported_results_res['stride duration'].values),decimals=5)*1000))
        self.lbl_CVstride.configure(text ='%.2f %%'%(np.round(np.std(self.exported_results_res['stride duration'].values)/np.mean(self.exported_results_res['stride duration'].values),decimals=5)*100))
        self.lbl_Nstride.configure(text ='%d'%(len(self.exported_results_res['stride duration'].values)))
        self.lbl_Nphase.configure(text ='%d'%(len(self.exported_results)))
        
        self.plot_stridetime()
        
               
    def on_row_edit(self, event):
        # Get the column id and item id of the cell
        # that is going to be edited

        col, item = self.treeview.get_event_info()
        print(col)
        print(item)
        

        # Define the widget editor to be used to edit the column value
        if col in ('Start','Stop','Turning'):
            self.treeview.inplace_entry(col, item)

    def on_cell_changed(self, event):
        col, item = self.treeview.get_event_info()
        
        print('Column {0} of item {1} was changed'.format(col, item))

    def on_row_selected(self, event):
        print('Rows selected', event.widget.selection())
        
        
    def plot_stridetime(self):
        self.figure2 = Figure(figsize=(6, 5), dpi=100)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=self.fcontainer2)
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas2, self.fcontainer2)
        self.toolbar.update()
        self.canvas2._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.b = self.figure2.add_subplot(111)
            
        self.b.plot(self.exported_results_res['stride duration'],zorder=1)
        self.b.set_ylabel('Stride duration (s)')
        self.b.set_xlabel('Stride number')
        self.b.axhline(y=np.mean(self.exported_results_res['stride duration']),label='Average Stride Time',color='r')
        self.b.axhline(y=np.mean(self.exported_results_res['stride duration'])-np.std(self.exported_results_res['stride duration']),linestyle='--',color='r',linewidth=0.5)
        self.b.axhline(y=np.mean(self.exported_results_res['stride duration'])+np.std(self.exported_results_res['stride duration']),linestyle='--',color='r',linewidth=0.5)
        self.b.text(0.8, 0.8,("Stride std:%s, Stride cov:%s"%(self.lbl_SDstride.cget("text"),self.lbl_CVstride.cget("text"))),size=10,transform=self.figure2.transFigure,ha="right", va="top", bbox=dict(facecolor='red', alpha=0.5))
        self.canvas2.draw()
        

        
    def plot_gaitup_phone(self):
        print("not done")
        
        self.figure1 = Figure(figsize=(4, 3), dpi=100)
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=self.fcontainer_gaituphand)
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas1, self.fcontainer_gaituphand)
        self.toolbar.update()
        self.canvas1._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        color = 'tab:blue'
        if self.hand:
            b = self.figure1.add_subplot(111)
            b.plot(self.acc_h_magnitude,zorder=1)
            b.set_ylabel('Acc m/s\u00b2', color=color)
            b.title.set_text("Hand")
            b.set_xticks([])
            s_h=int((self.ent_str_gaitup_hand.get()))
            b.scatter(s_h,self.acc_h_magnitude[s_h],color='red',marker="|",s=10000000000,zorder=2)
            b.set(xlim=(s_h-1000, s_h+1000))
            self.canvas1.draw()
        
        self.figure2 = Figure(figsize=(4, 3), dpi=100)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=self.fcontainer_gaitupfeet)
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas2, self.fcontainer_gaitupfeet)
        self.toolbar.update()
        self.canvas2._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        color = 'tab:blue'
        b = self.figure2.add_subplot(111)
        b.plot(self.acc_rf_magnitude,zorder=1)
        b.plot(self.acc_lf_magnitude,zorder=1)
        b.set_ylabel('Acc m/s\u00b2', color=color)
        b.title.set_text("feet")
        b.set_xticks([])
        s_f=int((self.ent_str_gaitup_feet.get()))
        b.scatter(s_f,self.acc_lf_magnitude[s_f], color='red',marker="|",s=10000000000,zorder=2)
        b.set(xlim=(s_f-1000, s_f+1000))
        self.canvas2.draw()
        
        self.figure3 = Figure(figsize=(4, 3), dpi=100)
        self.canvas3 = FigureCanvasTkAgg(self.figure3, master=self.fcontainer_phonewaist)
        self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas3, self.fcontainer_phonewaist)
        self.toolbar.update()
        self.canvas3._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        color = 'tab:blue'
        b = self.figure3.add_subplot(111)
        b.plot(self.acc_w_magnitude,zorder=1)
        b.set_ylabel('Acc m/s\u00b2', color=color)
        b.title.set_text("waist")
        b.set_xticks([])
        s_w=int((self.ent_str_phone_waist.get()))
        b.scatter(s_w,self.acc_w_magnitude[s_w],color='red',marker="|",s=10000000000,zorder=2)
        b.set(xlim=(s_w-1000, s_w+1000))
        
        self.canvas3.draw()
        
    def read_gaitup_csv(self):
        gaitup_dir=os.path.join(self.path,"Gaitup" )
        self.hand=True
        #hand norm
        filename=os.path.join(gaitup_dir,"hand.csv" )
        col_list = ["Time", "Gyro X","Gyro Y","Gyro Z","Accel X", "Accel Y","Accel Z"]
        try:
            df_h=pd.read_csv(filename, delimiter=",",skiprows=[0],usecols=col_list)

        except:
            self.hand=False
            print("no hand file")
            
        if self.hand:
            df_h = df_h.iloc[1:]
            df_h=df_h.astype('float32')
            
            self.df_h=df_h
            time=self.df_h['Time'].values
            #interpolate and filter
            
            self.df_h=self.df_h.set_index('Time')
            fs=100
            t_t=np.linspace(0, time[-1], num=np.int(time[-1]*fs), 
                            endpoint=True,dtype=np.float32)
    
            self.df_h=self.df_h.reindex(self.df_h.index.union(t_t))
            self.df_h=self.df_h.interpolate(method='linear', limit_direction='both',
                                            axis=0)
            self.df_h=self.df_h[self.df_h.index.isin(pd.Index(t_t))]
    
            self.df_h.index=np.around(self.df_h.index.astype('float32'),decimals=4)
    
            time =self.df_h.index
            
            acc_h=self.df_h.iloc[:,3:6]
            acc_h=acc_h.mul(9.8)
            gyro_h=self.df_h.iloc[:,0:3]
            
            self.CP_data_h=CP(acc=acc_h,gyro=gyro_h,app="manual_entry")
            self.CP_data_h.filter_data(acc=self.CP_data_h.acc_interp,gyro=self.CP_data_h.gyro_interp,N=10,fc=2,fs=100)
            self.CP_data_h.calculate_norm_accandgyro(gyro=self.CP_data_h.gyro_filtered,acc=self.CP_data_h.acc_filtered)
            self.acc_h_magnitude=self.CP_data_h.acc_magnitude
            self.gyro_h_magnitude=self.CP_data_h.gyro_magnitude
            
            print(self.acc_h_magnitude)
            
            s_h=self.detectstartofwalk(self.acc_h_magnitude,thresh=3)
            self.ent_str_gaitup_hand.delete(0,tk.END)
            self.ent_str_gaitup_hand.insert(0,s_h)
        
        filename=os.path.join(gaitup_dir,"left_foot.csv" ) 
        # left foot
        col_list = ["Time", "Gyro X","Gyro Y","Gyro Z","Accel X", "Accel Y","Accel Z"]
        df_lf=pd.read_csv(filename, delimiter=",",skiprows=[0],usecols=col_list)
        df_lf = df_lf.iloc[1:]
        df_lf=df_lf.astype('float32')
        
        self.df_lf=df_lf
        time=self.df_lf['Time'].values
        #interpolate and filter
        
        self.df_lf=self.df_lf.set_index('Time')
        fs=100
        t_t=np.linspace(0, time[-1], num=np.int(time[-1]*fs), 
                        endpoint=True,dtype=np.float32)

        self.df_lf=self.df_lf.reindex(self.df_lf.index.union(t_t))
        self.df_lf=self.df_lf.interpolate(method='linear', limit_direction='both',
                                        axis=0)
        self.df_lf=self.df_lf[self.df_lf.index.isin(pd.Index(t_t))]

        self.df_lf.index=np.around(self.df_lf.index.astype('float32'),decimals=4)

        time =self.df_lf.index
        
        acc_lf=self.df_lf.iloc[:,3:6]
        gyro_lf=self.df_lf.iloc[:,0:3]
        
        self.CP_data_lf=CP(acc=acc_lf,gyro=gyro_lf,app="manual_entry")
        self.CP_data_lf.calculate_norm_accandgyro(gyro=self.CP_data_lf.gyro_interp,acc=self.CP_data_lf.acc_interp)
        self.acc_lf_magnitude=self.CP_data_lf.acc_magnitude
        self.gyro_lf_magnitude=self.CP_data_lf.gyro_magnitude

        s_lf=self.detectstartofwalk(self.acc_lf_magnitude,thresh=5)
        
        filename=os.path.join(gaitup_dir,"right_foot.csv" )
        #right foot
        col_list = ["Time", "Gyro X","Gyro Y","Gyro Z","Accel X", "Accel Y","Accel Z"]
        df_rf=pd.read_csv(filename, delimiter=",",skiprows=[0],usecols=col_list)
        df_rf = df_rf.iloc[1:]
        df_rf=df_rf.astype('float32')
        
        self.df_rf=df_rf
        
        time=self.df_rf['Time'].values
        #interpolate and filter
        
        self.df_rf=self.df_rf.set_index('Time')
        fs=100
        t_t=np.linspace(0, time[-1], num=np.int(time[-1]*fs), 
                        endpoint=True,dtype=np.float32)

        self.df_rf=self.df_rf.reindex(self.df_rf.index.union(t_t))
        self.df_rf=self.df_rf.interpolate(method='linear', limit_direction='both',
                                        axis=0)
        self.df_rf=self.df_rf[self.df_rf.index.isin(pd.Index(t_t))]

        self.df_rf.index=np.around(self.df_rf.index.astype('float32'),decimals=4)

        time =self.df_rf.index
        
        acc_rf=self.df_rf.iloc[:,3:6]
        gyro_rf=self.df_rf.iloc[:,0:3]
        
        self.CP_data_rf=CP(acc=acc_rf,gyro=gyro_rf,app="manual_entry")
        self.CP_data_rf.calculate_norm_accandgyro(gyro=self.CP_data_rf.gyro_interp,acc=self.CP_data_rf.acc_interp)
        self.acc_rf_magnitude=self.CP_data_rf.acc_magnitude
        self.gyro_rf_magnitude=self.CP_data_rf.gyro_magnitude

        s_rf=self.detectstartofwalk(self.acc_rf_magnitude,thresh=5)
        
        s_f=np.minimum(s_rf,s_lf)
        self.ent_str_gaitup_feet.delete(0,tk.END)
        self.ent_str_gaitup_feet.insert(0,s_f)
        
        ###phone
        self.CP_data_phone=CP(self.phone_direct,app="geoloc")
        self.CP_data_phone.interpolategyrnacc(fs=100)
        self.CP_data_phone.filter_data(acc=self.CP_data_phone.acc_interp,gyro=self.CP_data_phone.gyro_interp,N=10,fc=3,fs=100)
        self.CP_data_phone.calculate_norm_accandgyro(gyro=self.CP_data_phone.gyro_filtered,acc=self.CP_data_phone.acc_filtered)
        self.acc_w_magnitude=self.CP_data_phone.acc_magnitude
        
        s_w=self.detectstartofwalk(self.acc_w_magnitude,thresh=2)
        
        self.ent_str_phone_waist.delete(0,tk.END)
        self.ent_str_phone_waist.insert(0,s_w)

    def plot_and_calculate_start(self,event=None):
        print("doing")
        self.read_gaitup_csv()
        self.plot_gaitup_phone()
        
        
    def read_gaitup_excel(self,event=None):
        
        gaitup_dir=os.path.join(self.path,"Gaitup_turn" )
        for root, dirs, files in os.walk(gaitup_dir):
            for file in files:
                if file.endswith(".xlsx"):
                    if 'right' in file:
                        right_step_file= file
                        print(file)
                    if 'left' in file:
                        left_step_file= file
                        print(file)
        #right 
        right_file=os.path.join(gaitup_dir,right_step_file)
        
        
        wb_right= load_workbook(right_file)
        ws_right=wb_right["Sheet1"]
        
        start_analysis_gaitup_right = ws_right['B14'].value
        stop_analysis_gaitup_right=ws_right['D14'].value
        
        ColD_right=ws_right['D']
        ColD_right=ColD_right[25::]
        
        turns_HS_right=[]
        straightwalk_HS_right=[]
        i=0
        for cl in ColD_right:
            if cl.value!= None:
                print(cl.value)
                i=i+1
                if cl.font.color != None and type(cl.font.color.rgb) == str:
                    x=cl.value+start_analysis_gaitup_right
                    turns_HS_right.append([i,x])
                else:
                    x=cl.value+start_analysis_gaitup_right
                    straightwalk_HS_right.append([i,x])
                
        turns_HS_right=np.vstack(turns_HS_right)
        straightwalk_HS_right=np.vstack(straightwalk_HS_right)
        
        #get index of straight walking steps in the main csv signal
        time=self.CP_data_rf.acc_interp.index.values
        straightwalk_HS_right_index=[]
        for hs in straightwalk_HS_right[:,1]:
            t1=np.where(time>hs)[0][0]
            t2=t1-1
            if np.abs(hs-time[t1])<=np.abs(hs-time[t2]):
                straightwalk_HS_right_index.append(t1)
            else:
                straightwalk_HS_right_index.append(t2)
        
        #get index of turn walking steps in the main csv signal
        Turn_HS_right_index=[]
        for hs in turns_HS_right[:,1]:
            t1=np.where(time>hs)[0][0]
            t2=t1-1
            if np.abs(hs-time[t1])<=np.abs(hs-time[t2]):
                Turn_HS_right_index.append(t1)
            else:
                Turn_HS_right_index.append(t2)
                
        straight_walking_periods_right=self.ranges(straightwalk_HS_right[:,0])
        
        turning_periods_right=self.ranges(turns_HS_right[:,0])
                
        
        #left
        left_file=os.path.join(gaitup_dir,left_step_file)
        
        
        wb_left= load_workbook(left_file)
        ws_left=wb_left["Sheet1"]
        
        start_analysis_gaitup_left = ws_left['B14'].value
        stop_analysis_gaitup_left=ws_left['D14'].value
        
        ColD_left=ws_left['D']
        ColD_left=ColD_left[25::]
        
        turns_HS_left=[]
        straightwalk_HS_left=[]
        i=0
        for cl in ColD_left:
            if cl.value!= None:
                i=i+1
                if cl.font.color != None and type(cl.font.color.rgb) == str:
                    x=cl.value+start_analysis_gaitup_left
                    turns_HS_left.append([i,x])
                else:
                    x=cl.value+start_analysis_gaitup_left
                    straightwalk_HS_left.append([i,x])
                
        turns_HS_left=np.vstack(turns_HS_left)
        straightwalk_HS_left=np.vstack(straightwalk_HS_left)
        
        #get index of straight walking steps in the main csv signal
        time=self.CP_data_lf.acc_interp.index.values
        straightwalk_HS_left_index=[]
        for hs in straightwalk_HS_left[:,1]:
            t1=np.where(time>hs)[0][0]
            t2=t1-1
            if np.abs(hs-time[t1])<=np.abs(hs-time[t2]):
                straightwalk_HS_left_index.append(t1)
            else:
                straightwalk_HS_left_index.append(t2)
        
        #get index of turn walking steps in the main csv signal
        Turn_HS_left_index=[]
        for hs in turns_HS_left[:,1]:
            t1=np.where(time>hs)[0][0]
            t2=t1-1
            if np.abs(hs-time[t1])<=np.abs(hs-time[t2]):
                Turn_HS_left_index.append(t1)
            else:
                Turn_HS_left_index.append(t2)
                
        straight_walking_periods_left=self.ranges(straightwalk_HS_left[:,0])
        
        turning_periods_left=self.ranges(turns_HS_left[:,0])
        
        self.HS_r_index=straightwalk_HS_right_index
        
        self.HS_l_index=straightwalk_HS_left_index

    
    def align_plot(self,event=None):
        print("aligning")
        s_h=int((self.ent_str_gaitup_hand.get()))
        s_w=int((self.ent_str_phone_waist.get()))
        s_f=int((self.ent_str_gaitup_feet.get()))
        
        self.CP_data_rf.manual_crop(ind_start=s_f)
        self.CP_data_lf.manual_crop(ind_start=s_f)
        self.CP_data_phone.manual_crop(ind_start=s_w)
        
        if self.hand:
            self.CP_data_h.manual_crop(ind_start=s_h)
        
        
        self.read_gaitup_excel(event=None)
        
        stop_ind_feet=np.maximum(self.HS_l_index[-1],self.HS_r_index[-1])+1
        
        
        
        self.CP_data_lf.manual_crop(ind_start=0,ind_stop=stop_ind_feet)
        self.CP_data_rf.manual_crop(ind_start=0,ind_stop=stop_ind_feet)
        
        self.CP_data_lf.calculate_norm_accandgyro(gyro=self.CP_data_lf.gyro_interp,acc=self.CP_data_lf.acc_interp)
        self.acc_lf_magnitude=self.CP_data_lf.acc_magnitude
        self.CP_data_rf.calculate_norm_accandgyro(gyro=self.CP_data_rf.gyro_interp,acc=self.CP_data_rf.acc_interp)
        self.acc_rf_magnitude=self.CP_data_rf.acc_magnitude
        
        # stop_ind_hand=s_h+s_f-stop_ind_feet
        stop_ind_hand=stop_ind_feet
        
        if self.hand:
            self.CP_data_h.manual_crop(ind_start=0,ind_stop=stop_ind_hand)
            self.CP_data_h.filter_data(acc=self.CP_data_h.acc_interp,gyro=self.CP_data_h.gyro_interp,N=10,fc=2,fs=100)
            self.CP_data_h.calculate_norm_accandgyro(gyro=self.CP_data_h.gyro_filtered,acc=self.CP_data_h.acc_filtered)
            self.acc_h_magnitude=self.CP_data_h.acc_magnitude
        
        # stop_ind_waist=s_w+s_f-stop_ind_feet
        stop_ind_waist=stop_ind_feet
        self.CP_data_phone.manual_crop(ind_start=0,ind_stop=stop_ind_waist)
        self.CP_data_phone.filter_data(acc=self.CP_data_phone.acc_interp,gyro=self.CP_data_phone.gyro_interp,N=10,fc=3,fs=100)
        self.CP_data_phone.calculate_norm_accandgyro(gyro=self.CP_data_phone.gyro_filtered,acc=self.CP_data_phone.acc_filtered)
        self.acc_w_magnitude=self.CP_data_phone.acc_magnitude

        self.figure4 = Figure(figsize=(4, 3), dpi=100)
        self.canvas4 = FigureCanvasTkAgg(self.figure4, master=self.fcontainer_allsignals)
        self.canvas4.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas4, self.fcontainer_allsignals)
        self.toolbar.update()
        self.canvas4._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        color = 'tab:blue'
        b = self.figure4.add_subplot(111)
        b.plot(self.acc_w_magnitude,zorder=1)
        if self.hand:
            b.plot(self.acc_h_magnitude,zorder=1)
        b.plot(self.acc_lf_magnitude,zorder=1)
        b.plot(self.acc_rf_magnitude,zorder=1)
        
        b.set_ylabel('Acc m/s\u00b2', color=color)
        b.title.set_text("allsignals")
        b.set_xticks([])
        
        self.canvas4.draw()
        self.calculate_gaitup_foot()
        
    def calculate_gaitup_foot(self):
        print("calculating gaitup feet")
        fs=100
        stride_time_r=np.diff(self.HS_r_index)/fs
        stride_time_l=np.diff(self.HS_l_index)/fs
        
        stride_time_r=np.array([i for i in stride_time_r if i > 0.8 and i < 1.5])
        stride_time_l=np.array([i for i in stride_time_l if i > 0.8 and i < 1.5])
        N=3
        mean=np.mean(stride_time_r)
        cut_off=N*np.std(stride_time_r)
        lower, upper =  mean- cut_off, mean + cut_off
        stride_time_r = np.array([i for i in stride_time_r if i > lower and i < upper])
        
        N=3
        mean=np.mean(stride_time_l)
        cut_off=N*np.std(stride_time_l)
        lower, upper =  mean- cut_off, mean + cut_off
        stride_time_l = np.array([i for i in stride_time_l if i > lower and i < upper])
        
        numberofstrides=len(stride_time_r)+len(stride_time_l)
        MeanStridetime=np.mean([np.mean(stride_time_r),np.mean(stride_time_l)])
        SDstridetime=np.maximum(np.around(np.std(stride_time_r),decimals=3),np.around(np.std(stride_time_l),decimals=3))
            
        COV=np.around((SDstridetime*100)/MeanStridetime,decimals=3)
        
        print("##feet##")
        print("the sd of stride duration is:")
        print(SDstridetime)
        
        print("the number of strides is:")
        print(numberofstrides)
        
        print("the cov of strides")
        print(COV)
        
            
        
    def plot_steps_onalignedsignals(self,event=None):

        self.figure6 = Figure(figsize=(5, 4), dpi=100)
        self.canvas6 = FigureCanvasTkAgg(self.figure6, master=self.fcontainer_signalwithstep)
        self.canvas6.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas6, self.fcontainer_signalwithstep)
        self.toolbar.update()
        self.canvas6._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        color = 'tab:blue'
       
        b = self.figure6.add_subplot(311)
        
        s_w=int((self.ent_str_phone_waist.get()))
        str_walk=int(float(self.ent_str_walk.get()))*100
        
        d=self.tot_peaks+str_walk-s_w
        print(d)
        
        b.plot(self.acc_w_magnitude,zorder=1)
        
        self.HS_l_index=np.array(self.HS_l_index)
        self.HS_r_index=np.array(self.HS_r_index)
        st_l=self.HS_l_index[np.where(self.HS_l_index<len(self.acc_w_magnitude))[0]]
        st_r=self.HS_r_index[np.where(self.HS_r_index<len(self.acc_w_magnitude))[0]]
        
        b.scatter(st_l,self.acc_w_magnitude[st_l],color='red',marker="o",zorder=2)
        b.scatter(st_r,self.acc_w_magnitude[st_r],color='green',marker="o",zorder=2)

        #remove steps that are greater than the length of the signal
        d_adjusted=[step for step in d if step<len(self.acc_w_magnitude)-1]
        print("the number of steps removed because of gaitup crop reasons")
        print(len(d)-len(d_adjusted))
        
        b.scatter(d_adjusted,self.acc_w_magnitude[d_adjusted],color='black')
        b.set_ylabel('Acc m/s\u00b2', color=color)
        b.title.set_text("Gauche")
        b.set_xticks([])

        if self.hand:
            a = self.figure6.add_subplot(312)
            a.plot(self.acc_h_magnitude,zorder=1)
            a.scatter(self.HS_l_index,self.acc_h_magnitude[self.HS_l_index],color='red',marker="|",s=10000000000,zorder=2)
            a.scatter(self.HS_r_index,self.acc_h_magnitude[self.HS_r_index],color='green',marker="|",s=10000000000,zorder=2)
    
            a.set_ylabel('Acc m/s\u00b2', color=color)
            a.set_xlabel('Time', color=color)
            a.title.set_text("Hand")
            
            self.a=a
        
        c = self.figure6.add_subplot(313)
            
        c.plot(self.acc_lf_magnitude,zorder=1)
        c.plot(self.acc_rf_magnitude,zorder=1)
        
        c.scatter(self.HS_l_index,self.acc_lf_magnitude[self.HS_l_index],color='red',marker="|",s=10000000000,zorder=2)
        c.scatter(self.HS_r_index,self.acc_rf_magnitude[self.HS_r_index],color='green',marker="|",s=10000000000,zorder=2)

        
        c.set_ylabel('Acc m/s\u00b2', color=color)
        c.set_xlabel('Time', color=color)
        c.title.set_text("Right/Left foot")
        
        self.canvas1.draw()
        
        
    def detect_steps_hand(self,event=None):

        if self.hand:
            #must change units
            self.CP_data_h.gtom2s()
            self.CP_data_h.deg2rad()
            
            self.CP_data_h.filter_data(acc=self.CP_data_h.acc_interp,gyro=self.CP_data_h.gyro_interp,N=10,fc=2,fs=100)
            self.CP_data_h.calculate_norm_accandgyro(gyro=self.CP_data_h.gyro_filtered,acc=self.CP_data_h.acc_filtered)
            
            self.acc_h_magnitude=self.CP_data_h.acc_magnitude
            self.gyro_h_magnitude=self.CP_data_h.gyro_magnitude
            
            
            
            
            
            self.CP_data_h.calculate_hand_features()
            accfeat=self.CP_data_h.acc_features
            gyrofeat=self.CP_data_h.gyro_features
            
            #load models and apply smartstep
            self.CP_data_h.detect_steps_SmartStep()
            
            steps_smart=self.CP_data_h.steps_smartstep
            
            self.CP_data_h.peakdet_m2(acc=False,plot_peak=False,detect_turn=False)
            
            valley=self.CP_data_h.peakandvalley["valley_index"]
            
            
            print("before filtration the number of steps is ")
            print("peak detection steps:")
            print(len(valley))
            print("smartStep steps")
            print(len(steps_smart))
            
            self.CP_data_h.computeVarStride(fs=100,remove_outliers=True,
                                            N=3,use_peaks=False,
                                            pocket=False)
            
            
            
            result=self.CP_data_h.cycle_temp
            
            print("the sd of stride duration is:")
            print(result['stridetime_std'])
            
            print("the number of strides is:")
            print(len(result['stridetime']))
            
            print("the cov of strides")
            print(result['stridetime_Cov'])
            
            
            plt.figure()
            plt.plot(self.gyro_h_magnitude)
            plt.scatter(valley,self.gyro_h_magnitude[valley])
            
            
            self.CP_data_h.computeVarStride(fs=100,remove_outliers=True,
                                N=3,use_smartstep=True,
                                pocket=False)
            
            
            result=self.CP_data_h.cycle_temp
            
            print("the sd of stride duration is:")
            print(result['stridetime_std'])
            
            print("the number of strides is:")
            print(len(result['stridetime']))
            
            print("the cov of strides")
            print(result['stridetime_Cov'])
            
            plt.figure()
            plt.plot(self.acc_h_magnitude)
            plt.scatter(steps_smart,self.acc_h_magnitude[steps_smart])
            plt.figure()
            plt.plot(self.gyro_h_magnitude)
            plt.scatter(steps_smart,self.gyro_h_magnitude[steps_smart])
            
  
        
    def plot_graph(self,event=None,plot_name=""):
        try: 
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.destroy()
        except AttributeError: 
            pass     

        # Setup matplotlib canvas
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.fcontainer)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        #set method on click
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self.on_click_canvas)
        
        
        # Setup matplotlib toolbar (optional)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.fcontainer)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        
        options = ['Select an option', 'Acceleration Norm raw', 'Gyroscope norm raw', 'Acceleration Norm filtered', 'Gyroscope Norm filtered', 
                   'Gyroscope signal steps','Acceleration signal steps','Mean acceleration gait cycle', 'Mean gyroscope gait cycle',  ]
        if plot_name=="":
            plot_name=self.plot_combo.get()
        else:
            plot_name='Acceleration Norm'
            
        if (plot_name=='Acceleration Norm'):
            self.a = self.figure.add_subplot(111)
            
            self.a.plot(self.CP_data.acc_magnitude,zorder=1)
            
            self.canvas.draw()
            
        if (plot_name=='Gyroscope norm'):
            self.a = self.figure.add_subplot(111)
            self.a.plot(self.CP_data.gyro_magnitude,zorder=1)
            self.canvas.draw()
            
        if (plot_name=='Gyroscope signal'):
            self.canvas.delete("all")
            
    def calculate_norm_accandgyro(self,gyro=[0],acc=[0]):
        
        matrix1=acc
        x=matrix1.iloc[:,0].values**2
        y=matrix1.iloc[:,1].values**2
        z=matrix1.iloc[:,2].values**2
        m=x+y+z
        mm=np.array([np.sqrt(i) for i in m])
        acc_magnitude=mm
        
        gyro_magnitude=0
        if len(gyro)==0:
            matrix2=gyro
            x=matrix2.iloc[:,0].values**2
            y=matrix2.iloc[:,1].values**2
            z=matrix2.iloc[:,2].values**2
            m=x+y+z
            mm=np.array([np.sqrt(i) for i in m])
            gyro_magnitude=mm
        
        return(acc_magnitude,gyro_magnitude)
    def detectstartofwalk(self,sig1,thresh=3): 
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
                    i=i++N_wf//4
            i=i+N_wf//4
            
        return(startp)


    def Calculate_lyapunov(self, event=None,tao=0,dim=0,three_dim=True):
        print("calculating lyapunov")
        if self.preprocessed==False or self.lyap_NBstrides != int((self.ent_Nbstridelyap.get())):
            self.lyap_NBstrides=int((self.ent_Nbstridelyap.get()))
            self.preprocessing_lyap(totalnumberofstride=self.lyap_NBstrides)

        tao=int((self.ent_embdim.get()))
        dim=int(self.ent_timedelay.get())
        
        print("calculating lyap on magnitude")
        _,self.debug=nld.lyap_r(self.acc_data, emb_dim=dim,lag=tao,
                                min_tsep=40, tau=0.01, trajectory_len=100*10, fit='poly', debug_plot=True,debug_data=True, plot_file="lyap", fit_offset=0)
    
        stride_duration=100
        plt.figure()
        d=self.debug[1]
        x=self.debug[0]/stride_duration
        plt.plot(x,d)
        
        j=stride_duration//2
        #plt.plot(debug[0],debug[1])
        z=np.polyfit(self.debug[0][0:j],self.debug[1][0:j],1)
        p = np.poly1d(z)
        plt.plot(self.debug[0][0:j]/stride_duration,p(self.debug[0][0:j]))
        lle=z[0]*100
        print("the lyap of acceleration norm is %.3f" %lle)
        self.lbl_lyapr.configure(text ='%.2f '%(lle))
        
        print("calculating lyap on three dimensions")
        _,self.debug=nld.lyap_r(self.threeD_acc_data[self.threeD_acc_data.columns[0]], 
                                emb_dim=dim,lag=tao, min_tsep=40, tau=0.01, trajectory_len=100*10, fit='poly', debug_plot=True,debug_data=True, plot_file="lyap", fit_offset=0)
        stride_duration=100
        plt.figure()
        d=self.debug[1]
        x=self.debug[0]/stride_duration
        plt.plot(x,d)
        
        j=stride_duration//2
        # plt.plot(debug[0],debug[1])
        z=np.polyfit(self.debug[0][0:j],self.debug[1][0:j],1)
        p = np.poly1d(z)
        plt.plot(self.debug[0][0:j]/stride_duration,p(self.debug[0][0:j]))
        lle=z[0]*100
        print("the lyap of X direction is %.3f" %lle)
        self.lbl_lyapx.configure(text ='%.2f '%(lle))
        
        _,self.debug=nld.lyap_r(self.threeD_acc_data[self.threeD_acc_data.columns[1]],
                                emb_dim=dim,lag=tao, min_tsep=40, tau=0.01, trajectory_len=100*10, fit='poly', debug_plot=True,debug_data=True, plot_file="lyap", fit_offset=0)
        stride_duration=100
        plt.figure()
        d=self.debug[1]
        x=self.debug[0]/stride_duration
        plt.plot(x,d)
        
        j=stride_duration//2
        #plt.plot(debug[0],debug[1])
        z=np.polyfit(self.debug[0][0:j],self.debug[1][0:j],1)
        p = np.poly1d(z)
        plt.plot(self.debug[0][0:j]/stride_duration,p(self.debug[0][0:j]))
        lle=z[0]*100
        print("the lyap of Y direction is %.3f" %lle)
        self.lbl_lyapy.configure(text ='%.2f '%(lle))
        
        _,self.debug=nld.lyap_r(self.threeD_acc_data[self.threeD_acc_data.columns[2]].values,
                                emb_dim=dim,lag=tao, min_tsep=40, tau=0.01, trajectory_len=100*10, fit='poly', debug_plot=True,debug_data=True, plot_file="lyap", fit_offset=0)
        stride_duration=100
        plt.figure()
        d=self.debug[1]
        x=self.debug[0]/stride_duration
        plt.plot(x,d)
        
        j=stride_duration//2
        #plt.plot(debug[0],debug[1])
        z=np.polyfit(self.debug[0][0:j],self.debug[1][0:j],1)
        p = np.poly1d(z)
        plt.plot(self.debug[0][0:j]/stride_duration,p(self.debug[0][0:j]))
        lle=z[0]*100
        print("the lyap of Z direction is %.3f" %lle)
        self.lbl_lyapz.configure(text ='%.2f '%(lle))

    def Calculate_embdimension(self,event=None,tao=10):
        print("calculating embedding dimension")
        if self.preprocessed==False or self.lyap_NBstrides != int((self.ent_Nbstridelyap.get())):
            self.lyap_NBstrides=int((self.ent_Nbstridelyap.get()))
            self.preprocessing_lyap(totalnumberofstride=self.lyap_NBstrides)
            
        self.f=dimension.fnn(self.acc_data, dim=[1,2,3,4,5,6,7,8,9,10], tau=tao, R=15.0, A=2.0, metric='euclidean', window=50,maxnum=None, parallel=True)
        plt.figure()
        plt.plot(self.f[0])
        plt.title("dim of magnitude")
        
        self.f=dimension.fnn(self.threeD_acc_data[self.threeD_acc_data.columns[0]], dim=[1,2,3,4,5,6,7,8,9,10], tau=tao, R=15.0, A=2.0, metric='euclidean', window=50,maxnum=None, parallel=True)
        plt.figure()
        plt.plot(self.f[0])
        plt.title("dim of accx")
        
        self.f=dimension.fnn(self.threeD_acc_data[self.threeD_acc_data.columns[1]], dim=[1,2,3,4,5,6,7,8,9,10], tau=tao, R=15.0, A=2.0, metric='euclidean', window=50,maxnum=None, parallel=True)
        plt.figure()
        plt.plot(self.f[0])
        plt.title("dim of accy")

        self.f=dimension.fnn(self.threeD_acc_data[self.threeD_acc_data.columns[2]], dim=[1,2,3,4,5,6,7,8,9,10], tau=tao, R=15.0, A=2.0, metric='euclidean', window=50,maxnum=None, parallel=True)
        plt.figure()
        plt.plot(self.f[0])
        plt.title("dim of accz")

    def preprocessing_lyap(self,totalnumberofstride=150):
        self.preprocessed=True
        #the signal when turns are removed
        accc=self.CP_data.walkingperiodsacc
        accc = pd.concat(accc,ignore_index=True)
        gyroo=self.CP_data.walkingperiodsgyro
        gyroo = pd.concat(gyroo,ignore_index=True)
        
        #calculating the norm of the signal when norms are removed and 
        #then normalising to a fixed number of strides and number of points
        self.CP_data.calculate_norm_accandgyro(gyro=gyroo,acc=accc)
        self.CP_data.normal_signal(threeD_acc=accc,threeD_gyro=gyroo,peaks=self.tot_peaks,remove_outliers=True,N=3,Numberofstrides=totalnumberofstride,plot=False)
        
        # retrieving the normalised signals
        self.acc_data=self.CP_data.norma_acc_strides
        self.gyro_data=self.CP_data.norma_acc_strides
        
        self.threeD_acc_data=self.CP_data.norma_threeD_acc
        self.threeD_gyro_data=self.CP_data.norma_threeD_gyro
        
    def Calculate_time_delay(self, event=None):
        
        print("calculating time delay")
        if self.preprocessed==False or self.lyap_NBstrides != int((self.ent_Nbstridelyap.get())):
            self.lyap_NBstrides=int((self.ent_Nbstridelyap.get()))
            self.preprocessing_lyap(totalnumberofstride=self.lyap_NBstrides)
            
        lag=delay.dmi(self.acc_data, maxtau=50, bins=64)
        plt.figure()
        plt.title("lag of magnitude")
        plt.plot(lag)
        
        lag=delay.dmi(self.threeD_acc_data[self.threeD_acc_data.columns[0]], maxtau=50, bins=64)
        plt.figure()
        plt.title("lag of x")
        plt.plot(lag)
        
        lag=delay.dmi(self.threeD_acc_data[self.threeD_acc_data.columns[1]], maxtau=50, bins=64)
        plt.figure()
        plt.title("lag of y")
        plt.plot(lag)
        
        lag=delay.dmi(self.threeD_acc_data[self.threeD_acc_data.columns[2]], maxtau=50, bins=64)
        plt.figure()
        plt.title("lag of z")
        plt.plot(lag)
        
    def on_accunit_selected(self,event=None):
        messagebox.showinfo('You choosed:', self.accunit_combo.get())
        
    def on_gyrounit_selected(self,event=None):
        messagebox.showinfo('You choosed:', self.gyrounit_combo.get())
        
    def on_plot_selected(self,event=None):
        messagebox.showinfo('You choosed:', self.plot_combo.get())
                
    def clear_graph(self,event=None):

        self.canvas.get_tk_widget().pack_forget() 
        self.toolbar.destroy()
        
        
    def quit(self, event=None):
        self.mainwindow.quit()
        
        

    def run(self):
        self.mainwindow.mainloop()
        
        
        
if __name__ == '__main__':
    app = MyApplication()
    app.run()