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

import os
from os import walk

import tkinter as tk
import pygubu
from tkinter import messagebox
from CPclass import phone as CP

import numpy as np

from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)# Implement the default Matplotlib key bindings.
from matplotlib.figure import Figure

import pandas as pd

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
pickable_artists = []


class MyApplication:
    
    def __init__(self):
        
        #variables
        self.walking_period=[]
        
        
        #1: Create a builder
        self.builder = builder = pygubu.Builder()

        #2: Load an ui file
        builder.add_from_file(os.path.join(CURRENT_DIR, 'notebook2.ui'))
        
        #3: Create the toplevel widget.
        self.mainwindow = builder.get_object('mainwindow')

        # Container for the matplotlib canvas and toolbar classes
        self.fcontainer = builder.get_object('fcontainer')
         
        #set filepathchooser
        self.filepath = builder.get_object('filepath')
        self.filepath_export=builder.get_object('path_choose_Export')
        
        #set combo_box
        self.accunit_combo = builder.get_object('combobox_1')
        options = ['Select an option', 'm/s^2', 'g']
        self.accunit_combo.config(values=options)
        
        
        self.gyrounit_combo = builder.get_object('combobox_2')
        options = ['Select an option', 'rad/s', 'deg/s']
        self.gyrounit_combo.config(values=options)
        
        self.plot_combo = builder.get_object('combobox_4')
        options = ['Select an option', 'Acceleration Norm', 'Gyroscope norm' ]
        self.plot_combo.config(values=options)
        
        #get labels
        self.lbl_files=builder.get_object('lbl_files')
        self.lbl_SDstride=builder.get_object('SD_stride')
        self.lbl_CVstride=builder.get_object('CV_stride')
        self.lbl_Nstride=builder.get_object('N_stride')
        self.lbl_Nphase=builder.get_object('N_phase')
        
        
        #get entries
        self.ent_str_walk = self.builder.get_object('str_walk')
        self.ent_stp_walk = self.builder.get_object('stp_walk')
        self.ent_turn = self.builder.get_object('step_turn')
        
        
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

        # Connect button callback
        builder.connect_callbacks(self)
        
        


        
    def on_click_canvas(self,event):
#        print('-----')
#        print('button:', event.button)
#        print('xdata, ydata:', event.xdata, event.ydata)
#        print('x, y:', event.x, event.y)
#        print('canvas:', event.canvas)
        print('inaxes:', event.inaxes)
        
        if self.toolbar.mode =="":
            print("toolbar is not selected")
            
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
                    self.peaks2=np.append(self.peaks2,x)
                    self.refresh_treeview()
                    
                    
                else:
                    print(self.walking_period)
                    self.walking_period=[i for i in self.walking_period if i>x+400 or i<x-400]
                    self.peaks2=[i for i in self.peaks2 if i>x+400 or i<x-400]
                    self.refresh_treeview()
                    print("walking_period")
                    print(self.walking_period)
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
        self.btn_cal_norm["state"] = "disabled"
        self.btn_plot["state"] = "disabled"
        self.btn_detect_start_stop["state"] = "disabled"
        self.btn_detect_step["state"] = "disabled"
        self.btn_detect_turn["state"] = "disabled"
        
        self.path = self.filepath.cget('path')
        self.export_path=self.path
#        self.filenames=[]
#        for root, dirs, files in os.walk(self.path):
#            for file in files:
#                if file.endswith(".txt"):
#                    self.filenames.append(os.path.join(root, file))
#        print(self.filenames)
        # show paths in label
        _, _, self.files = next(walk(self.path))

        self.lbl_files.configure(text = 'The files are %s ,%s and %s'%(self.files[0],self.files[1],self.files[2]))
        
    def on_path_changed_export(self, event=None):
        
        self.export_path= self.filepath_export.cget('path')
        print("export path is changed")

        
    def on_accunit_selected(self,event=None):
        messagebox.showinfo('You choosed:', self.accunit_combo.get())
        
    def on_gyrounit_selected(self,event=None):
        messagebox.showinfo('You choosed:', self.gyrounit_combo.get())
        
    def on_plot_selected(self,event=None):
        messagebox.showinfo('You choosed:', self.plot_combo.get())
        
        
    def Interpolate_and_filter(self,event=None):
        #acceleration signal
        self.CP_data=CP(self.path,app="geoloc")
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
        
    def Detect_turns(self,event=None):
        
        str_walk=int(float(self.ent_str_walk.get()))*100
        stp_walk=int(self.ent_stp_walk.get())*100
        last=0
        self.CP_data.peakdet_m2(acc=False,plot_peak=True,detect_turn=True)
        
        self.peaks=self.CP_data.peakandvalley["peak_index"]
        self.peaks2=[]
        
        self.phase=[]
        turn=int(self.ent_turn.get())
        self.phase.append((0,str_walk,self.peaks[0],turn))
        self.peaks2.append(str_walk)
        self.peaks2.append(self.peaks[0])
        
        

        for i in range(0,len(self.peaks)-2):
            if (self.peaks[i+1]>stp_walk):
                last=i
                break
            self.phase.append((i+1,self.peaks[i],self.peaks[i+1],turn))
            self.peaks2.append(self.peaks[i+1])
            
        if last!=0:   
            self.phase.append((last+1,self.peaks[last],stp_walk,turn))
            self.peaks2.append(self.peaks[last])
            self.peaks2.append(stp_walk)
            
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
            
        
    def Save_and_Export(self,event=None):
        
        self.exported_results.to_csv(self.export_path+"_exported_results.csv",index=False,header=True)
        self.exported_results_res.to_csv(self.export_path+"_exported_stride_results.csv",index=False,header=True)
        print("saved")
        
        
    def Detect_steps(self,event=None):
        phases_adj=[]
        # Numberofsteps=[]
        turn_strides=[]
        phases=[]
        
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
            print(phases_adj)
            
        self.CP_data.crop_medipole(fs=1,phases=phases_adj)
        
        accc=self.CP_data.walkingperiodsacc
        gyroo=self.CP_data.walkingperiodsgyro
        
        for i in range(0,len(accc)):
            self.CP_data.filter_data(acc=accc[i],gyro=gyroo[i],N=10,fc=3,fs=100)
            
            self.CP_data.calculate_norm_accandgyro(gyro=self.CP_data.gyro_filtered,acc=self.CP_data.acc_filtered)
    
            self.CP_data.peakdet_m2(acc=True,plot_peak=False,detect_turn=False)
            
            #remove 2 steps from beginging and end 
            peaks=self.CP_data.peakandvalley["peak_index"]
            
            self.CP_data.computeVarStride(fs=100,remove_outliers=True,N=3,use_peaks=True,pocket=False,remove_step=turn_strides[i])
            
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
            
        print(self.treeview_stat.get_children())
        exported_list=[]
        
        for line in self.treeview_stat.get_children():
            exported_list.append(self.treeview_stat.item(line)['values'])
            
        self.exported_results=pd.DataFrame(exported_list, columns=["Phase", "Mean stride duration", "standard deviation of stride duration","Coefficient of variance of stride duration", "Number of strides"])
        
        print(self.exported_results)
        
        exported_list=[]
        
        for line in self.treeview_res.get_children():
            exported_list.append(self.treeview_res.item(line)['values'])
            
        print(exported_list)
            
        self.exported_results_res=pd.DataFrame(exported_list, columns=["Phase","stride number", "stride duration"])
        
        print(self.exported_results)
        
        self.exported_results_res['stride duration']=pd.to_numeric(self.exported_results_res['stride duration'], downcast='float')
        
        self.lbl_SDstride.configure(text ='%.2f milliseconds'%(np.round(np.std(self.exported_results_res['stride duration'].values),decimals=5)*1000))
        self.lbl_CVstride.configure(text ='%.2f %%'%(np.round(np.std(self.exported_results_res['stride duration'].values)/np.mean(self.exported_results_res['stride duration'].values),decimals=5)*100))
        self.lbl_Nstride.configure(text ='%d'%(len(self.exported_results_res['stride duration'].values)))
        self.lbl_Nphase.configure(text ='%d'%(len(self.exported_results)))
        
        
            
            
        # print(exported_results)
            
        
               
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
        
        
        
        
    def plot_graph(self,event=None,plot_name=""):
        try: 
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.destroy()
        except AttributeError: 
            pass     

        # Setup matplotlib canvas
        self.figure = Figure(figsize=(7, 6), dpi=100)
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