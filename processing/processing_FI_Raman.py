#!/usr/bin/env python
# coding: utf-8

# **Raman data processing - Fluid inclusion vapor composition**
#### This script reads the Raman spectra of fluid inclusions to measure the area under peaks and calculate the mol% of gas species present (for now only CO2, N2 and CH4).

#*To ensure a smooth process, have all your files in the same folder. <br /> 
#If you have vapor spectra files mixed with other files, make sure that all the vapor data files are in txt format and have the notation **"-v"** in the file name.*

#Script written by [Fernando Araujo](https://github.com/thembubbles).



# ---------------- script for processing fluid inclusion Raman data (Raman spectroscopy)

# ------- Calculate area under peak from Raman files and process to gas mol%

# --- import modules

import os
import glob
import pandas as pd
import numpy as np

from tkinter import Tk, filedialog
import warnings

import rampy as rp
from sklearn.metrics import auc


def peak_area(spectrum, file=None):

    #get the raw values from the spectrum file
    x = spectrum['x'].to_numpy()
    

    #separate the spectrum in dataframes for each gas peak
    CO2_v1_peak = spectrum.query('1240 <= x <= 1300').copy()
    CO2_v2_peak = spectrum.query('1360 <= x <= 1420').copy()
    N2_peak = spectrum.query('2300 <= x <= 2345').copy()
    CH4_peak = spectrum.query('2890 <= x <= 2930').copy()


    #set the regions of interest and attaching points for the local baseline corrections
    bir_CO2_v1 = np.array([(1240,1260),(1290,1310)])
    bir_CO2_v2 = np.array([(1360,1380),(1410,1430)])
    bir_N2 = np.array([(2290,2310),(2330,2350)])
    bir_CH4 = np.array([(2880,2900),(2920,2940)])

    #transform the dataframe columns into numpy array to allow baseline correction
    CO2_v1_x = CO2_v1_peak['x'].to_numpy()
    CO2_v1_y = CO2_v1_peak['y'].to_numpy()
    CO2_v2_x = CO2_v2_peak['x'].to_numpy()
    CO2_v2_y = CO2_v2_peak['y'].to_numpy()
    N2_x = N2_peak['x'].to_numpy()
    N2_y = N2_peak['y'].to_numpy()
    CH4_x = CH4_peak['x'].to_numpy()
    CH4_y = CH4_peak['y'].to_numpy()


    #here the actual baseline removal takes place using the Rampy module
    #the IF conditional works to identify if the spectrum contains the specific wavenumbers of each gas
    if CO2_v1_x.size > 0:
        y_CO2_v1_PLS, back_CO2_v1 = rp.baseline(CO2_v1_x,CO2_v1_y,bir_CO2_v1,'drPLS',lam=10**5)
    else:
        CO2_v1_peak['x'], y_CO2_v1_PLS, back_CO2_v1 = [0] * len(x), [0] * len(x), [0] * len(x)
        print("the CO2 peak range was not measured for sample %s" %(file))

    if CO2_v2_x.size > 0:
        y_CO2_v2_PLS, back_CO2_v2 = rp.baseline(CO2_v2_x,CO2_v2_y,bir_CO2_v2,'drPLS',lam=10**5)
    else:
        CO2_v2_peak['x'], y_CO2_v2_PLS, back_CO2_v2 = [0] * len(x), [0] * len(x), [0] * len(x)
        print("the CO2 peak range was not measured for sample %s" %(file))

    if N2_x.size > 0:
        y_N2_PLS, back_N2 = rp.baseline(N2_x,N2_y,bir_N2,'drPLS',lam=10**5)
    else:
        N2_peak['x'], y_N2_PLS, back_N2 = [0] * len(x), [0] * len(x), [0] * len(x)
        print("the N2 peak range was not measured for sample %s" %(file))

    if CH4_x.size > 0:
        y_CH4_PLS, back_CH4 = rp.baseline(CH4_x,CH4_y,bir_CH4,'drPLS',lam=10**5)
    else:
        CH4_peak['x'], y_CH4_PLS, back_CH4 = [0] * len(x), [0] * len(x), [0] * len(x)
        print("the CH4 peak range was not measured for sample %s" %(file))

    
    #add the baseline corrected values to the peak dataframes
    CO2_v1_peak['y_CO2_v1_PLS'] = y_CO2_v1_PLS
    CO2_v2_peak['y_CO2_v2_PLS'] = y_CO2_v2_PLS
    N2_peak['y_N2_PLS'] = y_N2_PLS
    CH4_peak['y_CH4_PLS'] = y_CH4_PLS                    
    CO2_v1_peak['back_CO2_v1'] = back_CO2_v1
    CO2_v2_peak['back_CO2_v2'] = back_CO2_v2
    N2_peak['back_N2'] = back_N2
    CH4_peak['back_CH4'] = back_CH4        

    #calculate the area under curve using sklearn module for each peak after baseline correction 
    auc_CO2_v1 = auc(CO2_v1_peak['x'],CO2_v1_peak['y_CO2_v1_PLS'])
    auc_CO2_v2 = auc(CO2_v2_peak['x'],CO2_v2_peak['y_CO2_v2_PLS'])
    auc_N2 = auc(N2_peak['x'],N2_peak['y_N2_PLS'])
    auc_CH4 = auc(CH4_peak['x'],CH4_peak['y_CH4_PLS'])


    dict = {'area_CO2_v1': auc_CO2_v1, 'area_CO2_v2': auc_CO2_v2, 'area_N2': auc_N2, 'area_CH4': auc_CH4} 

    df_peak_areas = pd.Series(dict).fillna(0)

    df_peak_areas[df_peak_areas < 0] = 0
    
    

    return df_peak_areas



# --- set working directories

root = Tk() # pointing root to Tk() to use it as Tk() in program.
root.withdraw() # Hides small tkinter window.
root.attributes('-topmost', True) # Opened windows will be active. above all windows despite of selection.

base_dir = filedialog.askdirectory()+'/'


# --- look into working directory and create a list with selected files - .txt
os.chdir(base_dir)

FI_files = glob.glob('*-v*.txt')


df_Raman = pd.DataFrame(columns=['area_CO2_v1','area_CO2_v2','area_N2','area_CH4'])


np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')


for file in FI_files:

    current_file = pd.read_csv(base_dir+file,
                    encoding = "ANSI", sep = '\t', names=('x','y'), comment="#")

    # --- add file information
    current_inclusion = file.replace(".txt", "")
    
    
    data = peak_area(current_file,current_inclusion)
    a = pd.DataFrame(data).T
    
    a.insert(0, 'Sample', current_inclusion)
    
    
    df_Raman = pd.concat([df_Raman,a], axis=0, ignore_index=True)
    
df_Raman = df_Raman.fillna(0)



Raman_cross_section = {'SO2' : 5.3,
                       '12CO2_v1' : 1,
                       '12CO2_2v2' : 1.5,
                       '13CO2_2v2' :1.5,
                       'O2' : 1.2,
                       'CO' : 0.9,
                       'N2' : 1,
                       'H2S' : 6.4,
                       'CH4' : 7.5,
                       'NH3' : 5,
                       'H2' : 2.3}


df_V_mol = df_Raman[['Sample']].copy()


df_V_mol['CO2_v1_mol'] = df_Raman['area_CO2_v1'] / Raman_cross_section['12CO2_v1']
df_V_mol['CO2_v2_mol'] = df_Raman['area_CO2_v2'] / Raman_cross_section['12CO2_2v2']
df_V_mol['N2_mol'] = df_Raman['area_N2'] / Raman_cross_section['N2']
df_V_mol['CH4_mol'] = df_Raman['area_CH4'] / Raman_cross_section['CH4']


df_V_mol['mol_sum'] = df_V_mol.sum(axis=1)

df_Raman['X_CO2(mol%)'] = round(((df_V_mol['CO2_v1_mol']/df_V_mol['mol_sum']) + (df_V_mol['CO2_v2_mol']/df_V_mol['mol_sum'])),3)
df_Raman['X_N2(mol%)'] = round((df_V_mol['N2_mol']/df_V_mol['mol_sum']),3)
df_Raman['X_CH4(mol%)'] = round((df_V_mol['CH4_mol']/df_V_mol['mol_sum']),3)




# df_Raman[['sample','piece','field','analysis','rest']
#         ] = df_Raman["Sample"].str.split(pat='-', 
#                                           n=4,  
#                                           expand=True)
                                         

df_Raman = df_Raman[['Sample',
                     'X_CO2(mol%)', 'X_N2(mol%)', 'X_CH4(mol%)',
                     'area_CO2_v1', 'area_CO2_v2', 'area_N2', 'area_CH4']]



df_Raman.to_csv(base_dir+'FI_Raman_V_composition.csv',index=False)
df_Raman
