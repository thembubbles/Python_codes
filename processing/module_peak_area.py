# Author: Fernando Prado Araujo
# with contributions from Thomas Van Gerve

import numpy as np
import pandas as pd

from sklearn.metrics import auc
from csaps import csaps


# -------- Calculate Raman peak areas

def peak_area(spectrum, file=None):

    #get the raw values from the spectrum file 
    
    x = spectrum['x'].to_numpy()
    y = spectrum['y'].to_numpy()

    #separate the spectrum in multiple dataframes, one for each gas peak
    CO2_v1_peak = spectrum.query('1240 <= x <= 1300').copy()
    CO2_v2_peak = spectrum.query('1360 <= x <= 1420').copy()
    N2_peak = spectrum.query('2300 <= x <= 2345').copy()
    CH4_peak = spectrum.query('2890 <= x <= 2930').copy()


    #set the baseline interpolation regions and attaching points for the local baseline corrections
    bir1_CO2_v1 = [1240,1260]
    bir2_CO2_v1 = [1290,1310]
    bir1_CO2_v2 = [1360,1380]
    bir2_CO2_v2 = [1410,1430]
    bir1_N2 = [2290,2310]
    bir2_N2 = [2330,2350]
    bir1_CH4 = [2880,2900]
    bir2_CH4 = [2920,2940]
    
    #create an array with all the interpolation regions and reshape the area to have fix size
    birs = np.array(bir1_CO2_v1 + bir2_CO2_v1 + bir1_CO2_v2 + bir2_CO2_v2 + bir1_N2 + bir2_N2 + bir1_CH4 + bir2_CH4)
    birs = birs.reshape(len(birs) // 2, 2)


    #get the actual points in the spectrum by using the defined BIRS
    spec_stack = np.column_stack((x, y))
    for i, j in enumerate(birs):
        if i == 0:
            spectrumBir = spec_stack[(spec_stack[:, 0] > j[0]) & (spec_stack[:, 0] < j[1]), :]
        else:
            birRegion = spec_stack[(spec_stack[:, 0] > j[0]) & (spec_stack[:, 0] < j[1]), :]
            # xfit= np.vstack((xfit,xtemp))
            spectrumBir = np.row_stack((spectrumBir, birRegion))
    
    xbir, ybir = spectrumBir[:, 0], spectrumBir[:, 1]


    #transform the dataframe columns into numpy arrays to allow baseline correction
    CO2_v1_x = CO2_v1_peak['x'].to_numpy()
    CO2_v1_y = CO2_v1_peak['y'].to_numpy()
    CO2_v2_x = CO2_v2_peak['x'].to_numpy()
    CO2_v2_y = CO2_v2_peak['y'].to_numpy()
    N2_x = N2_peak['x'].to_numpy()
    N2_y = N2_peak['y'].to_numpy()
    CH4_x = CH4_peak['x'].to_numpy()
    CH4_y = CH4_peak['y'].to_numpy()


    #here the actual baseline removal takes place using the csaps package
    #the IF conditional works to identify if the spectrum contains the specific wavenumbers of each gas
    
    spline = csaps(xbir, ybir, smooth=1e-4) # this line creates the spline function to fit the spectra using the previously defined BIRs
    
    if CO2_v1_x.size > 0:
        y_CO2_v1_PLS = spline(CO2_v1_x)
    else:
        CO2_v1_peak['x'], y_CO2_v1_PLS, CO2_v1_y = np.zeros(np.shape(x)), np.zeros(np.shape(x)), np.zeros(np.shape(x))
        print("the CO2 peak range was not measured for sample %s" %(file))

    if CO2_v2_x.size > 0:
        y_CO2_v2_PLS = spline(CO2_v2_x)  
    else:
        CO2_v2_peak['x'], y_CO2_v2_PLS, CO2_v2_y = np.zeros(np.shape(x)), np.zeros(np.shape(x)), np.zeros(np.shape(x))
        print("the CO2 peak range was not measured for sample %s" %(file))

    if N2_x.size > 0:
        y_N2_PLS = spline(N2_x) 
    else:
        N2_peak['x'], y_N2_PLS, N2_y = np.zeros(np.shape(x)), np.zeros(np.shape(x)), np.zeros(np.shape(x))
        print("the N2 peak range was not measured for sample %s" %(file))

    if CH4_x.size > 0:
        y_CH4_PLS = spline(CH4_x)
    else:
        CH4_peak['x'], y_CH4_PLS, CH4_y = np.zeros(np.shape(x)), np.zeros(np.shape(x)), np.zeros(np.shape(x))
        print("the CH4 peak range was not measured for sample %s" %(file))

    
    #add the baseline corrected values to the peak dataframes
    CO2_v1_peak['y_CO2_v1_BL'] = CO2_v1_y - y_CO2_v1_PLS
    CO2_v2_peak['y_CO2_v2_BL'] = CO2_v2_y - y_CO2_v2_PLS
    N2_peak['y_N2_BL'] = N2_y - y_N2_PLS
    CH4_peak['y_CH4_BL'] = CH4_y - y_CH4_PLS         

    #calculate the area under curve using sklearn module for each peak after baseline correction 
    auc_CO2_v1 = auc(CO2_v1_peak['x'],CO2_v1_peak['y_CO2_v1_BL'])
    auc_CO2_v2 = auc(CO2_v2_peak['x'],CO2_v2_peak['y_CO2_v2_BL'])
    auc_N2 = auc(N2_peak['x'],N2_peak['y_N2_BL'])
    auc_CH4 = auc(CH4_peak['x'],CH4_peak['y_CH4_BL'])

    dict = {'area_CO2_v1': auc_CO2_v1, 'area_CO2_v2': auc_CO2_v2, 'area_N2': auc_N2, 'area_CH4': auc_CH4} 

    df_peak_areas = pd.Series(dict).fillna(0)

    df_peak_areas[df_peak_areas < 0] = 0
   
    
    return df_peak_areas




# -------- Calculate Raman cross-section

def cross_section(gas, laser=532):
        
        # Reference values extracted from Burke (2001) Lithos, 
        #dv is the gas Raman shift, 
        #suma is relative normalized differential Raman scattering cross-sections
    Raman_sigma = {'SO2' : {'dv' : 1151, "suma" : 4.03},
                   'CO2_v1' : {'dv' : 1285, "suma" : 0.80},
                   'CO2_2v2' : {'dv' : 1388, "suma" : 1.23},
                   'O2' : {'dv' : 1555, "suma" : 1.03},
                   'CO' : {'dv' : 2143, "suma" : 0.90},
                   'N2' : {'dv' : 2331, "suma" : 1},
                   'H2S' : {'dv' : 2611, "suma" : 6.8},
                   'CH4' : {'dv' : 2917, "suma" : 8.63},
                   'NH3' : {'dv' : 3336, "suma" : 6.32},
                   'H2O' : {'dv' : 3657, "suma" : 3.29},
                   'H2' : {'dv' : 4156, "suma" : 3.54}
                  }

    v0 = 10**7 / laser #convert laser wavelength from nm to cm-1

    vi = Raman_sigma[gas]['dv']  #gas species Raman shift

    h = 6.626*(10**(-27)) #Planck's constant [erg s]

    c = 2.998*(10**(10)) #light velocity [cm/s]

    k = 1.381*(10**(-16)) #Boltzmann’s constant [erg/K]

    T = 273.15 + 25 #absolute temperature [K], is this case considered as room temperature at 25 oC

    suma = Raman_sigma[gas]['suma'] #Suma values are relative normalized differential Raman scattering cross-sections

# Suma and sigma are the different scattering values for a Raman shift vi (in cm), 
# v0 is the laser wavelength used (in cm) (20 487, 19 435, and 15 802 for 488, 514, and 633 nm, respectively),
# h is Planck’s constant 6.626 P 10 erg s , 
# c is the light velocity 2.998 P 10 cm s , 
# k is Boltzmann’s constant 1.381 P 10 erg K , 
# and T is the absolute temperature.

    cross_section = suma/((((v0-vi)**(-4))/((v0-2331)**(-4)))*(1-(np.exp((-h*c*vi)/(k*T)))))
    #formula from Burke (2001) Lithos after Schrotter and ̈Klockner (1979)

    cross_section = round(cross_section,1)

    return cross_section