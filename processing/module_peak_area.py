import numpy as np
import pandas as pd

import rampy as rp
from sklearn.metrics import auc

#import matplotlib.pyplot as plt


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
    
    
    
    #plot peak features
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,5))

#     ax1.plot(x,y,"b.",label="data")
#     ax1.plot(CO2_v1_x,y_CO2_v1_PLS,"k-",label="model")
#     ax1.plot(CO2_v1_x,back_CO2_v1,"r--",label="baseline")
#     ax1.set_title('CO2_v1')
#     ax1.set_xlim([1230,1310])
#     ax2.plot(x,y,"b.",label="data")
#     ax2.plot(CO2_v2_x,y_CO2_v2_PLS,"k-",label="model")
#     ax2.plot(CO2_v2_x,back_CO2_v2,"r--",label="baseline")
#     ax2.set_title('CO2_2v2')
#     ax2.set_xlim([1350,1430])
#     ax3.plot(x,y,"b.",label="data")
#     ax3.plot(N2_x,y_N2_PLS,"k-",label="model")
#     ax3.plot(N2_x,back_N2,"r--",label="baseline")
#     ax3.set_title('N2_v1')
#     ax3.set_xlim([2290,2360])
#     ax4.plot(x,y,"b.",label="data")
#     ax4.plot(CH4_x,y_CH4_PLS,"k-",label="model")
#     ax4.plot(CH4_x,back_CH4,"r--",label="baseline")
#     ax4.set_title('CH4_v1')
#     ax4.set_xlim([2880,2940])

#     ax1.set_ylim([0,500])
#     ax2.set_ylim([0,500])
#     ax3.set_ylim([0,500])
#     ax4.set_ylim([0,4000])

#     plt.legend()
#     print(plt.show())
    
    return df_peak_areas