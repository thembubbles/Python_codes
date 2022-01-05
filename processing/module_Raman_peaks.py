# --- function to import data from spectrum and peak files and create a combined dataframe

def spectrum_data(spectrum_file):
    
    # --- PART 1: import spectrum data from input file and create a spectrum dataframe

        # --- set file name to be processed
    current_mineral = spectrum_file.replace(".dat", "").replace(".txt", "")

            # --- get input file and save as a spectrum dataframe

    df_spectrum = pd.read_csv(base_dir+"spectra/"+spectrum_file, header=None, skiprows=1, sep='\s+',
                             names=['XY', 'X Value', 'Y Value', 'Y Predict', 'Residual', 'Residual%', '90% Confidence', 'Limits', '90% Prediction', 'Limits2', 'Weights'])

    df_spectrum = df_spectrum.drop(columns=['XY']) 
        
        # --- add file information
    df_spectrum["file"] = current_mineral
    

                
       

    # --- PART 2: get corresponding peak information from input file 
         
    separator = input("choose the file separator (e.g., ',' ';' '/s' '/t': ")

        # --- get lines from processed peak file

    current_peak = open(base_dir+"peaks/"+current_mineral+'_peaks.txt', "r")
    original_current_lines = current_peak.readlines()
    current_lines = []

    for line in original_current_lines:
        partial_lines = line.split("  ")
        while "" in partial_lines:
            partial_lines.remove("")
        str_line = ""
        for item in partial_lines:
            str_line = str_line + item.replace(" ","")+ separator
        current_lines.append(str_line)

        # --- get values from lines

    counter = 0

    for line in current_lines:
        if "r^2CoefDet" in line:
            break
        counter+=1

    r2 = current_lines[counter+1].split(separator)[0]

        # --- get total number of peaks from last peak entry, at the end of peak file

    counter = 0

    for line in current_lines[::-1]:
        if "Peak" in line:
            peak_line = line
            break

    total_peaks = peak_line.split(separator)
    peak_number = int(total_peaks[0].replace("Peak",""))
    peak_number

        # --- get positions of overall tables

    counter = 0
    peaks_tables_loc = []

    for line in current_lines:
        if "Peak"+separator in line:
            peaks_tables_loc.append(counter)
        counter += 1

        # --- grab data per peak for each table

    peaks_dict = {}

    for ii in range(peak_number):
        peaks_dict[str(ii+1)] = {}

    for line_counter in peaks_tables_loc:    
        current_attributes = current_lines[line_counter].split(separator)

        for iii in range(line_counter+1,(line_counter+peak_number+1)):
            split_line = current_lines[iii].split(separator)

            kk=0

            for attribute in current_attributes: 

                peaks_dict[str(iii-(line_counter))][attribute.replace("\n","")] = split_line[kk].replace("\n","")
                kk+=1

            # --- final peak table

    df_peaks = pd.DataFrame.from_dict(peaks_dict, orient="index")

    df_peaks.drop("",axis=1,inplace=True)

    df_peaks["r2"] = r2
    df_peaks["mineral"] = current_mineral

    #return df_peaks

          

    # --- PART 3: combine peak parameters with spectrum dataframe

    df_spectrum_peaks = df_spectrum.copy()

        # --- create tuples that will receive each peak parameter

    wv_range = np.array(df_spectrum["X Value"])
    current_center_range = np.zeros(df_spectrum.shape[0])
    current_height_range = np.zeros(df_spectrum.shape[0])
    current_fwhm_range = np.zeros(df_spectrum.shape[0])
    current_a0_range = np.zeros(df_spectrum.shape[0])
    current_a1_range = np.zeros(df_spectrum.shape[0])
    current_a2_range = np.zeros(df_spectrum.shape[0])
    current_a3_range = np.zeros(df_spectrum.shape[0])
    

        # --- copy peak parameters from peak dataframe


    peak_parameters = df_peaks[['Center','Amplitude','FWHM','a0','a1','a2','a3']].astype(float) 
    # here a float conversion is needed because df_peaks has only strings, and np.where condition will compare wv_range numbers
    
   
        # --- loop through peaks in peak dataframe to copy and assign values to tuples created

    for peak, value in peak_parameters['Center'].iteritems():    
        current_center_range[np.where(wv_range>=value)[0][0]] = value
        current_height_range[np.where(wv_range>=value)[0][0]] = peak_parameters['Amplitude'][peak]
        current_fwhm_range[np.where(wv_range>=value)[0][0]] = peak_parameters['FWHM'][peak]
        current_a0_range[np.where(wv_range>=value)[0][0]] = peak_parameters['a0'][peak]
        current_a1_range[np.where(wv_range>=value)[0][0]] = peak_parameters['a1'][peak]
        current_a2_range[np.where(wv_range>=value)[0][0]] = peak_parameters['a2'][peak]
        current_a3_range[np.where(wv_range>=value)[0][0]] = peak_parameters['a3'][peak]
              
        
        # --- assign tuples with peak parameters as columns in the spectra dataframe

    df_spectrum_peaks["center"] = current_center_range
    df_spectrum_peaks["height"] = current_height_range
    df_spectrum_peaks["FWHM"] = current_fwhm_range
    df_spectrum_peaks["G/L"] = current_a3_range
    df_spectrum_peaks["a0"] = current_a0_range
    df_spectrum_peaks["a1"] = current_a1_range
    df_spectrum_peaks["a2"] = current_a2_range
    df_spectrum_peaks["a3"] = current_a3_range
    df_spectrum_peaks["R2"] = r2

    
        # --- Convert 'a' parameters to numeric for later calculations
    df_spectrum_peaks['a0'] = pd.to_numeric(df_spectrum_peaks['a0'])
    df_spectrum_peaks['a1'] = pd.to_numeric(df_spectrum_peaks['a1'])
    df_spectrum_peaks['a2'] = pd.to_numeric(df_spectrum_peaks['a2'])
    df_spectrum_peaks['a3'] = pd.to_numeric(df_spectrum_peaks['a3'])

    #df_spectrum_peaks.to_csv(current_mineral+".csv",index = None)
    return df_spectrum_peaks





# --- function to plot curves with the Gaussian+Lorentzian sum profile

def gauss_lor_sum(x, a_0, a_1, a_2, a_3):
    return (a_0*(((((a_3*np.sqrt(np.log(2)))/(a_2*np.sqrt(np.pi)))*(np.exp(-4*np.log(2)*(((x-a_1)/a_2)**2))))+((1-a_3)/(np.pi*a_2*(1+(4*(((x-a_1)/a_2)**2))))))/
                 (((a_3*np.sqrt(np.log(2)))/(a_2*np.sqrt(np.pi)))+((1-a_3)/(np.pi*a_2)))))
