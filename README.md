**Investigating the Streamflow Simulation Capability of a New Mass-Conserving Long Short-Term Memory (MC-LSTM) Model at Eight Watersheds Across the Contiguous United States**

Thank you for your interest in our research! 

This repository contains the preliminary version of the code needed to reproduce our findings. Updates may follow to enhance the code's accuracy and functionality.

**How to run the code (with minimal modification on paths. However, you may still need to adjust your paths accordingly as you run the code):**

(1) Create a folder, e.g. "test"

(2) Create a subfolder "Collaborative_Research" under "test".

(3) Download the repository and save to the folder created in (1), e.g. "test/MC-LSTM".

    Create a subfolder "trained_models" under "test/MC-LSTM".
    
    Create a subfolder "figureset" under "test/MC-LSTM".
    
    Create a subfolder "result_128nodes" under test/MC-LSTM
    
(4) Download the data (Data-library/MC-LSTM.zip), unzip it. It contains a folder named "Data". 
Save the folder "Data" to the path "test/Collaborative_Research".
    
(5) Now you can run mp_lstm.py mp_mclstm.py to trian LSTM and MC-LSTM model, repsectively. 

    * The trained models will be saved to "test/MC-LSTM/trained_models".
    
    * The loss curves will be saved to "test/MC-LSTM/figureset".
    
(6) When you have trained models saved, you can run "readin_models.py" to produce model prediction time series to a csv file for each CV fold for each watershed.

    * The csv files will be saved to "test/MC-LSTM/result_128nodes"
    

**Code for illustration (You can change the saving path at the end of the scripts):**
  
(1) mass_conservation_illustration.py: Produce the cumulative error time series plot.

(2) plot_cv_timeseries.py: Produce the time series plot.

(3) scatter_plot.py: Produce the high-flow scatter plot.

(4) wb_analysis.py: Produce the water balance scatter plot (i.e., RD_TC versus Pbias).

(5) sensitivity_analysis.py: Produce the statistical metrics variation with changing learning rates plot in Supporting Information.
