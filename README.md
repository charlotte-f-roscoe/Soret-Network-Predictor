# Soret Network Predictor
 **"Using Machine Learning to Predict Network Performance of a Micro-Structured Gas Separator Using the Soret Effect"**

Link to Report - TBA
 
 Charlotte Roscoe
 Worcester Polytechnic Institute -  Major Qualifying Project (MQP)
 01/26/2024

## Abstract

Hydrogen is a promising clean energy source, emitting no CO2 during use. However, production methods rely on fossil fuels, contributing to carbon emissions. To combat this, the Soret Team at the Shibaura Institute of Technology in Tokyo is investigating hydrogen separation using a thermophoretic microstructured separator device. Alternatively, this project simulates using multiple smaller devices in a network structure, employing a Random Cut Forest machine-learning regression model to predict separation. A custom GUI was developed to visualize and interact with this network. Results showed potential benefits in scalability. Future work involves replacing synthetic datasets with experimental data. This project advances clean energy and highlights the benefits of interdisciplinary research.


## Files
*data_extrapolation.py*
* Python file used to extrapolate base dataset.

*Full_Data_FINAL.csv*
* Complete dataset used for the machine learning model.
* Used in *separation_prediction_model.py*

*Old_Data.csv*
* Experimental data from past SIT experiments after projection, but before extrapolation.
* Used in *data_extrapolation.py*

*README.md*
* Current file -- README.

*separation_prediction_model.py*
* Soret network prediction program.

*SIT_Simulations.csv*
* Additional simulations provided by the Shibaura Institute of Technology.