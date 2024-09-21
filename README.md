# WGAN-GP data generation With Reconstruction Error Metric (REM)
This repository contains the main classes and functions needed to understand how to replicate the experiments on your data.

All our experiments were conducted on the IBISC lab server using an NVIDIA A40 GPU. Due to memory constraints and the large number of experiments, we have decided to include only the main experiments and scripts from the server in this repository.

# Requirements
Install the required python librairies:
if you are using conda: conda install --file requirements.txt
if not: pip install -r requirements.txt

# The data
The data is available at:
https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-3732

To preprocess it after dowloading it

at first adjust the data path in the preprocessing.py script in the data folder then run: python data_preprocessing.py 

# To Train WGAN-GP on your data
Run: python Wgan_best.py
## To tune the WGAN-GP 
Run: OptunaWgan.py

# To train the multiclass classification on the data preprocessed data (data 32403 features)
Run: MLP5times.py

# Metrics: Reconstruction Error Metric REM
To calculate precision and recall use the precision_recall.py script in the metrics in src folder.

To calculate our Metric RE use REM.py script in the metrics in src folder.
