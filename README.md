# Directly quantify verbatim copy in Quick sampling realisations

This is the codebase supporting my masters thesis.

The results and figure creation can be found in the files mentioned below.
The other files are either supporting functions or experimental code. 
Most results code blocks show either a way to generate the data or to load it from a file.

The QS data files path are configured in "data.py".
The generation of these files was done using "generateSim.py" (code authored by Dr. Mathieu Gravey https://www.mgravey.com/)


QS Stones results: verbatim_metrics/qs_data_results.py

QS Strebelle results: verbatim_metrics/qs_strebelle_data_results.py

Synthetic data results: verbatim_metrics/synth_data_results.py

PCA data generation: verbatim_metrics/pca_results.py


The code is separated in code blocks denoted by "#%%", most IDE's can handle these.
All code was run on an Apple M1 processor with 8GB RAM, 8 core cpu (4 high performance).
No code block took longer than 1 hour.
