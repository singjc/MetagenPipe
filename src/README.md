# README for src scripts

* download_raw_tables.sh:
	- exactly what it sounds like. download supplementary data files and preprocessed data for Gupta 2020 paper
* Gupta_2020_Precompiled_Cleaned.ipynb:
	- cleanup precompiled data from authors
* clean_Gupta_2020_tables.ipynb:
	- cleans up supplemnentary tables 1 and 4. TODO: change name of this file

* MicroBiome.py:
	- utilities for doing analysis and modeling with precompiled data
* LogisticRegressionCV.ipynb:
	- train logistic regression model and assess accurary for precompiled data
* PCA.ipynb:
	- PCA analysis of microbiome data

* setup.sh:
	- setup script called by Dockerfile
