conda create -n py38_tf python=3.8
# installs metaphlan for getting bacterial species frequencies 
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda install  -y -c bioconda python=3.7 metaphlan
# installs wrapper for fastq dumbps
conda install -y -c bioconda bioinfokit
# installs kneaddata for preprocessing
conda install -c bioconda kneaddata
# other useful packages
conda install numpy -y
conda install scipy -y
conda install pandas -y
conda install seaborn -y
conda install scikit-learn -y
conda install -c conda-forge matplotlib -y
conda install -c conda-forge scikit-plot -y
conda install -c conda-forge xgboost -y
conda install jupyter -y
conda install openpyxl -y
conda install tensorflow -y
conda install keras -y
conda install -c anaconda pydot
