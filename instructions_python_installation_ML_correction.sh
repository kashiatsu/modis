#Download Anaconda in your home directory 
#wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
#Install the anaconda using 
#bash Anaconda3-2022.10-Linux-x86_64.sh 

#Activate conda using
#source anaconda3/bin/activate

#Update conda using 
#conda update conda

#Create env using 
module load anaconda3-py/2020.11

conda create -n ml_chimere python=3.8

conda activate ml_chimere

conda config --add channels conda-forge

conda install --file  requirement.txt

pip3 install pyyaml==6.0
pip3 install pyaml==21.10.1

conda install -n ml_chimere cartopy==0.21.0
conda install -n ml_chimere xarray
conda install -n ml_chimere tensorflow==2.10.0
conda install -n ml_chimere scikit-learn==1.1.2
conda install -n ml_chimere netcdf4


conda install jupyterlab
pip install lckr-jupyterlab-variableinspector

conda install -c anaconda xarray

pip install netcdf4
pip install h5netcdf
pip install scipy
pip install store
conda install -n ml_chimere libxgboost==1.5.1

conda install -c anaconda -n ml_chimere keras==2.10.0
conda install -c anaconda -n ml_chimere keras-tuner




#1. installer FoxyPoxy sur ton ordi
#https://addons.mozilla.org/fr/firefox/addon/foxyproxy-standard/
#Dans les 'Options' de FoxyProxy Ajouter un proxy avec le port 1080 
#Type de Proxy: SOCKS5,  Adress IP: localhost, Indiquer "jupyter" dans menu depliant
#Activer le proxy.

#2. se connecter par ssh sur vangogh2 avec le port 1080:
ssh -XY cuesta@vangogh2.lisa.u-pec.fr -D 1080

#3. lancer le jupyter sur vangogh2 pour le port 1080:
jupyter notebook --no-browser --ip=0.0.0.0 --port=1080

#4. Ouvrir l'adresse générée sur l'écran dans le Firefox.
http://127.0.0.1:1080/?token=
#5. Tuer le processus si besoin à la fin.
