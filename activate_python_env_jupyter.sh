
export PYTHONPATH=${PYTHONPATH}:/DATA/IASI/INPUT/juan/SPECAT_2/ML_CHIMERE_AOD_correction/lib

module load anaconda3-py/2020.11

conda activate ml_chimere

jupyter lab --no-browser --ip=0.0.0.0 --port=1080


