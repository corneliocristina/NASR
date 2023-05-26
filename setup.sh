mkdir data

conda create --name nasr python=3.9
conda activate nasr

mkdir data/original_data

pip install pyswip
conda install joblib
conda install numpy
conda install matplotlib
conda install -c conda-forge mnist
conda install tqdm
conda install h5py

python3.9 -m pip install ipykernel

conda install -c conda-forge optuna