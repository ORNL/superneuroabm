# Installing Cupy for GCP Using NVIDIA GPU
* Make sure that there is enough space on your disk for your VM
### Install NVIDIA-Driver, any over 550 should work. This example used 570

```bash
sudo apt install nvidia-driver-570
sudo reboot
```
### Get CUDA runtime libraries 
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

sudo add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

sudo apt-get update

sudo apt-get -y install cuda

sudo apt-get -y install libcudnn8

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

source ~/.bashrc

nvcc --version
```
### Install Conda using MiniConda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
#ADD TO PATH
echo 'export PATH=$HOME/miniconda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```
### Create an enviorment directory for your conda enviorment and enviorment
```bash
mkdir envs
cd envs
# MAKE ENV
conda create --prefix <Path for env> python=3.11.0
source activate <Path for env>
```
## Cupy installation instructions
Using nvidia-smi check your cuda version. Version tested with is 12.8
Install cupy-cuda12x using pip; modify for cuda version<12 & >12
```bash
python -m pip install --no-cache-dir cupy-cuda12x==13.6.0
#check installation done correctly
python -c "import cupy; print(cupy.__file__)"
#run this as a test to make sure it works
python -c "import cupy as cp; x=cp.array([1,2,3]); print('Array test:', x*2)"
```
### Install Superneuroabm
Clone repo and install dependencies
install mpi4py
```bash 
sudo apt-get update
sudo apt-get install -y libopenmpi-dev openmpi-bin
pip install --no-cache-dir --force-reinstall mpi4py
```
install all other dependencies 
```bash
pip install sagesim==0.4.0dev1
pip install networkx matplotlib pyyaml
pip install superneuroabm==1.0.0dev1
```
Now the unit tests can be run without error
Note at the time of this all functioning tests work off of the 62-update-and-expand-unit-tests
### running test
```bash
python -m unittest /tests/logic_gates_test_lif.py
```