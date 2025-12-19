# VeriFair

This repository contains the source code, benchmark models, and datasets for the paper
**"Verification-Guided Fairness Repair for Neural Networks"**, presented in [Conf Place].

## Model Repair

### VeriFair

We recommend to use a conda environment to make things easier:

```shell
conda create -n verifair python=3.7
conda activate verifair
```

Run the setup script to install requirements

```shell
bash setup.sh
```

Use run_repair.sh to automate the evaluation pipeline across datasets:

```shell
bash run_repair.sh [DATASET]
```

Supported Datasets:
AC - Adult
BM - Bank Marketing
GC - German Credit
DF - Default
UCI - Student Performance

Example Usage:

```shell
bash run_repair.sh AC
bash run_repair.sh BM
bash run_repair.sh GC
bash run_repair.sh DF
bash run_repair.sh UCI
```

## Counterexample Generation

### Fairify

The current version has been tested on Python 3.7.
It is recommended to install Python virtual environment for the tool.
You can install the Python version and its virtual environment by running:

```shell
sudo apt install python3.7 python3.7-venv
```

First, navigate to the Fairify folder and create a Python virtual environment.

```shell
cd Fairify
python3.7 -m venv fenv
```

Then, activate the created virtual environment and check that it is set up correctly.

```shell
source fenv/bin/activate
python --version
python -m pip --version
```

Finally, install the required packages:

```shell
python -m pip install -r requirements.txt
```

First, navigate to the Fairify folder and activate the virtual Python environment:

```shell
cd Fairify/
source fenv/bin/activate
```

Now, navigate to the src folder and run the script file to verify on all the models for any of the four evaluated datasets. The protected attributes used in the evaluation are the same as those presented in Table 2 of the paper.

```shell
cd src/
./fairify.sh [dataset] # 'AC' for Adult, 'GC' for German, 'BM' for Bank, 'compas' for Compas
```

### FairQuant

Ensure that you have `make` and `gcc` installed on your machine (the versions we used are GNU Make 4.2.1 and gcc 9.4.0).

The only additional installation is for [OpenBLAS](http://www.openblas.net).
You can download the library by following the [OpenBLAS's Installation Guide](https://github.com/OpenMathLib/OpenBLAS/wiki/Installation-Guide) or running the following commands:

```shell
# Download the tar file
wget https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.6/OpenBLAS-0.3.6.tar.gz
tar -xzf OpenBLAS-0.3.6.tar.gz

# Set up installation path
export INSTALL_PREFIX=$HOME/OpenBLAS # or wherever you want to install OpenBLAS
mkdir $INSTALL_PREFIX

# Install
cd OpenBLAS-0.3.6
make
make PREFIX=$INSTALL_PREFIX install

# Check that OpenBLAS has been installed correctly
ls $INSTALL_PREFIX/include # you should see files such as cblas.h
ls $INSTALL_PREFIX/lib # you should see files such as libopenblas.so
```

Once OpenBLAS has been successfully installed in `$INSTALL_PREFIX`, you can remove the tar file and the OpenBLAS-0.3.6 folder.

First, set up the paths so the compiler and runtime can find the headers and libraries:

```shell
export INSTALL_PREFIX=$HOME/OpenBLAS # or wherever you have installed OpenBLAS
export LIBRARY_PATH=$LIBRARY_PATH:$INSTALL_PREFIX/lib
export C_INCLUDE_PATH=$LD_LIBRARY_PATH:$INSTALL_PREFIX/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INSTALL_PREFIX/lib
```

Now, navigate to the FairQuant folder and generate the executable file that will be used by the script files:

```shell
cd FairQuant
make all # generate 'network_test' exec file
```

Finally, run the desired verification on any of the four datasets using their corresponding scripts. Note that the script files take one protected attribute (PA) as its argument.

```shell
./adult.sh [PA] # 'sex' (in paper)
./bank.sh [PA] # 'age' (in paper)
./german.sh [PA] # 'age' (in paper) or 'sex'
./compas.sh [PA] # 'race' (in paper) or 'sex' / 'age'
```

The results of running these scripts are also saved under `FairQuant/res` folder as .txt files.
Specifically, they report the time the verification problem took, number of counterexamples found, as well as final Certified, Falsified, and Undecided rates.
These are the data reported as FairQuant results in Table 2 of the paper.
