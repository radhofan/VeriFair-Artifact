# FairQuant

This repository contains the source code, benchmark models, and datasets for the paper
**"FairQuant: Certifying and Quantifying Fairness of Deep Neural Networks"**, presented in ICSE 2025 at Ottawa, Canada.

The PDF can be found at: https://arxiv.org/abs/2409.03220

This artifact can also be found in Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14890262.svg)](https://doi.org/10.5281/zenodo.14890262)




### Authors
* Brian Hyeongseok Kim, University of Southern California (brian.hs.kim@usc.edu)
* Jingbo Wang, Purdue University (wang6203@purdue.edu)
* Chao Wang, University of Southern California (wang626@usc.edu)


## Instructions
These instructions assume that you are running Ubuntu 20.04.

### Using the VM
For convenience, we provide a VirtualBox VM with FairQuant and Fairify working out-of-the-box [here](https://drive.google.com/file/d/1tmTgk5DYLvVjayTvDOyDlGNj9mntyE3a/view?usp=drive_link).
To use this VM, please download the [VirtualBox](https://www.virtualbox.org/) (the version we used is 7.0).
Once you open the application, you can click "Add" and select the .vbox file inside the folder, which should open up the VM. The VM login username and password are both "fairquant".
The contents of this repo can be found under `/home/fairquant/FairQuant-Artifact/` inside the VM.

If you are using the VM, you may skip the next section about local installation and proceed to "Run the tool".



## Local Installation

Here, we provide additional instructions for local installation without the VM.

First, clone the [artifact repository](https://github.com/briankim113/FairQuant-Artifact) by running:
```
git clone https://github.com/briankim113/FairQuant-Artifact.git
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


### Fairify (for comparison)

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


## Run the tool
If you are using the VM or have installed everything locally, you can follow the instructions below to run the two tools.

Before doing so, please note that the `compas-7` network files are compressed as .tar.gz files, so please extract them by running the following command:
```shell
cd models/compas
tar -xzf compas-7.h5.tar.gz  # h5 for Fairify
tar -xzf compas-7.nnet.tar.gz  # nnet for FairQuant 
```


### FairQuant
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


### Fairify (for comparison)
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

As an aside, `Verify-compas.py` will only run `compas-1.h5` by default because it does not scale to other larger models. You can change this by setting `onlyRunFirstModel` variable to False.

The results are saved under `Fairify/src/[dataset_name]/res` folder as .csv files.
These are the raw results from the Fairify tool.

To interpret the results in the format of Table 2, please run `interpret_fairify.py` file in the `Fairify/` folder.
You can run it as follows: 
```shell
python3 interpret_fairify.py [dataset] [csv_file]
```
It will output the Certified, Falsified, Undecided rates as well as the number of counterexamples.

Note that Fairify's original implementation randomly shuffles the order of partitions prior to verification step, causing the reproduced results to vary marginally with each run and may differ slightly from the Fairify results in Table 2 of the paper.  
Furthermore, the tool continues the verification for a given partition if it started before the timeout is called; for the comparison in the paper, we manually interrupt its verification after the 30 minute timeout for fair comparison.


## Index

**Tools**
1. [FairQuant source code](./FairQuant/)
2. [Fairify source code](./Fairify/) (for comparison)

**Benchmarks**
1. [Datasets](./data)
2. [Models](./models)

The source code for our tool FairQuant is adapted from the following projects:
1. [ReluVal (USENIX Security '18)](https://github.com/tcwangshiqi-columbia/ReluVal)
2. [ReluDiff (ICSE '20)](https://github.com/pauls658/ReluDiff-ICSE2020-Artifact)



## Badges

**Available**:
Our artifact is publicly shared in the GitHub repository: https://github.com/briankim113/FairQuant-Artifact. 
Additionally, we provide an archival release on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14890262.svg)](https://doi.org/10.5281/zenodo.14890262).

**Functional and Reusable**:
Our artifact contains both the software components and the data with full functionality to reproduce the results from the associated paper. This includes the source code for FairQuant, the code we used for Fairify for comparison, the four datasets used for training, and the neural network models used for evaluation. We further release both the original h5 versions and the nnet-translated versions of the network models, for the different tool requirements.

We provide both a VM with everything working out-of-the-box and detailed instructions on how to install everything locally on a fresh Ubuntu 20.04 version.
Furthermore, we provide automated scripts and detailed instructions to run easily all of the experiments in the associated paper.
