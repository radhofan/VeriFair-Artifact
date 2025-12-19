# VeriFair

This repository contains the source code, benchmark models, and datasets for the paper
**"Verification-Guided Fairness Repair for Neural Networks"**, presented in [Conf Place].

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
