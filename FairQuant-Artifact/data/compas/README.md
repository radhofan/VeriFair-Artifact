# COMPAS Recividism

## Source & Data Set Description:
<https://github.com/propublica/compas-analysis>
<https://github.com/adebayoj/fairml/>

The original dataset is provided by ProPublica to predict recidivism.
We utilize the preprocessed dataset by FairML.

`gen_dataset.py` reads the raw data and creates the `.npy` file after preprocessing.
`train.ipynb` reads the `.npy` file and trains neural networks in TensorFlow.

## Relevant Papers

* Julia Angwin, Jeff Larson, Surya Mattu and Lauren Kirchner, "Machine Bias",  ProPublica, 2016, https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing.

* Julius Adebayo, "FairML: ToolBox for diagnosing bias in predictive modeling", Massachusetts Institute of Technology, Department of Electrical Engineering and Computer Science, 2016, https://dspace.mit.edu/handle/1721.1/108212.