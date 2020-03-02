# FALL
Fast anchor regularized local linear model

## Authors
This work has been done at RIKEN AIP by Mathis Petrovich and Makoto Yamada.


## Installation
```bash
git clone https://github.com/Mathux/fall.git
cd fall
```

### Python requirements
This code needs Python 3 and the following packages:

* click
* numpy
* pandas
* optuna
* sklearn
* matplotlib

The packages can be installed with:

```bash
pip install --user -r requirements.txt
```

## Usage
The implementation of our method can be found in ``src/models/fall.py``. Please import ``Fall`` from this file and use it as a sklearn model (``model.fit`` and ``model.predict``)

```python3
# Import the model
from src.models.fall import Fall

# Import some dataset
from src.data.concrete import Concrete

# Create the model
model = Fall(k=40, K_anchors=20, K_prediction=5, lam=10)

# Load the data
data = Concrete().datasets()

# Fit the model
model.fit(data["train"]["X"], data["train"]["Y"])

# Predict with the test data
y_pred = model.predict(data["test"]["X"])

# Compute the mean squared error
from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_pred, data["test"]["Y"])
```


## Experiments
```
Usage: python main.py [OPTIONS]

Options:
  --dataset [fishtoxicity|aquatictoxicity|concrete|superconduct|parkinson_updrs_motor|parkinson_updrs_total]
  --model [all|fall|netlasso|ridge|lasso|knn|krr]
  --cv INTEGER                    Number of splits for the cross-validation.
  --nepochs INTEGER               Number of times to compute everything.
  --timeout INTEGER               Timeout for computing the best hyper
                                  parameters.
  --output TEXT                   Name of the latex output file to store the
                                  results.
  --verbose / --no-verbose        Verbosity for optuna.
  --help                          Show this message and exit.
```

For exemple to test our method on the fishtoxicity dataset for 1 epochs, by doing a 3-cross-validation with 3 seconds to find the hyper parameters:

```bash
python main.py --dataset fishtoxicity --model fall --cv 3 --nepochs 1 --timeout 3
```

## Figures
You can recreate the figures of the paper by running this.

```
Usage: python figures.py [OPTIONS]

Options:
  --folder TEXT       Folder to store the figures.
  --show / --no-show  Show the plots during the generation.
  --help              Show this message and exit.
```

## Reference
If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper:

```
@techreport{PetM:2020,
  author = "Petrovich, Mathis and Yamada, Makoto",
  title = "Fast local linear regression with anchor regularization",
  year = "2020",
  month = "2",
  url = "http://hdl.handle.net/2433/245860",
  abstract = "Regression is an important task in machine learning and data mining. It has several applications in various domains, including finance, biomedical, and computer vision. Recently, network Lasso, which estimates local models by making clusters using the network information, was proposed and its superior performance was demonstrated. In this study, we propose a simple yet effective local model training algorithm called the fast anchor regularized local linear method (FALL). More specifically, we train a local model for each sample by regularizing it with precomputed anchor models. The key advantage of the proposed algorithm is that we can obtain a closed-form solution with only matrix multiplication; additionally, the proposed algorithm is easily interpretable, fast to compute and parallelizable. Through experiments on synthetic and real-world datasets, we demonstrate that FALL compares favorably in terms of accuracy with the state-ofthe- art network Lasso algorithm with significantly smaller training time (two orders of magnitude)."
}
```
