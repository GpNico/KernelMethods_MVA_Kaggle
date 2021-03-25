# KernelMethods_MVA_Kaggle
Code associated to the MVA Kaggle Challenge of the course Kernel Methods.

# Suggested Plan

# Technical details

## Requirements:
- Python 3.7.1

## Installation

### Anaconda

Create the conda environment from the `environment.yml` file
```
conda env create -f environment.yml
```

If you have installed a new package, please export the environment:
```
conda env export --no-builds | findstr -v "prefix" > environment.yml
```
## Test

To test if all is working (after modifying a file for example) run in terminal :

```
pytest
```

## Run

To run a specific config, use the following command:
```{bash}
python run.py --config pipelines/gaussianKernelRidge.yml --output gaussianKernelRidge.csv
```
The `--output` flag correspond to the name of the csv storing the predictions of the model, it will be stored by default in the `submissions` directory. 

To obtain our final model predictions, use the `start.py` script:
```{bash}
python start.py
```