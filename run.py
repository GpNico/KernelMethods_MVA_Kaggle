# run.py
"""
Script for running a specific pipeline from a given yaml config file
"""

import argparse
import yaml
from importlib import import_module

import pandas as pd

def import_from_path(path_to_module, obj_name = None):
    """
    Import an object from a module based on the filepath of
    the module and the string name of the object.

    If obj_name is None, return the module instead.
    """
    module_name = path_to_module.replace("/",".").strip(".py")
    module = import_module(module_name)
    if obj_name == None:
        return module
    obj = getattr(module, obj_name)
    return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("-c", "--config", help = "File path to the config file")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    
    # import pdb; pdb.set_trace()
    # Read dataset
    X = pd.read_csv(config["dataset"]["X"]["filepath"],
                    delimiter = " ", header = None)
    ## It is currently very important to drop Id before splitting or preprocessing
    y = pd.read_csv(config["dataset"]["y"]["filepath"]).drop("Id", axis = 1) 

    # Split dataset
    ds_splitter = import_from_path(config["split"]["filepath"],
                                   config["split"]["class"]) (**config["split"]["parameters"])

    ds_splitter.generate_idx(y)
    X_train, X_test = ds_splitter.split(X)
    y_train, y_test = ds_splitter.split(y)

    # Preprocess dataset
    preprocess = import_from_path(config["preprocess"]["filepath"])

    for transform in config["preprocess"]["X"]:
        X_train = getattr(preprocess, transform["transform"])(X_train, **transform["parameters"])
        X_test = getattr(preprocess, transform["transform"])(X_test, **transform["parameters"])

    for transform in config["preprocess"]["y"]:
        y_train = getattr(preprocess, transform["transform"])(y_train, **transform["parameters"])
        y_test = getattr(preprocess, transform["transform"])(y_test, **transform["parameters"])

    # Fit model
    model_params = config["model"]["parameters"]
    if "kernel" in model_params:
        model_params["kernel"] = import_from_path(model_params["kernel"]["filepath"],
                                                  model_params["kernel"]["class"])
    model = import_from_path(config["model"]["filepath"],
                             config["model"]["class"])(**config["model"]["parameters"])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    evaluation = import_from_path(config["evaluation"]["filepath"])
    metrics = []
    values = []
    for metric in config["evaluation"]["metrics"]:
        metrics.append(metric)
        values.append(getattr(evaluation, metric)(y_pred, y_test))
    
    results = {"metrics": metrics, "values": values}
    print(pd.DataFrame.from_dict(results))