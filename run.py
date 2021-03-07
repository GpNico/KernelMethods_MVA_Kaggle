# run.py
"""
Script for running a specific pipeline from a given yaml config file
"""

import argparse
import yaml
from importlib import import_module
import numpy as np

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
    parser.add_argument("-o", "--output", help = "Path to the output file")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
        
    if args.output != None:
        output = True
        out_csv = args.output
        dfs = []
    else:
        output = False
    
    # Importing pipeline elements
    ds_splitter = import_from_path(config["split"]["filepath"],
                                   config["split"]["class"]) (**config["split"]["parameters"])

    preprocess = import_from_path(config["preprocess"]["filepath"])

    model_params = config["model"]["parameters"]
    if "kernel" in model_params:
        kernel_func = import_from_path(model_params["kernel"]["filepath"],
                                       model_params["kernel"]["class"])
        kernel_params = model_params["kernel"]["parameters"]
        model_params["kernel"] = lambda X, Y: kernel_func(X,Y,**kernel_params)
    model = import_from_path(config["model"]["filepath"],
                             config["model"]["class"])(**config["model"]["parameters"])
    
    evaluation = import_from_path(config["evaluation"]["filepath"])

    # Lists filling information for the output dataframe
    datasets = []
    metrics = []
    values = []

    # Applying pipeline
    # Iterate over datasets
    for i, dataset in enumerate(config["datasets"]):
        # Read dataset
        X = pd.read_csv(dataset["X"]["filepath"],
                        **dataset["X"]["parameters"])
        ## It is currently very important to drop Id before splitting or preprocessing
        y = pd.read_csv(dataset["y"]["filepath"],
                        **dataset["y"]["parameters"]).drop("Id", axis = 1)
        
        if output:
            test = pd.read_csv(dataset["test"]["filepath"],
                            **dataset["test"]["parameters"])

        # Split dataset
        ds_splitter.generate_idx(y)
        X_train, X_test = ds_splitter.split(X)
        y_train, y_test = ds_splitter.split(y)

        # Preprocess dataset
        for transform in config["preprocess"]["X"]:
            X_train = getattr(preprocess, transform["transform"])(X_train, **transform["parameters"])
            X_test = getattr(preprocess, transform["transform"])(X_test, **transform["parameters"])

        for transform in config["preprocess"]["y"]:
            y_train = getattr(preprocess, transform["transform"])(y_train, **transform["parameters"])
            y_test = getattr(preprocess, transform["transform"])(y_test, **transform["parameters"])
        
        if output:
            for transform in config["preprocess"]["X"]:
                test = getattr(preprocess, transform["transform"])(test, **transform["parameters"])

        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if output:
            y_pred_test = model.predict(test)
            y_pred_test = (y_pred_test + 1)/2
            id = np.arange(1000*i, 1000*(i + 1))
            dic = {'Id': id, 'Bound': y_pred_test[:, 0]}
            df = pd.DataFrame(data = dic)
            dfs.append(df)
            

        # Evaluate model
        for metric in config["evaluation"]["metrics"]:
            datasets.append(dataset["name"])
            metrics.append(metric)
            values.append(getattr(evaluation, metric)(y_pred, y_test))
    
    df = pd.concat(dfs).astype('int32')
    df.to_csv('submissions/' + out_csv, index = False)
        
    results = {"datasets": datasets, "metrics": metrics, "values": values}
    print(pd.DataFrame.from_dict(results))