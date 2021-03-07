# datasetPreprocess.py
# Preprocessing should should have a fit and a transform step

def convert_numpy(df):
    return df.to_numpy()

def drop_columns(df, ids):
    return df.drop(ids, axis = 1)

def recode(df):
    df["Bound"] = 2*df["Bound"] - 1
    return df
    
def squeeze(array):
    return array.squeeze()