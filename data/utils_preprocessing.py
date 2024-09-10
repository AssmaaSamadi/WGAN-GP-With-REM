import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_test_val(labelcell,celldata,valratio):

    """ Helper function to split data between train and validation
    
    Parameters:
        labelcell(pandas.DataFrame): The labels.
        celldata (pandas.DataFrame): The expression level data
        valration (float): The needed ratio to split data between test and validation set
    Returns:
        X_train (pandas.DataFrame): The training expression level data.
        y_train (pandas.DataFrame): The training labels.
        X_test (pandas.DataFrame): The testing expression level data.
        y_test (pandas.DataFrame): The testing labels.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(celldata, labelcell, test_size=valratio, stratify=labelcell)
    #X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=valratio, stratify=y_train)
    return X_train,y_train, X_test, y_test 

def standardized(X): 
    
    """ Helper function to standardized the expression level data
    
    Parameters:
    
        X(pandas.DataFrame): The expression level.
        
    Returns:
    
        X (numpy.ndarray): The standard scaled data.
        
    """
    X = StandardScaler().fit_transform(X)
    
    return X

 #encode cancer type  
def map_labels_to_indices(y):
    """ Helper function to encode cancer phenotype to 0 and 1
    Parameters:
        y (numpy.ndarray): Cancer phenotype.
        
    Returns:
        encoded (torch.tensor): The encoded cancer phenotype
    """
        
    label_to_index = {"normal": 0, "cancer": 1}
    
    encoded = torch.tensor([label_to_index[label] for label in y])
    
    return encoded

def map_type_to_indices(y):
    """ Helper function to encode data type (cell line / patient) to 0 and 1
    Parameters:
        y (numpy.ndarray): Cancer phenotype.
        
    Returns:
        encoded (torch.tensor): The encoded data type
    """
        
    label_to_index = {"cell line": 0, "patient": 1}
    
    encoded = torch.tensor([label_to_index[label] for label in y])
    
    return encoded
def df_to_label(data):
    
    """ Helper function to separate expression data,tissue labels and cancer phenotype into 3 data frame
    
    Parameters:
        data (pandas.DataFrame): The data frame that contains expression level and its labels
    Returns:
        data (pandas.DataFrame): Probes expression level.
        labelc (pandas.DataFrame): Cancer phenotype
        labelp (pandas.DataFrame): Tissue types
    """
    
    labelc = data['Cancer']
    labelp = data['Part']
    data.drop('Cancer', axis=1, inplace=True)
    data.drop('Part', axis=1, inplace=True)
    
    return data,labelc,labelp
