import numpy as np
import torch
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
import itertools
from torch import nn

device = "cuda:3" if torch.cuda.is_available() else "cpu" #0 indicates the GPU you gonna use



def get_accuracy(y_true, y_pred):
    
    """
    Function to calculate testing accuracy
    
    Parameters:
        y_true (numpy.ndarray): The original(True) label
        y_pred (numpy.ndarray): The predicted label by the trained mlp
    
    Returns:
        (float): The testing accuracy 
    """
    return int(np.sum(np.equal(y_true,y_pred))) / y_true.shape[0]

class EarlyStopper(): # in utils
    """
    The class of early stopping of the mlp to avoid model overfitting
    
    """
    
    
    def __init__(self, patience=1, min_delta=0):
        """
        Fucntion to initiate the parameters of the early stopping process
        
        Parameters :
        patience (int): How many epochs to wait after no improvement in loss variation
        min_delta (float): Minimum change in the loss to qualify as an improvement.
        
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        
        """
        The main function of the early stopping
        
        Parameters: 
        validation_loss (float): The current validation loss to assess if we stop the current validation epoch or no
        Returns:
            (bool): True to stop the validation epoch , false to continue
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    
##function to calculate accuracy of trained model 


def test_multiAccuracy(testloader,model,dropout,out_dim,datatype:str):
    """
        path: path where the dictionary state file(.pth) is of the trained classifier
        testloader: testloader data
        -----
        return test accuracy, and real and predicted label
        """
    #first_batch = next(iter(testloader))
    #data, labels = first_batch
# Extract the images from the batch
    #data_shape = tuple(data[0].shape)

# Convert each dimension to an integer
   # input_size = tuple(int(dim) for dim in data_shape)[0]

    output_fn = torch.nn.Softmax(dim=1) # we instantiate the softmax activation function for the output probabilities

    model = model(dropout,out_dim).to(device)
    
    model_name = type(model).__name__
    path = os.path.join( "src", "Classification", "modelpath",f'{model_name}_{datatype}_trainedalllabel2.pth')
    
    model.load_state_dict(torch.load(path))
    
    model.eval()
    with torch.no_grad():
      idx = 0
      for batch in testloader:
        inputs,labels=batch
        inputs=inputs.to(device)
        labels=labels.to(device)
        inputs = inputs.float()
        labels = labels.long()

        if idx==0:
          t_out = model(inputs)
          t_out = output_fn(t_out).detach().cpu().numpy()
          t_out=t_out.argmax(axis=1)
          ground_truth = labels.detach().cpu().numpy()

        else:
          out = model(inputs)
          t_out = np.hstack((t_out,output_fn(out).argmax(axis=1).detach().cpu().numpy()))
          ground_truth = np.hstack((ground_truth,labels.detach().cpu().numpy()))
        idx+=1
        
    test_Accuracy = int(np.sum(np.equal(ground_truth,t_out))) / ground_truth.shape[0]
    
    return test_Accuracy,ground_truth,t_out


def test_Accuracypc(testloader,model):
    """
        path: path where the dictionary state file(.pth) is of the trained classifier
        testloader: testloader data
        -----
        return test accuracy, and real and predicted label
        """
    first_batch = next(iter(testloader))
    data, labels = first_batch
# Extract the images from the batch
    data_shape = tuple(data[0].shape)

# Convert each dimension to an integer
    input_size = tuple(int(dim) for dim in data_shape)[0]


    model = model(input_size).to(device)
    
    model_name = type(model).__name__
    path = os.path.join( "src", "Classification", "modelpath",f'{model_name}_trainedpc.pth')
    save_path = os.path.join( "src", "Classification","modelpath",f'{model_name}_trainedpc.pth')
    
    model.load_state_dict(torch.load(path))
    
    model.eval()
    with torch.no_grad():
      idx = 0
      for batch in testloader:
        inputs,labels=batch
        inputs=inputs.to(device)
        labels=labels.to(device)
        inputs = inputs.float()
        labels = labels.float()
        
        if idx==0:
          t_out = model(x=inputs)
          t_out = t_out.squeeze()
          t_out = (t_out>=0.5)       
          t_out = t_out.detach().cpu().numpy()
            
          ground_truth = labels.detach().cpu().numpy()
        else:
          out = model(x=inputs)
          out = out.squeeze()
          t_out = np.hstack((t_out,out.detach().cpu().numpy()))
          t_out = (t_out>=0.5)
          ground_truth = np.hstack((ground_truth,labels.detach().cpu().numpy()))
        idx+=1
        
    test_Accuracy = int(np.sum(np.equal(ground_truth,t_out))) / ground_truth.shape[0]
    
    return test_Accuracy,ground_truth,t_out





def compute_class_weight(labels):
    # Compute class frequencies
    class_counts = np.bincount(labels)
    
    # Avoid division by zero if any class is missing
    class_counts[class_counts == 0] = 1
    
    # Compute class weights
    weights = 1. / class_counts
    weights = weights / np.sum(weights)  # Normalize to sum to 1
    
    # Create a dictionary mapping class indices to weights
    class_weight = {i: weights[i] for i in range(len(class_counts))}
    return class_weight



def test_multibase(testloader,model,datatype:str):
    """
        path: path where the dictionary state file(.pth) is of the trained classifier
        testloader: testloader data
        -----
        return test accuracy, and real and predicted label
        """
    #first_batch = next(iter(testloader))
    #data, labels = first_batch
# Extract the images from the batch
    #data_shape = tuple(data[0].shape)

# Convert each dimension to an integer
   # input_size = tuple(int(dim) for dim in data_shape)[0]

    output_fn = torch.nn.Softmax(dim=1) # we instantiate the softmax activation function for the output probabilities

    model = model(nb_input=32043, nb_output=11, drop=0, noise_std=0.0).to(device)
    
    model_name = type(model).__name__
    path = os.path.join( "src", "Classification", "modelpath",f'{model_name}_{datatype}_trainedbase.pth')
    
    model.load_state_dict(torch.load(path))
    
    model.eval()
    with torch.no_grad():
      idx = 0
      for batch in testloader:
        inputs,labels=batch
        inputs=inputs.to(device)
        labels=labels.to(device)
        inputs = inputs.float()
        labels = labels.long()

        if idx==0:
          t_out = model(inputs)
          t_out = output_fn(t_out).detach().cpu().numpy()
          t_out=t_out.argmax(axis=1)
          ground_truth = labels.detach().cpu().numpy()

        else:
          out = model(inputs)
          t_out = np.hstack((t_out,output_fn(out).argmax(axis=1).detach().cpu().numpy()))
          ground_truth = np.hstack((ground_truth,labels.detach().cpu().numpy()))
        idx+=1
        
    test_Accuracy = int(np.sum(np.equal(ground_truth,t_out))) / ground_truth.shape[0]
    
    return test_Accuracy,ground_truth,t_out
