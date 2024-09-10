import torch
from torch import nn
from typing import List

from data.dataloader import dataloader
from src.Classification.trainer import train_modelopt,train_basemodel
import optuna
from optuna import trial
import torch.optim as optim
import torch.nn.functional as F
device = "cuda:3" if torch.cuda.is_available() else "cpu" #0 indicates the GPU you gonna use


class modelobest(nn.Module):
    
    """ The class of our the multiclass classifier model, multilayer perceptron (MLP). For tissue classification.
    """
    
    def __init__(self,dropout: float, output_dims: List[int]) -> None:
        
        """ Initial function to define the architecture of the multilayer perceptron. That took as input the number of neurons, then it defines the number of layers.
        
        Parameters:
            dropout (float): The dropout ratio of neurons needed for each layer.
            output_dims (List[int]): List of numbers of neurons in each layer of the classifier 

        """
        
        super().__init__()
        layers: List[nn.Module] = []
        input_dim: int = 32043
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, 11))

        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """ Funtion to define the forward pass  of the MLP.
      ----
        
      Parameters:
            data (torch.Tensor): the probe expression data
            
      Returns:
        logits (torch.Tensor): the output tensor. The 16 probabilites of a sample has the 16 labels. 
        

      """
        logits = self.layers(data)
        return logits











class MLP3(nn.Module):# batch normalisation oor at all , size of hidden layer 64 is small: 1024, 512, 256  124
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1064),
            nn.BatchNorm1d(1064),  # Batch normalization layer after the first linear layer
            nn.ReLU(),
            nn.Dropout(0.5),     # Dropout layer after the ReLU activation
            nn.Linear(1064, 512),
            nn.BatchNorm1d(512),  # Batch normalization layer after the second linear layer
            nn.ReLU(),
            nn.Dropout(0.5),     # Dropout layer after the ReLU activation
            nn.Linear(512, 1), 
            nn.Sigmoid()         # Sigmoid activation for binary classification
        )
        

    def forward(self, x):
        return self.layers(x)

    
    
    
class modeloptunamulti(nn.Module):
    def __init__(self,trial,dropout: float, output_dims: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        input_dim: int = 32043
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, 16))#check number of classes

        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.layers(data)
        return logits
    
    
    
def suggest_predefined_units(trial, n_layers):
    # Define the predefined combinations of output dimensions
    predefined_combinations = {1 : [1024 ,512,256,128] ,2: [ [128, 64] , [256, 128], [512, 256],[1024, 512]],
         3: [[256, 128, 64],[512, 256, 128],[1024, 512, 256]],
                               
        4: [[256, 128, 64,32], [512, 256, 128,64],[1024, 512, 256,128]],
                               
        5:[[512, 256, 128,64,32], [1024, 512, 256,128,64]]
    }
    # Select a predefined combination index
    combination_index = trial.suggest_int("combination_index", 0, len(predefined_combinations[n_layers]) - 1)

    # Retrieve the selected combination
    chosen_combination = predefined_combinations[n_layers][combination_index]
    trial.set_user_attr("output_dims", chosen_combination)  # Store the chosen combination
    trial.params['output_dims'] = chosen_combination
    return chosen_combination

def objective(trial : optuna.trial, datatype:str):
    # We optimize the number of layers, hidden units in each layer and dropouts. and batch size
    n_layers = trial.suggest_int(name = "n_layers", low =2, high = 5)
    dropout = trial.suggest_float(name ="dropout", low=0.2, high=0.5)
    
    # define all possible numbers of neurons for each possible number of layers
    predefined_combinations = {1 : [1024 ,512,256,128] ,2: [ [128, 64] , [256, 128], [512, 256],[1024, 512]],
         3: [[256, 128, 64],[512, 256, 128],[1024, 512, 256]],
                               
        4: [[256, 128, 64,32], [512, 256, 128,64],[1024, 512, 256,128]],
                               
        5:[[512, 256, 128,64,32], [1024, 512, 256,128,64]]
    }
    #output_dims =[trial.suggest_int("n_units_l{}".format(i), 64,1024,step = 64) for i in range(n_layers)]
    
    output_dims =suggest_predefined_units(trial, n_layers)
    trial.set_user_attr("output_dims", output_dims)
    trial.params['output_dims'] = output_dims
    batchsize = trial.suggest_categorical("batch_size", [16,32,50,64]) 
    data_loader = dataloader() 
    train_part, val_part, test_part = data_loader.prep_loaderpart(valratio=0.1,datatype= datatype, batch_size = 16,standardize=True)
   
    model = modeloptunamulti(trial, dropout, output_dims).to(device)
    
    
    
    lr = trial.suggest_float("lr", 1e-4, 1e-2,log=True) # learning rate
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    loss_train, acc_train, loss_valid, acc_valid = train_modelopt(model, 30, train_part, val_part, learning_rate=lr, optimizer=optimizer)
    #print(model) #just to see if the architecture changed as we want


    return acc_valid[-1]

def train_bestparam(dropout,output_dims,epochs,train_part, val_part,test_part, learning_rate, optimizer_name,datatype):
    
    """ Helper Function to train multiclass classification model, that took the hyperparameters and feed them to the initial training function 'train_modelopt'
    
    Parameters:
        dropout (float): The dropout ratio of neurons needed for each layer.
        output_dims (numpy.ndarray): The number of neurons in each layer of the classifier 
        epochs (int): The number of epochs of the training
        train_part (torch.utils.data.DataLoader): The training dataloader.
        val_part (torch.utils.data.DataLoader): The validation data loader
        test_part (torch.utils.data.DataLoader): The testing dataloader.
        learning_rate (float): The learning rate for the optimizer.
        optimizer_name (str): The optimizer name
        datatype (str): Type of the data 'cellline or patient' that we need to classify.
    
    Returns:
    
        loss_train (numpy.ndarray): The training losses for all epochs.
        acc_train (numpy.ndarray): The training accuracies for all epochs.
        loss_valid (numpy.ndarray): The validation losses for all epochs.
        acc_valid (numpy.ndarray): The validation accuracies for all epochs.
        test_accuracy (int): The test accuracy.
    """
    
    #initiate model to device
    model = modelobest(dropout, output_dims).to(device)
    
    #initiate the optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
    
    #lunch the training
    loss_train, acc_train, loss_valid, acc_valid,test_accuracy = train_modelopt(model, epochs, train_part, val_part,test_part, learning_rate, optimizer,datatype)
    
    return loss_train, acc_train, loss_valid, acc_valid, test_accuracy

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x
    
    
class basemodel(nn.Module):
    def __init__(self, nb_input=32043, nb_output=11, drop=0, noise_std=0.0):
        super(basemodel, self).__init__()
        
        self.noise = GaussianNoise(noise_std)
        
        self.layer1 = nn.Sequential(
            nn.Linear(nb_input, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(),
            nn.Dropout(drop)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(500, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Dropout(drop)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Dropout(drop)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Dropout(drop)
        )
        
        self.output_layer = nn.Linear(50, nb_output)
        
    def forward(self, x):
        x = self.noise(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return F.softmax(x, dim=1)
    

def train_base(epochs,train_part, val_part,test_partp, learning_rate, optimizer_name,momentums,class_weights,datatype:str):
    """ Helper Function to initiate B.Hanczar model and to train it , that took the hyperparameters and feed them to the initial training function 'train_basemodel'
    
    Parameters:
        epochs (int): The number of epochs of the training
        train_part (torch.utils.data.DataLoader): The training dataloader.
        val_part (torch.utils.data.DataLoader): The validation data loader
        test_partp (torch.utils.data.DataLoader): The testing dataloader.
        learning_rate (float): The learning rate for the optimizer.
        optimizer_name (str): The optimizer name
        momentums (float): The momentum value of the optimizer
        class_weight (torch.tensor): The weight for each label
        datatype (str): Type of the data 'cellline or patient' that we need to classify.
    
    Returns:
    
        loss_train (numpy.ndarray): The training losses for all epochs.
        acc_train (numpy.ndarray): The training accuracies for all epochs.
        loss_valid (numpy.ndarray): The validation losses for all epochs.
        acc_valid (numpy.ndarray): The validation accuracies for all epochs.
        test_accuracy (int): The test accuracy.
    """
    model = basemodel(nb_input=32043, nb_output=11, drop=0, noise_std=0.0).to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate,momentum=momentums)
    loss_train, acc_train, loss_valid, acc_valid,test_accuracy= train_basemodel(model, epochs, train_part, val_part,test_partp, learning_rate, optimizer,class_weights,datatype)
    
    return loss_train, acc_train, loss_valid, acc_valid,test_accuracy


