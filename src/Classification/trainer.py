import torch
import numpy as np
import torch.optim as optim
from torch import nn
from src.Classification.utils_classification import EarlyStopper, get_accuracy
import os
from tqdm import tqdm

device = "cuda:3" if torch.cuda.is_available() else "cpu" #0 indicates the GPU you gonna use

def train_modelopt(NNp,epoch,train_loader, val_loader, testloader,learning_rate,optimizer,datatype:str):
     
        
  """ Main function to train the multiclass classifier  
  
  Parameters:
  
      NNp(nn.Module):
      epoch (int): The number of epochs of the training
      train_loader (torch.utils.data.DataLoader): The training dataloader.
      val_loader (torch.utils.data.DataLoader): The validation data loader
      testloader (torch.utils.data.DataLoader): The testing dataloader.
      learning_rate (float): The learning rate for the optimizer.
      optimizer (torch.optim.Optimizer): The optimizer
      datatype (str): Type of the data 'cellline or patient' that we need to classify.
      
  Returns:
      loss_train (numpy.ndarray): The training losses for all epochs.
      acc_train (numpy.ndarray): The training accuracies for all epochs.
      loss_valid (numpy.ndarray): The validation losses for all epochs.
      acc_valid (numpy.ndarray): The validation accuracies for all epochs.
      test_accuracy (int): The test accuracy.
  """
  first_batch = next(iter(train_loader))
  data, labels = first_batch
# Extract the Microarray data from the batch
  data_shape = tuple(data[0].shape)

# Convert each dimension to an integer
  input_size = tuple(int(dim) for dim in data_shape)[0]    
  

  early_stopper = EarlyStopper(patience=3, min_delta=0.001)
  epochs =epoch
  output_fn = torch.nn.Softmax(dim=1) # we instantiate the softmax activation function for the output probabilities
  criterion = nn.CrossEntropyLoss()


  loss_valid,acc_valid =[],[]
  loss_train,acc_train =[],[]

  for epoch in tqdm(range(epochs)):

    # Training loop
    NNp.train() # always specify that the model is in training mode
    running_loss = 0.0 # init loss
    running_acc = 0.

    # Loop over batches returned by the data loader
    for idx, batch in enumerate(train_loader):

      # get the inputs; batch is a tuple of (inputs, labels)
      inputs, labels = batch
      inputs = inputs.to(device) # put the data on the same device as the model
      labels = labels.to(device)

      # put to zero the parameters gradients at each iteration to avoid accumulations
      optimizer.zero_grad()
      inputs = inputs.float()
      labels = labels.long()
      # forward pass + backward pass + update the model parameters
      out = NNp(inputs) # get predictions
      out = out.squeeze()
      out = out.float()
      loss = criterion(out, labels) # compute loss
      loss.backward() # compute gradients
      optimizer.step() # update model parameters according to these gradients and our optimizer strategy

      # Iteration train metrics
      running_loss += loss.view(1).item()
      t_out = output_fn(out.detach()).cpu().numpy() # compute softmax (previously instantiated) and detach predictions from the model graph
        
      t_out=t_out.argmax(axis=1) # the class with the highest energy is what we choose as prediction
      ground_truth = labels.cpu().numpy() # detach the labels from GPU device
      running_acc += get_accuracy(ground_truth, t_out)

    ### Epochs train metrics ###
    acc_train.append(running_acc/len(train_loader))
    loss_train.append(running_loss/len(train_loader))

    # compute loss and accuracy after an epoch on the train and valid set
    NNp.eval() # put the model in evaluation mode (this prevents the use of dropout layers for instance)

    ### VALIDATION DATA ###
    with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
      idx = 0
      for batch in val_loader:
        inputs,labels=batch
        inputs=inputs.to(device)
        labels=labels.to(device)
        inputs = inputs.float()
        labels = labels.long()
        if idx==0:
          t_out = NNp(inputs)
          t_loss = criterion(t_out, labels).view(1).item()
          t_out = output_fn(t_out).detach().cpu().numpy() # compute softmax (previously instantiated) and detach predictions from the model graph
          t_out=t_out.argmax(axis=1) # the class with the highest energy is what we choose as prediction
          ground_truth = labels.cpu().numpy() # detach the labels from GPU device
        else:
          out = NNp(inputs)
          t_loss = np.hstack((t_loss,criterion(out, labels).item()))
          t_out = np.hstack((t_out,out.argmax(axis=1).detach().cpu().numpy()))
          ground_truth = np.hstack((ground_truth,labels.detach().cpu().numpy()))
        idx+=1
        if early_stopper.early_stop(np.mean(t_loss)):
          print("Early stopping triggered!")
          break
      acc_valid.append(get_accuracy(ground_truth,t_out))
      loss_valid.append(np.mean(t_loss))

    #print('| Epoch: {}/{} | Train: Loss {:.4f} Accuracy : {:.4f} '\
       # '| Val: Loss {:.4f} Accuracy :{:.4f}\n'.format(epoch+1,epochs,loss_train[epoch],acc_train[epoch],loss_valid[epoch],acc_valid[epoch]))
  #model_name = type(NNp).__name__
  #save_path = os.path.join( "src", "Classification","modelpath",f'{model_name}_{datatype}_trainedalllabel2.pth')
  #torch.save(NNp.state_dict(), save_path)

  NNp.eval()
  with torch.no_grad():
    idx = 0
    for batch in testloader:
        inputs,labels=batch
        inputs=inputs.to(device)
        labels=labels.to(device)
        inputs = inputs.float()
        labels = labels.long()

        if idx==0:
          t_out = NNp(inputs)
          t_out = output_fn(t_out).detach().cpu().numpy()
          t_out=t_out.argmax(axis=1)
          ground_truth = labels.detach().cpu().numpy()

        else:
          out = NNp(inputs)
          t_out = np.hstack((t_out,output_fn(out).argmax(axis=1).detach().cpu().numpy()))
          ground_truth = np.hstack((ground_truth,labels.detach().cpu().numpy()))
        idx+=1
        
  test_Accuracy = int(np.sum(np.equal(ground_truth,t_out))) / ground_truth.shape[0]


  return loss_train, acc_train, loss_valid, acc_valid,test_Accuracy





def train_basemodel(NNp,epoch,train_loader, val_loader,test_partp,learning_rate,optimizer,class_weights,datatype:str):

    
  """ Main function to train the B.Hanczar model on the data with 32043 features and 11 labels  
  
  Parameters:
  
      NNp(nn.Module):
      epoch (int): The number of epochs of the training
      train_loader (torch.utils.data.DataLoader): The training dataloader.
      val_loader (torch.utils.data.DataLoader): The validation data loader
      test_partp (torch.utils.data.DataLoader): The testing dataloader.
      learning_rate (float): The learning rate for the optimizer.
      optimizer (torch.optim.Optimizer): The optimizer
      class_weight (torch.tensor): The weight for each label
      datatype (str): Type of the data 'cellline or patient' that we need to classify.
      
  Returns:
      loss_train (numpy.ndarray): The training losses for all epochs.
      acc_train (numpy.ndarray): The training accuracies for all epochs.
      loss_valid (numpy.ndarray): The validation losses for all epochs.
      acc_valid (numpy.ndarray): The validation accuracies for all epochs.
      test_accuracy (int): The test accuracy.
  """  
  first_batch = next(iter(train_loader))
  data, labels = first_batch
# Extract the Microarray data from the batch
  data_shape = tuple(data[0].shape)

# Convert each dimension to an integer
  input_size = tuple(int(dim) for dim in data_shape)[0]    
    
  #NNp = NN(input_size=input_size).to(device)
  early_stopper = EarlyStopper(patience=10, min_delta=0.001)
  epochs =epoch
  output_fn = torch.nn.Softmax(dim=1) # we instantiate the softmax activation function for the output probabilities
  criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
  #optimizer = optim.Adam(NNp.parameters(), lr=0.001) # we instantiate Adam optimizer that takes as inputs the model parameters and learning rate
  # Init

  loss_valid,acc_valid =[],[]
  loss_train,acc_train =[],[]

  for epoch in tqdm(range(epochs)):

    # Training loop
    NNp.train() # always specify that the model is in training mode
    running_loss = 0.0 # init loss
    running_acc = 0.

    # Loop over batches returned by the data loader
    for idx, batch in enumerate(train_loader):

      # get the inputs; batch is a tuple of (inputs, labels)
      inputs, labels = batch
      inputs = inputs.to(device) # put the data on the same device as the model
      labels = labels.to(device)

      # put to zero the parameters gradients at each iteration to avoid accumulations
      optimizer.zero_grad()
      inputs = inputs.float()
      labels = labels.long()
      # forward pass + backward pass + update the model parameters
      out = NNp(inputs) # get predictions
      out = out.squeeze()
      out = out.float()
      loss = criterion(out, labels) # compute loss
      loss.backward() # compute gradients
      optimizer.step() # update model parameters according to these gradients and our optimizer strategy

      # Iteration train metrics
      running_loss += loss.view(1).item()
      t_out = output_fn(out.detach()).cpu().numpy() # compute softmax (previously instantiated) and detach predictions from the model graph
        
      t_out=t_out.argmax(axis=1) # the class with the highest energy is what we choose as prediction
      ground_truth = labels.cpu().numpy() # detach the labels from GPU device
      running_acc += get_accuracy(ground_truth, t_out)

    ### Epochs train metrics ###
    acc_train.append(running_acc/len(train_loader))
    loss_train.append(running_loss/len(train_loader))

    # compute loss and accuracy after an epoch on the train and valid set
    NNp.eval() # put the model in evaluation mode (this prevents the use of dropout layers for instance)

    ### VALIDATION DATA ###
    with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
      idx = 0
      for batch in val_loader:
        inputs,labels=batch
        inputs=inputs.to(device)
        labels=labels.to(device)
        inputs = inputs.float()
        labels = labels.long()
        if idx==0:
          t_out = NNp(inputs)
          t_loss = criterion(t_out, labels).view(1).item()
          t_out = output_fn(t_out).detach().cpu().numpy() # compute softmax (previously instantiated) and detach predictions from the model graph
          t_out=t_out.argmax(axis=1) # the class with the highest energy is what we choose as prediction
          ground_truth = labels.cpu().numpy() # detach the labels from GPU device
        else:
          out = NNp(inputs)
          t_loss = np.hstack((t_loss,criterion(out, labels).item()))
          t_out = np.hstack((t_out,out.argmax(axis=1).detach().cpu().numpy()))
          ground_truth = np.hstack((ground_truth,labels.detach().cpu().numpy()))
        idx+=1
        if early_stopper.early_stop(np.mean(t_loss)):
          print("Early stopping triggered!")
          break
      acc_valid.append(get_accuracy(ground_truth,t_out))
      loss_valid.append(np.mean(t_loss))

    print('| Epoch: {}/{} | Train: Loss {:.4f} Accuracy : {:.4f} '\
        '| Val: Loss {:.4f} Accuracy :{:.4f}\n'.format(epoch+1,epochs,loss_train[epoch],acc_train[epoch],loss_valid[epoch],acc_valid[epoch]))
  #model_name = type(NNp).__name__
  #save_path = os.path.join( "src", "Classification","modelpath",f'{model_name}_{datatype}_trainedbase.pth')
  #torch.save(NNp.state_dict(), save_path)
  
    
  #calculate test_accuracy
  NNp.eval()
  with torch.no_grad():
    idx = 0
    for batch in test_partp:
        inputs,labels=batch
        inputs=inputs.to(device)
        labels=labels.to(device)
        inputs = inputs.float()
        labels = labels.long()

        if idx==0:
          t_out = NNp(inputs)
          t_out = output_fn(t_out).detach().cpu().numpy()
          t_out=t_out.argmax(axis=1)
          ground_truth = labels.detach().cpu().numpy()

        else:
          out = NNp(inputs)
          t_out = np.hstack((t_out,output_fn(out).argmax(axis=1).detach().cpu().numpy()))
          ground_truth = np.hstack((ground_truth,labels.detach().cpu().numpy()))
        idx+=1
        
  test_Accuracy = int(np.sum(np.equal(ground_truth,t_out))) / ground_truth.shape[0]


  return loss_train, acc_train, loss_valid, acc_valid,test_Accuracy
