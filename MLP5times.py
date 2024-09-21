import numpy as np
from data.dataloader import dataloader
from src.Classification.models import train_bestparam


#define the hyperparameters

output_dims =[1024, 512, 256]
dropout = 0.28502096485376677
learning_rate  =0.0003215395488204129
optimizer_name = 'SGD'

#define function to train the mlp ( 5 times)


def Mlp_times():
    
    """
        Function to train the multiclass classifier 5 times , and it saves the 5 test accuracies to the local as an numpy.ndarray
    """
    accuracy = np.zeros(5)
    
    data_loader = dataloader() 
    for i in range(5):
        train_partp, val_partp, test_partp = data_loader.prep_loaderpart(valratio=0.1,standardize=True, batch_size = 50,datatype= 'patient')
        loss_trainpart,acc_trainpart,loss_validpart,acc_validpart, test_accuracy =train_bestparam(dropout,output_dims,100,train_partp, val_partp,test_partp, learning_rate, optimizer_name,'patient')
        accuracy[i] = test_accuracy
    
    np.save('results/accuracy_times.npy',accuracy)

# Launch the training
    
Mlp_times()  
