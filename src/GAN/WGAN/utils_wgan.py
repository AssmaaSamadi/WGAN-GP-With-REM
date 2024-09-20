from src.GAN.WGAN.Model import Generator1, Discriminator1, Generatoropt
import torch
import os
import numpy as np
from data.utils_visualization import umap_embedding
import matplotlib.pyplot as plt
import seaborn as sns


def plot_recall_prec(precision,recall):

    """ 
    
    Funtion to plot precision and recall value with different k nearest value and it saves the plot in the result file.
    ----
        
    Parameters:
            precision (numpy.ndarray): The precision values for different k

            recall (numpy.ndarray): The recall values for different k

    """
    
    x = np.arange(1, 30)
    plt.figure(figsize=(10, 6))
    plt.plot(x, precision, label='Precision', marker='o')
    plt.plot(x, recall, label='Recall', marker='x')

# Adding labels and title
    plt.xlabel('K')
    plt.ylabel('Values')
    plt.title('Precision and Recall for Different K nearest neighbor value for WGAN on cancerous data ')
    plt.legend()

# Display the plot
    plt.grid(True)
    file_path = os.path.join('results/GAN', 'precision_recall_WGAN_cancerousdata.png')

# Save the plot
    plt.savefig(file_path)
    plt.show()
    
def extract_data_as_tensor(data_loader): 
    """ Funtion to extract the probe expression level from the dataloader.
      ----
        
      Parameters:
            data_loader (torch.utils.data.DataLoader): The data loader
            
      Returns:
        all_data (torch.tensor): The probe expression level data 
    """
    data_list = []

    for t,h,k in data_loader:

        data_list.append(t.cpu())

    # Concatenate all data and labels into single tensors
    all_data = torch.cat(data_list)
    idx = np.random.choice(np.arange(len(all_data)), min(len(all_data ), 2000))
    x_real = all_data[idx]
                           
    return x_real
    
    
def umap_plot1(Xreal,Xfake):
    
    """ Funtion to plot a 2D manifold approximation of real and fake data and save the plot in the results file.
      ----
        
      Parameters:
            Xreal (torch.tensor): The real data
            Xfake (torch.tensor): The fake data
            

    """
    Xrealn = Xreal.numpy()
    Xfake = Xfake.numpy()
    idx = np.random.choice(np.arange(len(Xrealn)), min(len(Xrealn), len(Xfake)), replace=False)
    x = np.concatenate((Xrealn[idx], Xfake), axis=0)
    labels1 = np.array(['real']*len(idx)+['fake']*len(idx))
    #labels2 = np.concatenate((labels_tissues[idx], labels_tissues[idx]), axis=0)
    umap_proj = umap_embedding(x)
    np.save('results/GAN/umap_best_canceroust', umap_proj)
    np.save('results/GAN/label_umap_canceroust', labels1)

    sns.set(style="darkgrid")
    ax = sns.scatterplot(x=umap_proj[:, 0],y= umap_proj[:, 1], hue=labels1,palette="muted",alpha=0.2)#transparency or opacity to check data overlapping
    plt.title('Umap for real cancerous and generated data ')
    file_path2 = os.path.join('results/GAN', 'UMAP_WGAN_cancerous.png')
# Save the pl
    plt.savefig(file_path2)
    #plt.show()
    
