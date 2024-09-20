import numpy as np
import torch
from data.dataloader import dataloader
from src.GAN.WGAN.Trainer import WGAN
from src.GAN.WGAN.utils_wgan import plot_recall_prec,umap_plot1,extract_data_as_tensor
from src.Metrics.precision_recall import knn_precision_recall_features

# initiate the dataloader class
data_loader = dataloader() 

# load the train val and test loader 
train_loader, val_loader, test_loader = data_loader.prep_loadergan2(datatype= 'patient',valratio=0.2,batch_size = 64 ,standardize=True)


WGAN = WGAN(critic_rate=0.00005,gen_rate=0.001)

# train the WGAN-GP
x_real, x_gen = WGAN.main_train_wgp(500,train_loader,val_loader,5,10)

#save the generated data and the trained loader used ib this training

torch.save(train_loader,'results/GAN/trainloadcancerousdata.pt')
torch.save(x_gen,'results/GAN/x_gencancerousdata.pt')

x_real2 = extract_data_as_tensor(train_loader)

torch.save(x_real2,'results/GAN/x_realcancerousdata.pt')

nhood_sizes = [i for i in range(1, 30)]
combined_metrics = {'precision': [], 'recall': []}
for size in nhood_sizes:
    metrics = knn_precision_recall_features(x_real2, x_gen, nhood_sizes=[size], row_batch_size=1000, col_batch_size=54675, num_gpus=1)
    combined_metrics['precision'].append(metrics['precision'][0])
    combined_metrics['recall'].append(metrics['recall'][0])

# Convert lists to arrays
combined_metrics['precision'] = np.array(combined_metrics['precision'])
combined_metrics['recall'] = np.array(combined_metrics['recall'])

# plot precision and recall with different value of k, this will be saved in the result file
plot_recall_prec(combined_metrics['precision'],combined_metrics['recall'])

# plot 2D manifold approximation for real and generteed data, this will be saved in the result file
umap_plot1(x_real2,x_gen)

