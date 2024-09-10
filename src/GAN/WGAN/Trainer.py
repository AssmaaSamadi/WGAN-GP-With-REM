from src.GAN.WGAN.Model import Generator1, Discriminator1, Generatoropt
import torch
import csv
import torch.autograd as autograd
from torch import autocast
from data.dataloader import dataloader,TensorDataset
from src.Metrics.precision_recall import knn_precision_recall_features
import time 
import random
from tqdm import tqdm 
import numpy as np
import optuna
import torch.optim as optim
device = "cuda:1"


class WGAN():
    
    """  Class of Wasserstein GAN with Gradient Penalty (WGAN-GP).
    """
    
    def __init__(self, critic_rate:float ,gen_rate:float):
        
        """
            initial function to initiate the generator and discriminator and set the training parameters.
        ----
        Parameters:
           critic_rate (float): the learning rate of the optimizer of the discriminator
           gen_rate (float): the learning rate of the optimizer of the generator
        """
        self.device = "cuda:1" #if torch.cuda.is_available() else "cpu"
        self.x_dim = 32043
        self.z_dim = 128
        self.G1 = Generator1(dim_z = self.z_dim, dim_output = self.x_dim).to(self.device)
        self.D1 = Discriminator1(dim_x = self.x_dim).to(self.device)
        self.lr2 = gen_rate
        self.lr1 = critic_rate
        self.G_optimizer1 = torch.optim.Adam(self.G1.parameters(), lr = self.lr2)
        self.D_optimizer1 = torch.optim.Adam(self.D1.parameters(), lr = self.lr1)
        self.scalerG = torch.cuda.amp.GradScaler()
        self.scalerD = torch.cuda.amp.GradScaler()
        
    def gradient_penalty(self,real_data:torch.tensor,fake_data:torch.tensor):
        
            """ 
                Function that compute the gradient penalty of the Wasserstein GAN GP.
            ----
            Parameters:
                real_data (torch.tensor): real data
                fake_data (torch.tensor): generated data by the generator
            Returns:
            gp (torch.float32): gradient penalty i.e mean squared gradient norm on interpolations (||Grad[D(interpolated)]||2-1)^2
            """
   
        # Fixed batch size
            BATCH_SIZE = real_data.size()[0]

        
        # Sample alpha from uniform distribution alpha its epsilon in the code
            alpha = torch.rand(BATCH_SIZE, 1, 1, requires_grad=True, device=real_data.device)

   
        # Interpolation between real data and fake data.
            interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
  

        # Get outputs from critic
            disc_outputs = self.D1(interpolated)
            grad_outputs = torch.ones_like(
                disc_outputs,
                requires_grad=False,
                device=real_data.device)

        # Retrieve gradients
            gradients = autograd.grad(
                outputs=disc_outputs,
                inputs=interpolated,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True)[0]

        # Compute gradient penalty
            gradients = gradients.view(BATCH_SIZE, -1)
            grad_norm = gradients.norm(2, dim=1)

            return torch.mean((grad_norm - 1) ** 2)
    
    
    def train_criticgp(self,real_data,noisedata,lambdaa):
        
        """ 
            Function that define the training way of the discriminator.
            
        ----
        Parameters:
                real_data (torch.tensor): real data
                noisedata (torch.tensor): noise data of dimension dim_z
                lambdaa (integer) :  The Penalty Coeficient.
        Returns:
            gp (torch.float32): The gradien penalty obtained after training the discriminator
            disc_loss (torch.float32): The discriminator loss obtained after training the discriminator
        """
    #train critic 
        self.D_optimizer1.zero_grad()
    
        for p in self.D1.parameters():
                p.requires_grad = True
    #only train critic        
        for p in self.G1.parameters():
                p.requires_grad = False

                
        with autocast(device_type='cuda', dtype=torch.float16):        
                
            x_real = real_data.view(-1, self.x_dim).to(self.device)        
            gen_outputs = self.G1(noisedata).detach()
            critic_outputs = self.D1(gen_outputs)
            x_real = x_real.float()
            critic_real = self.D1(x_real)
     # d loss gp in the autocast
            d_loss= torch.mean(critic_real)
            gp =self.gradient_penalty(x_real,gen_outputs)
    # Adversarial loss
            disc_loss = (-d_loss
                + torch.mean(critic_outputs)
                + lambdaa* gp )


        

        self.scalerD.scale(disc_loss).backward()
        
        self.scalerD.step(self.D_optimizer1)
        self.scalerD.update()
        
        return disc_loss, gp
    
    def train_generatorgp(self,noisedata):
        
        """ 
            Function that define the training way of the generator.
        ----
        Parameters:
            noisedata (torch.tensor): noise data of dimension dim_z
        Returns:
            gen_loss (torch.float32): The discriminator loss obtained after training the generator
        """
        self.G1.train()  # Train mode

    # Reset gradients back to 0
        self.G_optimizer1.zero_grad()

    # We train only the generator
        for p in self.G1.parameters():
            p.requires_grad = True

    # Avoid gradient computations on the critic
        for p in self.D1.parameters():
            p.requires_grad = False
            
    # autocast to approximate and reduce training time:        
        with autocast(device_type='cuda', dtype=torch.float16):    
                

            gen_loss = -self.D1(self.G1(noisedata)).mean()

    # Backpropagate 
        self.scalerG.scale(gen_loss).backward()

    # update parameters
        self.scalerG.step(self.G_optimizer1)
        self.scalerG.update()

        return gen_loss
    
    def real_fake_data(self,ValDataLoader,dim_z):
        
        """ 
            Function to generate data after finalizing the training of the Wasserstein GAN GP.
        ----
        Parameters:
            dim_z (torch.tensor): the dimension of the noise data.
            ValDataLoader (torch.utils.data.DataLoader): the validation data .
        Returns:
            x_real (numpy.ndarray: the validation data used.
            x_gen (numpy.ndarray): the generated data by the trained WGAN-GP
        """
        x_gen = []

        x_real = []

        self.G1.eval()  # Evaluation mode

        with torch.no_grad():
            for batch,cancerlabel,partlabel in ValDataLoader :
        # To GPU, else CPU
                batch = batch.to(self.device)
                batch_size = len(batch)
        # Get random latent variables z
                batch_z = torch.randn(batch_size, dim_z).to(self.device)
                
        # Generator forward pass with concatenated variables
                gen_outputs = self.G1(batch_z)
                
                x_gen.append(gen_outputs.cpu())
                x_real.append(batch.cpu())


        # Concatenate and to array
        x_gen = torch.cat(x_gen, 0).detach().numpy()
        
        # Loader returns tensors (but on CPU directly)
        x_real = torch.cat(x_real, 0).detach().numpy()

        return x_real, x_gen

    
    def main_train_wgp(self,epochs,train_loader,ValDataLoader,iters_critic,lambdaa):
        
        """ 
            The main function to train the Wasserstein GAN with gradient penalty combining the training of the generator and the discriminator.
        ----
        Parameters:
        
            epochs (integer): The number of epochs of the training.
            train_loader (torch.utils.data.DataLoader): the training data loader.
            ValDataLoader (torch.utils.data.DataLoader): the validation data .
            iters_critic (integer):The number of training times of the discriminator
            lambdaa (integer): The Penalty Coeficient
            
        Returns:
        
            x_real (torch.tensor): the validation data used.
            x_gen (torch.tensor): the generated data by the trained WGAN-GP
        """
        # save losses in a csv file
        with open('results/GAN/best_gan_cancerous.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Epoch', 'Discriminator Loss', 'Generator Loss', 'Gradient Penalty'])

            display_step = 5

        #start training process
            for epoch in tqdm(np.arange(0,epochs)):

              G_loss_batch = 0
              Di_loss_batch = 0
              gpp_batch = 0
              for batch,cancerlabel,partlabel in train_loader:
                  img_batch= batch.to(self.device)#real data
                
                  disc_loss_iters = []
                  gp_iter = [] # csv log update it checkpoint

                  batch_z = torch.randn(img_batch.shape[0], self.z_dim,device = self.device)# 

                
                  for iter in range(iters_critic):
 
        # Train critic and return loss
                    disc_loss, gp_l = self.train_criticgp(
                                img_batch,
                                batch_z,lambdaa)
                    gp_iter.append(gp_l.item())
                    disc_loss_iters.append(disc_loss.detach().item())
                      
                  Di_loss_batch+=disc_loss_iters[-1]
                  gpp_batch+=gp_iter[-1]

                  gen_loss = self.train_generatorgp(batch_z)
                  
                  G_loss_batch+=gen_loss.detach().item()

              gpp_avg = gpp_batch/len(train_loader)
              di_loss_avg = Di_loss_batch/len(train_loader)
              g_loss_avg = G_loss_batch/len(train_loader)

              csvwriter.writerow([epoch, di_loss_avg, g_loss_avg, gpp_avg])
              if (epoch % display_step == 0 ) or epoch == 0 or (epoch==epochs-1):
        # Validation
                x_real, x_gen =self.real_fake_data(ValDataLoader,self.z_dim)
            g_path= 'src/GAN/WGAN/model_paths/generatorcancerous.pth'
            d_path = 'src/GAN/WGAN/model_paths/discriminatorcancerous.pth'
            torch.save(self.G1.state_dict(), g_path) 
            torch.save(self.D1.state_dict(), d_path) 
            return   torch.tensor(x_real), torch.tensor(x_gen)

        
def objective(trial):
    
    """ 
        Objective function to do the hyperparameters tunning of Wassertein Gan
    ---
    Returns:
            f1: the f1 score between the precision and recall metric of each trial
    """

    
    batchsize = trial.suggest_categorical("batch_size", [64,128,256]) 
    
    data_loader = dataloader() 
    train_loader, val_loader, test_loader = data_loader.prep_loadergan2(datatype= 'patient',valratio=0.2,batch_size = batchsize ,standardize=True)
    valuebn = trial.suggest_categorical('BN', [True, False])
    Generator2 = Generatoropt(trial, dim_z = 128, BN = valuebn).to(device)
    D = Discriminator1(dim_x = 32043).to(device)
    
    lrg = trial.suggest_categorical("lrg",[0.01,0.001,0.0001,0.0002,0.00005])
    lrd = trial.suggest_categorical("lrd",[0.01,0.001,0.0001,0.0002,0.00005])# learning rate
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    
    optimizerg = getattr(optim, optimizer_name)(Generator2.parameters(), lr=lrg)
    optimizerd = getattr(optim, optimizer_name)(D.parameters(), lr=lrd)
    WGAN2 = WGANopt(optimizerg ,optimizerd,Generator2,D)
    x_real, x_gen = WGAN2.main_train_wgpopt(100,train_loader,val_loader,5,10)
    #print(model) #just to see if the architecture changed as we want
    metrics = knn_precision_recall_features(x_real,x_gen,
        nhood_sizes=[30],
        row_batch_size=1000,
        col_batch_size=32043,
        num_gpus=1)
    f1 = (metrics['precision'] * metrics['recall'])/(metrics['precision'] + metrics['recall'])

    return f1


# class WGAN for hyperparameters tunning Optuna

class WGANopt():
    
    """  
        Class of Wasserstein GAN with Gradient Penalty (WGAN-GP) used by optuna hyperparameters optimization framework.
        
    """
    
    def __init__(self, optimizerg ,optimizerd,Generator1,D):
        """
            Initial function to initiate the generator and discriminator defined by the optimization framework.
        ----
        Parameters:
        
           optimizerg (torch.optim.Optimizer): the generator optimizer
           optimizerd (torch.optim.Optimizer): the discriminator optimizer
           Generator1 (torch.nn.Module): The generator
           D (torch.nn.Module) : The discriminator
           
        """
        self.device = "cuda:3" #if torch.cuda.is_available() else "cpu"
        self.x_dim = 32043
        self.z_dim = 128
        self.G1 = Generator1
        self.D1 = D
        self.G_optimizer1 = optimizerg
        self.D_optimizer1 = optimizerd
        self.scalerG = torch.cuda.amp.GradScaler()
        self.scalerD = torch.cuda.amp.GradScaler()
    def gradient_penalty(self,real_data:torch.tensor,fake_data:torch.tensor):
            """
                Compute gradient penalty.
            ----
            
            Parameters:
            
                real_data (torch.tensor): real data
                fake_data (torch.tensor): generated data
                
            Returns:
            
                gp (torch.tensor): gradient penalty i.e mean squared gradient norm on interpolations (||Grad[D(x_inter)]||2-1)^2
                
            """
   
        # Fixed batch size
            BATCH_SIZE = real_data.size()[0]

        
        # Sample alpha from uniform distribution alpha its epsilon in the code
            alpha = torch.rand(BATCH_SIZE, 1, 1, requires_grad=True, device=real_data.device)

   
        # Interpolation between real data and fake data.
            interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
  
        # Generator forward pass 

        # Get outputs from critic
            disc_outputs = self.D1(interpolated)
            grad_outputs = torch.ones_like(
                disc_outputs,
                requires_grad=False,
                device=real_data.device)

        # Retrieve gradients
            gradients = autograd.grad(
                outputs=disc_outputs,
                inputs=interpolated,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True)[0]

        # Compute gradient penalty
            gradients = gradients.view(BATCH_SIZE, -1)
            grad_norm = gradients.norm(2, dim=1)

            return torch.mean((grad_norm - 1) ** 2)
    
    
    def train_criticgp(self,real_data,noisedata,lambdaa):
        
        """ 
            Function that define the training way of the discriminator.
        ----
        Parameters:
        
                real_data (torch.tensor): real data
                noisedata (torch.tensor): noise data of dimension dim_z
                lambdaa (integer) :  The Penalty Coeficient.
        Returns:
        
            gp (torch.float32): The gradien penalty obtained after training the discriminator
            disc_loss (torch.float32): The discriminator loss obtained after training the discriminator
            
        """
        #train critic 
        self.D_optimizer1.zero_grad()
    
        for p in self.D1.parameters():
                p.requires_grad = True
        #make sure that we only train critic        
        for p in self.G1.parameters():
                p.requires_grad = False

                
        with autocast(device_type='cuda', dtype=torch.float16):        
                
            x_real = real_data.view(-1, self.x_dim).to(self.device)        
            gen_outputs = self.G1(noisedata).detach()
            critic_outputs = self.D1(gen_outputs)
            x_real = x_real.float()
            critic_real = self.D1(x_real)
        # d loss gp in the autocast
            d_loss= torch.mean(critic_real)
            gp =self.gradient_penalty(x_real,gen_outputs)
        # Adversarial loss
            disc_loss = (-d_loss
                + torch.mean(critic_outputs)
                + lambdaa* gp )


        self.scalerD.scale(disc_loss).backward()
        
        self.scalerD.step(self.D_optimizer1)
        self.scalerD.update()
        # reconvert loss to float 32 with the scaler
        
        return disc_loss, gp
    
    def train_generatorgp(self,noisedata):
        
        """ 
            Function that define the training way of the generator.
        ----
        
        Parameters:
        
            noisedata (torch.tensor): noise data of dimension dim_z
            
        Returns:
            gen_loss (torch.float32): The discriminator loss obtained after training the generator
        """
        self.G1.train()  # Train mode

        # Reset gradients back to 0
        self.G_optimizer1.zero_grad()

        # We train only the generator
        for p in self.G1.parameters():
            p.requires_grad = True

        # Avoid gradient computations on the critic
        for p in self.D1.parameters():
            p.requires_grad = False
        # autocast to do approximation in order to minimise training time    
        with autocast(device_type='cuda', dtype=torch.float16):    
                    

            # # Compute losses
            gen_loss = -self.D1(self.G1(noisedata)).mean()
       

        # Backpropagate
        self.scalerG.scale(gen_loss).backward()

        # Update parameters
        self.scalerG.step(self.G_optimizer1)
        self.scalerG.update()

        return gen_loss
    
    def real_fake_data(self,ValDataLoader,dim_z):
        
        """ 
            Function to generate data after finalizing the training of the Wasserstein GAN GP.
        ----
        Parameters:
        
            dim_z (torch.tensor): the dimension of the noise data.
            ValDataLoader (torch.utils.data.DataLoader): the validation data.
            
        Returns:
        
            x_real (numpy.ndarray: the validation data used.
            x_gen (numpy.ndarray): the generated data by the trained WGAN-GP
            
        """

        x_gen = []
        x_real = []

        self.G1.eval()  # Evaluation mode

        with torch.no_grad():
            for batch,cancerlabel,partlabel in ValDataLoader :
                # To GPU, else CPU
                batch = batch.to(self.device)
                batch_size = len(batch)
                # Get random latent variables z
                batch_z = torch.randn(batch_size, dim_z).to(self.device)
                
                # Generator forward pass with concatenated variables
                gen_outputs = self.G1(batch_z)
                
                x_gen.append(gen_outputs.cpu())
                x_real.append(batch.cpu())


        # Concatenate and to array
        x_gen = torch.cat(x_gen, 0).detach().numpy()
        # Loader returns tensors (but on CPU directly)
        x_real = torch.cat(x_real, 0).detach().numpy()

        return x_real, x_gen
        
    
    
    def main_train_wgpopt(self,epochs,train_loader,ValDataLoader,iters_critic,lambdaa):
        
            """ 
                The main function to train the Wasserstein GAN with gradient penalty combining the training of the generator and the discriminator with optuna parameters.
            ----
            Parameters:
            
                epochs (integer): The number of epochs of the training.
                train_loader (torch.utils.data.DataLoader): the training data loader.
                ValDataLoader (torch.utils.data.DataLoader): the validation data .
                iters_critic (integer):The number of training times of the discriminator
                lambdaa (integer): The Penalty Coeficient
                
            Returns:
            
                x_real (torch.tensor): the validation data used.
                x_gen (torch.tensor): the generated data by the trained WGAN-GP
                
            """
            display_step = 5
            for epoch in tqdm(np.arange(0,epochs)):

              for batch,cancerlabel,partlabel in train_loader:
                  img_batch= batch.to(self.device)#real data
                  # Get random latent variables batch_z (noise data)
                  batch_z = torch.randn(img_batch.shape[0], self.z_dim,device = self.device)

                  for iter in range(iters_critic):
                    
            
                         
        # Train critic and return loss
                    disc_loss, gp_l = self.train_criticgp(
                                img_batch,
                                batch_z,lambdaa)
 
                  gen_loss = self.train_generatorgp(batch_z)
                  
              
              if (epoch % display_step == 0 ) or epoch == 0 or (epoch==epochs-1):
                    
        # generate data now in the validation mode
                x_real, x_gen =self.real_fake_data(ValDataLoader,self.z_dim)
            
            return   torch.tensor(x_real), torch.tensor(x_gen)
        
