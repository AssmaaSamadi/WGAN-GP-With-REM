import torch
from torch import nn
from typing import List


class Generator1(nn.Module):
    
  """ Generator class with best parameters.
  """

  def __init__(self, dim_z, dim_output):
      """ Main function to define the architecture of the generator.
      ----
      Parameters:
            dim_z (float): dimension of noise data.
            dim_output (float): dimension of the generated data by the generator

      """
      super(Generator1, self).__init__()
      self.fc1 = nn.Linear(dim_z, 256)
      self.bn1 = nn.BatchNorm1d(256)
      self.fc2 = nn.Linear(256, 512)
      self.bn2 = nn.BatchNorm1d(512)
      self.fc3 = nn.Linear(512, 1024)
      self.bn3 = nn.BatchNorm1d(1024)
      self.fc4 = nn.Linear(1024, dim_output)
      self.leaky = nn.LeakyReLU(0.2)
        
  def forward(self, x):
      """ Funtion to define the forward pass  of the generator.
      ----
        
      Parameters:
        x (torch.tensor): the noise torch tensor of dimension dim_z
            
      Returns:
        x (torch.tensor): the generated data from the noise by the generator 
        

      """
      x = self.leaky(self.bn1(self.fc1(x)))
      x = self.leaky(self.bn2(self.fc2(x)))
      x = self.leaky(self.bn3(self.fc3(x)))
      x = self.fc4(x) 
      return x

    
class Generatoropt(nn.Module):
    """ Generator class used for optuna hyperparameters tunning
    """
    
    
    def __init__(self, trial,dim_z, BN:bool)-> None:
         """ Funtion to define the architecture of the tuned generator.
         ----
        
         Parameters:
            dim_z (float): dimension of noise data.
            BN (bool): True or false, to precise if there will be a batch normalization layer
            
         """
        
        
         super(Generatoropt, self).__init__()
    
         layers: List[nn.Module] = []
         if BN:
                layers.append(nn.Linear(dim_z, 256))
                layers.append(nn.BatchNorm1d(256))
                layers.append(nn.LeakyReLU(0.2)) 
                layers.append(nn.Linear(256, 512))
                layers.append(nn.BatchNorm1d(512))
                layers.append(nn.LeakyReLU(0.2)) 
                layers.append(nn.Linear(512, 1024))
                layers.append(nn.BatchNorm1d(1024))
                layers.append(nn.LeakyReLU(0.2)) 
                layers.append(nn.Linear(1024, 32043))
        
         else:
                layers.append(nn.Linear(dim_z, 256))
                layers.append(nn.LeakyReLU(0.2)) 
                layers.append(nn.Linear(256, 512))
                layers.append(nn.LeakyReLU(0.2)) 
                layers.append(nn.Linear(512, 1024))
                layers.append(nn.LeakyReLU(0.2)) 
                layers.append(nn.Linear(1024, 32043)) 
                
         self.layers = nn.Sequential(*layers)
    def forward(self, x) -> torch.Tensor:
         """ Funtion to define the forward pass  of the generator.
         ----
        
         Parameters:
            x (torch.tensor): the noise torch tensor of dimension dim_z
            
         Returns:
            gen (torch.tensor): the generated data from the noise by the generator 
         """
         gen = self.layers(x)
         return gen 

    
class Discriminator1(nn.Module):
    
  """ Discriminator class
  """
    
  def __init__(self, dim_x):
      """ Funtion to define the architecture of the discriminator.
      ----
        
      Parameters:
            dim_x (float): dimension of input data.            
      """
      super(Discriminator1, self).__init__()
      self.fcd1 = nn.Linear(dim_x, 512)
      self.fcd2 = nn.Linear(512,256)
      self.fcd3 = nn.Linear(256,1)
      self.leakyd = nn.LeakyReLU(0.2)
  def forward(self, x):
      """ Funtion to define the forward pass  of the discriminator.
      ----
        
      Parameters:
      x (torch.tensor): the input data that will be discriminate
            
      Returns:
      x (torch.tensor): the discriminated data 
      """
      x =self.leakyd(self.fcd1(x))
      x = self.leakyd(self.fcd2(x))
      x = self.fcd3(x)
      return x
