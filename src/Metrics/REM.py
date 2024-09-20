import numpy as np
import torch 
from numpy import trapz
from scipy.stats import gaussian_kde



def REM(data,eigenvectors):
    
    """ 
    Function to calculate PCA based reconstruction error metric
    
    Parameters:
    data (numpy.ndarray): The expression data level 
    eigenvectors (numpy.ndarray): The eigenvector resultant from the PCA of patient data.
    
    
    Returns:
        distances (numpy.ndarray) : The REM metric of the data
    """
    # calculate the projection of the data to the space formed by the eigen vectors
    
    data_proj = data @ eigenvectors.T @ eigenvectors
    
    # calculate difference between of the point and its projection
    diff = data-data_proj

    # Compute the squared Euclidean distances for each difference
    squared_distances = np.sum(diff ** 2, axis=1)

    # Compute the Euclidean distances
    distances = np.sqrt(squared_distances)
    
    return distances 



def EPC(real_REM,fake_REM):
    """this function to calculate the proportion of common area of the two rem distribution.
    ----
    Parameters:
        real_REM (numpy.ndarray): the calculated RE metric for real data
        fake_REM (numpy.ndarray): the calculated RE metric for generated data
    Returns:
        tensor of pairwise distances 
    """
# Create KDE objects using the same bandwidth for both datasets
    kde1 = gaussian_kde(real_REM, bw_method='scott')
    kde2 = gaussian_kde(fake_REM, bw_method='scott')

# Define a common x-axis that covers both kde1 and kde2
    x_common = np.linspace(min(real_REM.min(), fake_REM.min()), max(real_REM.max(), fake_REM.max()), 1000)

# Evaluate KDEs on the common x-axis
    y1 = kde1(x_common)
    y2 = kde2(x_common)

# Find the minimum of the two KDEs at each point (area of overlap)
    y_min = np.minimum(y1, y2)

# Calculate the area of intersection using the trapezoidal rule
    area_of_intersection = trapz(y_min, x_common)

    return area_of_intersection
    
    
