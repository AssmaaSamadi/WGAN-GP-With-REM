import torch
from data.utils_preprocessing import train_test_val, standardized, map_labels_to_indices,map_type_to_indices, df_to_label
from data.preprocessing import typepreprocee
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle


from src.Classification.utils_classification import compute_class_weight

class dataloader():
    
    """  Class of dataloader to load data for the classification tasks and the WGAN-GP trainings.
    """
    
    
    def __init__(self):
       
        self.refer_columns = ['Characteristics.disease.','Characteristics.organism.part.']
        self.column = 'part'
        self.path = "/home/commun/MicroArray/E-MTAB-3732/class.csv"
        #self.path_ids ='Microarrays_files/Microaarays'
        self.patharr = "/home/commun/MicroArray/E-MTAB-3732/E-MTAB-3732.data2.csv"
        self.file_path = 'data/Microarrays_files'
        
    def processing_rna_data(self):
        """
        Processing probe expression data of Microarray data
        """
        Processor = typepreprocee()
        Processor.preprocessing()# adjust it to save it as parquet
    # add function to load data as numpy from the parquet saved file  
    
    def prep_loader(self, datatype:str , valratio:float,batch_size:int,standardize:bool=True):
        
        """
        Function to prepare data for binary classification of cancer phenotype (cancer, normal)
        
        Parameters:
            datatype (str): To precise which data to classify, patient or cell line.
            valratio (float): The ratio to split data in trainning and validation set.
            batch_size (int): The size to split data by batches in the dataloader
            standardize (bool): True to standard scaling the data, False to not.
        
        Returns:
            
            trainloader (torch.utils.data.DataLoader): The training data loader that will be used in training of the mlp
            valloader (torch.utils.data.DataLoader): The validation data loader 
            testloader (torch.utils.data.DataLoader): The testloader data loader
        
        
        """
        
        path_train = f"{self.file_path}/{datatype}_data/{datatype}data_train2"
        path_test = f"{self.file_path}/{datatype}_data/{datatype}data_test2"
        data = pd.read_parquet(path_train)
        data = data[data['Part']!='abdomen']
        datatest = pd.read_parquet(path_test)
        datatest = datatest[datatest['Part']!='abdomen']
        labeltest= datatest['Cancer']
        label = data['Cancer']
        data.drop('Cancer', axis=1, inplace=True)
        data.drop('Part', axis=1, inplace=True)
        datatest.drop('Cancer', axis=1, inplace=True)
        datatest.drop('Part', axis=1, inplace=True)
        #path_data = f"{self.file_path}/{datatype}data.npy"
        X_train,y_train,X_val,y_val = train_test_val(label,data,valratio)
        #standardize Microarray data
        
        if standardize:
            X_train = standardized(X_train)
            X_val = standardized(X_val)
            X_test = standardized(datatest)
        else:
             X_train = X_train.to_numpy()
             X_val = X_val.to_numpy()
             X_test = datatest.to_numpy() 
        #enocode label
        y_train = map_labels_to_indices(y_train)
        y_test = map_labels_to_indices(labeltest)
        y_val = map_labels_to_indices(y_val)
        X_train =torch.from_numpy(X_train)
        X_test =torch.from_numpy(X_test)
        X_val =torch.from_numpy(X_val)
        
        
        trainloader = torch.utils.data.DataLoader(TensorDataset(X_train,y_train), batch_size=batch_size, shuffle=True,        num_workers=1)#,drop_last=True)
        valloader = torch.utils.data.DataLoader(TensorDataset(X_val,y_val), batch_size=batch_size, shuffle=True, num_workers=1)#,drop_last=True)2 or 6 or 8 (0 it slows my training) I tied 10,15 is the same as 1)
        testloader = torch.utils.data.DataLoader(TensorDataset(X_test,y_test), batch_size=batch_size, shuffle=True, num_workers=1)#,drop_last=True)
        
        return trainloader,valloader,testloader
    
    def prep_loaderpartbaseline(self, valratio: int, datatype: str, batch_size: int, standardize: bool = True):
        
        """
        Function to prepare data for multiclass classification  as in the article of B.Hanczar and V. Bourgeais: Transfer learning.. In this some tissue types will be dropped.
        
        Parameters:
        
            datatype (str): To precise which data to classify, patient or cell line.
            valratio (float): The ratio to split data in trainning and validation set.
            batch_size (int): The size to split data by batches in the dataloader
            standardize (bool): True to standard scaling the data, False to not.
        
        Returns:
            
            trainloader (torch.utils.data.DataLoader): The training data loader that will be used in training of the mlp
            valloader (torch.utils.data.DataLoader): The validation data loader 
            testloader (torch.utils.data.DataLoader): The testloader data loader
        
        
        """
        #load saved data from the local
        path_train = f"{self.file_path}/{datatype}_data/{datatype}data_train2"
        path_test = f"{self.file_path}/{datatype}_data/{datatype}data_test2"
        data = pd.read_parquet(path_train)
        
        # drop less represented tissue types as in the article
        data = data[~data['Part'].isin(['abdomen', 'adrenal', 'pancreas','uterus','stomach','lymph node'])]
        datatest = pd.read_parquet(path_test)
        datatest = datatest[~datatest['Part'].isin(['abdomen', 'adrenal', 'pancreas','uterus','stomach','lymph node'])]
        #get labels
        labeltest= datatest['Part']
        label = data['Part']
        data.drop('Cancer', axis=1, inplace=True)
        data.drop('Part', axis=1, inplace=True)
        datatest.drop('Cancer', axis=1, inplace=True)
        datatest.drop('Part', axis=1, inplace=True)

        #split data between training and validation
        X_train,y_train,X_val,y_val = train_test_val(label,data,valratio)
    
        #standardize data if needed
        if standardize:
            X_train = standardized(X_train)
            X_val = standardized(X_val)
            X_test = standardized(datatest)
            
        #with open('data/Microarrays_files/patientpartlabelmulti', 'rb') as f:
            #LABELS = pickle.load(f)
        #enocode label
        encoder = LabelEncoder()
        
        # Fit and encode the label
        X_encoded = encoder.fit(y_train)
        classes = encoder.classes_
        y_train = encoder.transform(y_train)

        y_trainpart =torch.tensor(y_train)
        y_testpart = torch.tensor(encoder.transform(labeltest))
        y_valpart = torch.tensor(encoder.transform(y_val))
        X_train =torch.from_numpy(X_train)
        X_test =torch.from_numpy(X_test)
        X_val =torch.from_numpy(X_val)
        
        #compute weight as it proposed in the article
        weight = compute_class_weight(y_train)
        class_weights_tensor = torch.tensor(list(weight.values()), dtype=torch.float32)
        
        # get the loaders
        trainloader = torch.utils.data.DataLoader(TensorDataset(X_train,y_trainpart), batch_size, shuffle=True,num_workers=1,drop_last=True)
        valloader = torch.utils.data.DataLoader(TensorDataset(X_val,y_valpart), batch_size, shuffle=True, num_workers=1,drop_last=True)
        testloader = torch.utils.data.DataLoader(TensorDataset(X_test,y_testpart), batch_size, shuffle=True, num_workers=1,drop_last=True)
        
        
        pathname = f"{self.file_path}/{datatype}partlabelmultibas"
        with open(pathname, 'wb') as f:
            pickle.dump(classes, f)
        
        
        return trainloader,valloader,testloader,class_weights_tensor
    
    
    def prep_loaderpart(self, valratio: int, datatype: str, batch_size: int, standardize: bool = True):
        
                
        """ 
        Function to split the training data and prepare the training, validation and testing data loader for the training of the tissue type multi class classification. Where data loaders include tissue type.
        ----
        Parameters:
        
            valratio (float): The split ratio of the data between train and validation.
            datatype (str): To precise which data to prepare patient or cellline data.
            batch_size (int): The size of batches in the data loader.
            standardize (bool):It precise if the data will be standardized or no.
            
        Returns:
            
            trainloader (torch.utils.data.DataLoader): The training data loader that will be used in training the WGAN-GP.
            valloader (torch.utils.data.DataLoader): The validation data loader.
            testloader (torch.utils.data.DataLoader): The testloader data loader.

        """
        
        path_train = f"{self.file_path}/{datatype}_data/{datatype}data_train2"
        path_test = f"{self.file_path}/{datatype}_data/{datatype}data_test2"
        data = pd.read_parquet(path_train)
        data = data[data['Part']!='abdomen']
        datatest = pd.read_parquet(path_test)
        datatest = datatest[datatest['Part']!='abdomen']
        labeltest= datatest['Part']
        label = data['Part']
        data.drop('Cancer', axis=1, inplace=True)
        data.drop('Part', axis=1, inplace=True)
        datatest.drop('Cancer', axis=1, inplace=True)
        datatest.drop('Part', axis=1, inplace=True)
        X_train,y_train,X_val,y_val = train_test_val(label,data,valratio)
    
    
        if standardize:
            X_train = standardized(X_train)
            X_val = standardized(X_val)
            X_test = standardized(datatest)
        with open('data/Microarrays_files/patientpartlabelmulti', 'rb') as f:
            LABELS = pickle.load(f)
        #enocode label
        encoder = LabelEncoder()
        
        # Fit and transform the data
        X_encoded = encoder.fit(LABELS)
        classes = encoder.classes_
        y_trainpart =torch.tensor(encoder.transform(y_train))
        y_testpart = torch.tensor(encoder.transform(labeltest))
        y_valpart = torch.tensor(encoder.transform(y_val))
        X_train =torch.from_numpy(X_train)
        X_test =torch.from_numpy(X_test)
        X_val =torch.from_numpy(X_val)
        
        
        trainloader = torch.utils.data.DataLoader(TensorDataset(X_train,y_trainpart), batch_size, shuffle=True,num_workers=1,drop_last=True)
        valloader = torch.utils.data.DataLoader(TensorDataset(X_val,y_valpart), batch_size, shuffle=True, num_workers=1,drop_last=True)
        testloader = torch.utils.data.DataLoader(TensorDataset(X_test,y_testpart), batch_size, shuffle=True, num_workers=1,drop_last=True)
        
        
        ##pathname = f"{self.file_path}/{datatype}partlabelmulti"
       # with open(pathname, 'wb') as f:
           # pickle.dump(classes, f)
        
        
        return trainloader,valloader,testloader
    
    
    
    def prep_loadergan2(self, valratio: int, datatype: str, batch_size:int, standardize: bool = True):
        
        """ 
        Function to split the training data and prepare the training, validation and testing data loader for the training of the WGAN-GP. Where data loaders include tissue type and cancer phenotype.
        
        Parameters:
        
            valratio (float): The split ratio of the data between train and validation.
            datatype (str): To precise which data to prepare patient or cellline data.
            batch_size (int): The size of batches in the data loader.
            standardize (bool):It precise if the data will be standardized or no.
            
        Returns:
            
            trainloader (torch.utils.data.DataLoader): The training data loader that will be used in training the WGAN-GP
            valloader (torch.utils.data.DataLoader): The validation data loader 
            testloader (torch.utils.data.DataLoader): The testloader data loader

        """

        
        path_train = f"{self.file_path}/{datatype}_data/{datatype}data_train2"
        path_test = f"{self.file_path}/{datatype}_data/{datatype}data_test2"
        data = pd.read_parquet(path_train)
        datatest = pd.read_parquet(path_test)
        # drop abdomen since in cell line there is no abdomen tissue type
        data = data[data['Part']!='abdomen']
        datatest = datatest[datatest['Part']!='abdomen']
        X_train,X_val = train_test_split(data, test_size=valratio, shuffle=True,stratify = data['Part'])
        X_train,y_c_train,y_p_train=df_to_label(X_train)
        X_val,y_c_val,y_p_val=df_to_label(X_val)
        X_test,y_c_test,y_p_test=df_to_label(datatest)
        if standardize:
            X_train = standardized(X_train)
            X_val = standardized(X_val)
            X_test = standardized(X_test)
        
        
        
        else:
             X_train = X_train.to_numpy()
             X_val = X_val.to_numpy()
             X_test = datatest.to_numpy() 
                
        #enocode cancer label
        y_ctrain = map_labels_to_indices(y_c_train)
        y_ctest = map_labels_to_indices(y_c_test)
        y_cval = map_labels_to_indices(y_c_val)
        with open('data/Microarrays_files/patientpartlabelmulti', 'rb') as f:
            LABELS = pickle.load(f)
        
        #enocode part label
        encoder = LabelEncoder()
        
        # Fit and transform the data
        X_encoded = encoder.fit(LABELS)
        classes = encoder.classes_
        y_trainpart =torch.tensor(encoder.transform(y_p_train))
        y_testpart = torch.tensor(encoder.transform(y_p_test))
        y_valpart = torch.tensor(encoder.transform(y_p_val))
        
        
        X_train =torch.from_numpy(X_train)
        X_test =torch.from_numpy(X_test)
        X_val =torch.from_numpy(X_val)
        

        trainloader = torch.utils.data.DataLoader(TensorDataset(X_train,y_ctrain,y_trainpart), batch_size, shuffle=True, num_workers=2, pin_memory=True,drop_last=True)
        valloader = torch.utils.data.DataLoader(TensorDataset(X_val,y_cval,y_valpart), batch_size, shuffle=True, num_workers=2,pin_memory=True,drop_last=True)
        testloader = torch.utils.data.DataLoader(TensorDataset(X_test,y_ctest,y_testpart), batch_size, shuffle=True, num_workers=2,pin_memory=True,drop_last=True)
        
        
        pathname = f"{self.file_path}/{datatype}partlabel"
        with open(pathname, 'wb') as f:
           pickle.dump(classes, f)
        
        
        return trainloader,valloader,testloader
        
        
    def prep_loader2(self, batch_size:int,standardize:bool=True):
        """
        Function to prepare data for binary classification patient and cell line tissue
        
        Parameters:
            batch_size (int): The size to split data by batches in the dataloader
            standardize (bool): True to standard scaling the data, False to not.
        
        Returns:
            
            trainloader (torch.utils.data.DataLoader): The training data loader that will be used in training of the mlp
            valloader (torch.utils.data.DataLoader): The validation data loader 
            testloader (torch.utils.data.DataLoader): The testloader data loader
        
        
        """
        pathc_train='data/Microarrays_files/cell_data/celldata_train2'
        pathc_test='data/Microarrays_files/cell_data/celldata_test2'
        pathp_test='data/Microarrays_files/patient_data/patientdata_test2'
        pathp_train='data/Microarrays_files/patient_data/patientdata_train2'
        datatc = pd.read_parquet(pathc_train)
        datatestc = pd.read_parquet(pathc_test)
        datatp = pd.read_parquet(pathp_train)
        datatestp = pd.read_parquet(pathp_test)        
        datatp = datatp[datatp['Part']!='abdomen']
        datatestp = datatestp[datatestp['Part']!='abdomen']
        datatp['type'] = 'patient'
        datatestp['type'] = 'patient'
        datatestc['type'] = 'cell line'
        datatc['type'] = 'cell line'
        combined_train = pd.concat([datatp, datatc], axis=0)
        combined_test = pd.concat([datatestp, datatestc], axis=0)
        labeltest= combined_test['Cancer'] 
        parttest = combined_test['Part']
        typetest = combined_test['type']
        combined_test.drop('Cancer', axis=1, inplace=True)
        combined_test.drop('Part', axis=1, inplace=True)
        combined_test.drop('type', axis=1, inplace=True)
        #path_data = f"{self.file_path}/{datatype}data.npy"
        #standardize Microarray data
        df_train, df_val = train_test_split(combined_train, test_size=0.1, stratify=combined_train['type'], random_state=42)
        print(df_train.shape)
        labelval= df_val['Cancer'] 
        labeltrain = df_train['Cancer']
        partval = df_val['Part']
        parttrain = df_train['Part']
        typetrain = df_train['type']
        typeval = df_val['type']
        df_train.drop('Cancer', axis=1, inplace=True)
        df_train.drop('Part', axis=1, inplace=True)
        df_train.drop('type', axis=1, inplace=True)
        df_val.drop('Cancer', axis=1, inplace=True)
        df_val.drop('Part', axis=1, inplace=True)
        df_val.drop('type', axis=1, inplace=True)

        if standardize:
            X_train = standardized(df_train)
            X_val = standardized(df_val)
            X_test = standardized(combined_test)
        else:
             X_train = df_train.to_numpy()
             X_val = df_val.to_numpy()
             X_test = combined_test.to_numpy() 
        #enocode label
        y_train = map_type_to_indices(typetrain)
        y_test = map_type_to_indices(typetest)
        y_val = map_type_to_indices(typeval)
        X_train =torch.from_numpy(X_train)
        X_test =torch.from_numpy(X_test)
        X_val =torch.from_numpy(X_val)
        
        
        trainloader = torch.utils.data.DataLoader(TensorDataset(X_train,y_train), batch_size=batch_size, shuffle=True,        num_workers=1)#,drop_last=True)
        valloader = torch.utils.data.DataLoader(TensorDataset(X_val,y_val), batch_size=batch_size, shuffle=True, num_workers=1)#,drop_last=True)2 or 6 or 8 (0 it slows my training) I tied 10,15 is the same as 1)
        testloader = torch.utils.data.DataLoader(TensorDataset(X_test,y_test), batch_size=batch_size, shuffle=True, num_workers=1)#,drop_last=True)
        
        return trainloader,valloader,testloader
