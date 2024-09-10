import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.model_selection import train_test_split
import biomart
from biomart import BiomartServer
import pandas as pd


class typepreprocee():

    """  Class to preprocess the public data and save it to the local.
    """
    
    def __init__(self):
        
        """  Function to define path of the public data and the paths where the preprocessed data will be saved
        """ 
        self.refer_columns = ['Characteristics.disease.','Characteristics.organism.part.']
        self.column = 'part'
        self.path = "/home/commun/MicroArray/E-MTAB-3732/class.csv"
        
        self.file_path = 'data/Microarrays_files/'
        self.cell_path = 'data/Microarrays_files/cell_data/'
        self.patient_path = 'data/Microarrays_files/patient_data/'
        self.pathmic = "/home/commun/MicroArray/E-MTAB-3732/E-MTAB-3732.data2.csv"
        
    
    def typeArraypreprocess(self):
        
        """  The main function that defines the preprocessing methods for cell line and patient data
        """ 
        
        data = pd.read_csv(self.path)

        
        data.dropna(subset = ['part'],inplace=True)

        cellline = data[data['cell']=="cell line"]
        patient = data[data['cell']=="patient"]

        
        dfMicroarray =pd.read_csv(self.pathmic) #this line takes a lot of time
        
        #get only the row that correspond to a coding gene
        
        coding_gene_data = self.get_coding_gene()
        
        # get only the gene expression level of probes that matches a coding gene from the Microarray data
        dfMicro = dfMicroarray[dfMicroarray['CompositeSequence Identifier'].isin(coding_gene_data['affy_hg_u133_plus_2'])]
        
        columns = dfMicro.columns
        # split the microarray data between cell line and patient microarray
        cellcol = [x for x in columns if x in cellline['Source.Name'].to_numpy()]
        patientcol = [x for x in columns if x in patient['Source.Name'].to_numpy()]
        celldata = dfMicro[dfMicro.columns.intersection(cellcol)]
        patientdata = dfMicro[dfMicro.columns.intersection(patientcol)] 

        celldata = celldata.to_numpy()
        patientdata = patientdata.to_numpy()
        celldataa = celldata.T
        patientdataa= patientdata.T
        
        # Get Tissue types and phenotype cancer for cell line and patient data
        labelpatient =patient['Cancer'].values
        labelcell =cellline['Cancer'].values
        partpatient = patient['part'].values
        partcell =cellline['part'].values
        
        # create a pd data frame of cell microarray data with cancer phenotype and parts
        datacell = pd.DataFrame(celldataa)
        series_cell = pd.Series(labelcell, name='Cancer')
        series_cellparat =pd.Series(partcell, name='Part')
        
        # get the probe cell line expression level and tissues type and cancer phenotype in one dataframe
        celldata = pd.concat([series_cell,series_cellparat, datacell], axis=1)
        
        # split cellline data between test and train
        celldata_train, celldata_test = train_test_split(celldata, test_size=0.1, shuffle=True)
        
        # save cell line train and test data to local as parquet 
        celldatatable = pa.Table.from_pandas(celldata_train)
        path_celltrain = self.cell_path+ 'celldata_train2'
        pq.write_table(celldatatable, path_celltrain)
        
        celldatatablet = pa.Table.from_pandas(celldata_test)
        path_celltraint = self.cell_path+ 'celldata_test2'
        pq.write_table(celldatatablet, path_celltraint)       
        
        
        # do same work for patient data 
        
        # create a pd data frame of cell microarray data with cancer phenotype and parts
        datapatient = pd.DataFrame(patientdataa)
        series_patient = pd.Series(labelpatient, name='Cancer')
        series_patientparat =pd.Series(partpatient, name='Part')
        
        patientdata = pd.concat([series_patient,series_patientparat, datapatient], axis=1)
        # split patient data between test and train
        patientdata_train, patientdata_test = train_test_split(patientdata, test_size=0.1, shuffle=True)
        
        # save patient train an test data to local as parquet 
        patientdatatable = pa.Table.from_pandas(patientdata_train)
        path_patienttrain = self.patient_path+ 'patientdata_train2'
        pq.write_table(patientdatatable, path_patienttrain)
        
        patientdatatablet = pa.Table.from_pandas(patientdata_test)
        path_patienttraint = self.patient_path+ 'patientdata_test2'
        pq.write_table(patientdatatablet, path_patienttraint)       
        
        
        
        #patientdatacancer = patientdata[labelpatient == "cancer"]
        #celldatacancer = celldata[labelpatient == "cancer"]
        #np.save(self.file_path+ 'celldata.npy',celldata)
        #np.save(self.file_path+'patientdata', patientdata)
        #np.save(self.file_path+ 'patientlabel.npy',labelpatient)
        #np.save(self.file_path+'celllabel.npy', labelcell)
        #np.save(self.file_path+ 'partpatientcancer.npy',partpatientcancer)
        #np.save(self.file_path+'partcellcancer.npy', partcellcancer)   
        #np.save(self.file_path+'patientdatacancer.npy', patientdatacancer)
        #np.save(self.file_path+'patientdatacancer.npy', patientdatacancer)
        # adjust it to load one type patharr    
    
    
    
    
    def preprocessing(self):
        
            """   The function that calls the preprocessing and saves cellline and patient data ( train-test) as pandas data frames in the Micrparrays_files 
            """
        
            self.typeArraypreprocess()
            
            
            
    def get_coding_gene(self):
        
        """   Function used to extract only the affymetrix probe Ids that match a coding gene. It helps also in dimensionality reduction
    
        Returns:
            result_df (pd.DataFrame): A dataframe contains the affymetrix probes IDs that are matched to coding genes with the Id of these genes
        """        
        
        # Connect to the Biomart server
        server = BiomartServer("http://www.ensembl.org/biomart")

        # Get the dataset and mart
        mart = server.datasets['hsapiens_gene_ensembl']

        # Define attributes and filters
        attributes = ['affy_hg_u133_plus_2','ensembl_gene_id', 'gene_biotype']
        # Perform the query
        response = mart.search({'attributes': attributes})
        data = []
        for line in response.iter_lines():
            data.append(line.decode('utf-8').split('\t'))

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=attributes)
        # Counts the appearence of the affymetrix probe IDs
        value_counts = df['affy_hg_u133_plus_2'].value_counts()

        # Choose only affymetrix probes ID that is matched to only one gene
        values_occuring_once = value_counts[value_counts == 1].index.tolist()
        result_df = df[df['affy_hg_u133_plus_2'].isin(values_occuring_once)]
        
        
        # Choose from the later only probes Ids that are matched to coding genes
        result_df =result_df[result_df['gene_biotype']== 'protein_coding']
        
        return result_df
        
