import byteplot as bp
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


path_name_benign = 'Sample\\'
path_name_malicious = 'Sample\\'

class PDFDataset(Dataset):
    def __init__(self,plot_type):
        # call the __init__() of its parent class
        super().__init__()
        # We need to know if the incoming pdfs are converted to grayscale images using the byte plot or markov plot
        self.plot_type = plot_type
        self._load()
        # do something to initialize the pdf dataset object
        pass

    def __len__(self):
        # return the number of instances of the dataset
        # how to reference of this list like you did in your dataset?
        return len(self.labels)

    def __getitem__(self, idx):
        # return X, y
        # X is the byte plot or markov plot
        # y is the corresponding label of X
        # still confused about this function and tensors in general?
        X, y= 1, 2
        X, y = self._to_tensor(X, y)
        return X, y

    def _to_tensor(self, X, y):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
    # Get the actual dataset pdfs and load them into a dictionary(good and bad pdfs)
    def _load(self):
        # In a sample directory with only 10 pdfs for now
        benign_files = (path_name_benign)
        malicious_files = (path_name_malicious)
        # Dictionary to store each plot for good and bad pdfs
        data_by_type = {'benign':None,'malicious':None}
        path_list_benign = []
        path_list_malicious = []
        # only 2 key/value pairs in the dictionary, 'benign' and 'malicious'. Each value will be a list containing x and y path names to corresponding grayscale images
        if self.plot_type == 'byte_plot':
            # Convert all pdfs to images and save their paths in a list
            for file_name in os.listdir(benign_files):
                if file_name.endswith('pdf'):
                    # convert returns the path of the newly created image in a string after converting it, which (should) be added as a value in the dictionary
                    path_list_benign.append(bp.convert(benign_files,file_name,256))
            # add this list to the dictionary as benign's value
            data_by_type['benign'] = path_list_benign
            # Do the same for the malicious files
            for file_name in os.listdir(malicious_files):
                if file_name.endswith('pdf'):
                    path_list_malicious.append(bp.convert(benign_files,file_name,256))
            data_by_type['malicious'] = path_list_malicious

        elif self.plot_type == 'markov_plot':
            data_by_type['benign'] = 1 # put the function call here to convert
            data_by_type['malicious'] = 1 # put the function call here to convert

        # Creates one large list by concatenating 2 smaller lists. Each smaller list has size = total number of values under the corresponding key in the data_by_type dictionary(benign or malicious). The large list (labels) has the same size as the entire dataset, and consists of x entries of "malicious" and y entries of "benign"
        labels = []
        for k in data_by_type.keys():
            n = len(data_by_type[k])
            label = np.repeat(k,n)
            labels.extend(label)
        


def testcase_test_pdfdataset():
    dataset = PDFDataset('byte_plot')

    # setup dataloader
    # check pytorch document for the parameter list
    dl = DataLoader(
        dataset,
        batch_size=32, 
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        )
    
    # loop through dataset
    for X, y in dl:
        # print out X and y 
        print(X, y)

if __name__ == '__main__':
    testcase_test_pdfdataset()