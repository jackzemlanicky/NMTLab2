import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


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
        return len(self.labels)

    def __getitem__(self, idx):
        # return X, y
        # X is the byte plot or markov plot
        # y is the corresponding label of X
        X, y= 1, 2
        X, y = self._to_tensor(X, y)
        return X, y

    def _to_tensor(self, X, y):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
    # Get the actual dataset pdfs and load them into a dictionary(good and bad pdfs)
    def _load(self):
        benign_files = ('Data/CLEAN_PDF_9000_files')
        malicious_files = ('Data/MALWARE_PDF_PRE_04-2011_10982_files')
        # Dictionary to store each plot for good and bad pdfs
        data_by_type = {'benign':None,'malicious':None}
        if self.plot_type == 'byte_plot':
            # This is where I believe I have to use my program that converts pdf's to grayscale images using the byte plot strategy and load the images into the dictionary
            data_by_type['benign'] = 1 # put the function call here to convert
            data_by_type['malicious'] = 1 # put the function call here to convert
            
        elif self.plot_type == 'markov_plot':
            data_by_type['benign'] = 1 # put the function call here to convert
            data_by_type['malicious'] = 1 # put the function call here to convert

        # Not sure the point of this label, since we kmow from the above dictionary whether we are getting malicious or benign pdf files
        labels = []
        for k in data_by_type.keys():
            n = len(data_by_type[k])
            label = np.repeat(k,n)
            labels.extend(label)
        # Not exactly sure what index and  "_" are being assigned to, unelss they are both being assigned the self.findclass call
        index, _ = self._find_class_(labels,one_hot=False)
        self.labels = index


def testcase_test_pdfdataset():
    dataset = PDFDataset()

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