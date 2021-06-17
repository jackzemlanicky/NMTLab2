import torch
from torch.utils.data import Dataset, DataLoader
import os


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
        lenght = 100
        return lenght

    def __getitem__(self, idx):
        # return X, y
        # X is the byte plot or markov plot
        # y is the corresponding label of X
        X, y= 1, 2
        X, y = self._to_tensor(X, y)
        return X, y

    def _to_tensor(self, X, y):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
    # Get the actual dataset folders (good and bad pdfs)
    def _load(self):
        benign_files = ('/Data/CLEAN_PDF_9000_files')
        malicious_files = ('/Data/MALWARE_PDF_PRE_04-2011_10982_files')
        # Dictionary to store each plot for good and bad pdfs
        data_by_type = {'benign':None,'malicious':None}
        if self.plot_type == 'byte_plot':
            return byte_plots
            # This is where I believe I have to use my program that converts pdf's to grayscale images using the byte plot strategy.
        elif self.plot_type == 'markov_plot':
            return markov_plots
            # Use Tyler's markov plot to convert each pdf from the dataset and load them into the dictionary


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