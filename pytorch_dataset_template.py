import byteplot as bp
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2


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
        # return X, y, which is the array at index idx and the label (benign or malicious) at idx
        # X is the byte plot or markov plot
        # y is the corresponding label of X
        X = self.X[idx]
        y = self.labels[idx]
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
                    # converts each image and adds its respective integer array to the dictionary
                    bp.convert(benign_files,file_name,256)
                    img_file_name=file_name.replace("pdf","png")
                    path_name = f"{benign_files}{img_file_name}"
                    path_list_benign.append(cv2.imread(path_name,cv2.IMREAD_UNCHANGED))
            # add this list to the dictionary as benign's value
            data_by_type['benign'] = path_list_benign
            # Do the same for the malicious files
            for file_name in os.listdir(malicious_files):
                if file_name.endswith('pdf'):
                    bp.convert(malicious_files,file_name,256)
                    img_file_name=file_name.replace("pdf","png")
                    path_name = f"{malicious_files}{img_file_name}"
                    path_list_malicious.append(cv2.imread(path_name,cv2.IMREAD_UNCHANGED))
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
        self.labels = labels
        # Implement list of inputs, X
        X = []
        for k in data_by_type.keys():
            X.extend(data_by_type[k])
        self.X = X

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
    # print(dataset.X)
    # loop through dataset
    for X, y in dl:
        # print out X and y 
        print(X, y)
# __name__ is an attribute of the file itself, essentially a 'main' function
if __name__ == '__main__':
    testcase_test_pdfdataset()