#Part of this code has been developed on the basis of the work presented in the following github repository: "https://github.com/michal-nahlik/solafune-finding-mining-sites-2024.git"

#Importing the libraries
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import timm
from torchvision.transforms import v2


#Function for adding NDVI and NDMI index & normalizing between -1 and 1
def adding_index(sat_image):
    NDMI = (sat_image[..., 7] - sat_image[..., 10]) / (sat_image[..., 7] + sat_image[..., 10] + 1e-10) #Calculates NDMI index
    NDVI = (sat_image[..., 7] - sat_image[..., 3]) / (sat_image[..., 7] + sat_image[..., 3] + 1e-10) #Calculates NDVI index
    sat_image = np.concatenate([sat_image * 2 - 1, np.expand_dims(NDMI, axis=-1),np.expand_dims(NDVI, axis=-1),], axis=-1,) #Adds the indices as new channels
    return sat_image


#Class for handling test image loading and transforming
class SatelliteTestDataset(Dataset):
    def __init__(self, directory, file_names, transform_pipeline=None):
        self.directory = directory  #Directory where test images are stored
        self.file_names = file_names
        self.transform_pipeline = transform_pipeline  #Transformations to be applied

    def __len__(self):
        return len(self.file_names)  #Returns the number of items in the dataset

    def __getitem__(self, index):
        #Loads an image file, adds NDVI and NDMI indices, and applies transformations
        sat_image = tifffile.imread(f"{self.directory}{self.file_names[index]}")
        sat_image = adding_index(sat_image) 
        sat_image = self.transform_pipeline(sat_image) 
        return {"image": sat_image}

    
#Class for defining parameters
class CFG:
    seed=777 #Sets seed number
    num_folds = 5 #Defining number of folds
    train_folds = [0, 1, 2, 3, 4] 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    apex=True 

    model_name = 'resnet50' #Specifying model name to be used
    epochs = 20 #Defining the number of epochs
    weights =  torch.tensor([0.206, 0.794],dtype=torch.float32) #Class weights to handle imbalance in dataset

    clip_val = 1000. #Value for gradient clipping to prevent exploding gradients
    batch_size = 16 #Defining batch size
    gradient_accumulation_steps = 1 #Number of steps to accumulate gradients before performing a backpropagation

    lr = 1e-4 #Learning rate
    weight_decay=1e-2 #Setting weight decay
    

#Defining transformations to for the test set
test_resize = v2.Compose([
    v2.ToTensor(),  #Converting images to tensors
    v2.Resize((224, 224)),  #Resizing images for model compatibility
    v2.ToDtype(torch.float32, scale=False),])

#Loading the test data
test_data = pd.read_csv('/home/u387021/thesis/mining/uploadsample.csv', names=["file_name", "label"], header=None)

#Initializing the test dataset and dataloader with a batch size of 1000 for performance measurement
test_dataset = SatelliteTestDataset('/home/u387021/thesis/mining/data/test/', test_data["file_name"].values, test_resize)
large_test_loader = DataLoader(
    test_dataset,
    batch_size=1000,  
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    drop_last=True  
)

#Ensuring that the device is set correctly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Loading the trained model
model = timm.create_model(CFG.model_name, in_chans=14, num_classes=2, pretrained=False)
model_path = 'best_model_VT_resnet50_66_fold_4.pth'  #Replacing with tested model's path
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

#Start the timer, run the model on the batch, and end the timer
start_time = time.time()
with torch.no_grad():
    for batch in large_test_loader:
        inputs = batch["image"].to(device)
        outputs = model(inputs)
        break  #We only measure the first batch
end_time = time.time()

#Calculating the throughput
time_taken = end_time - start_time
throughput = 1000 / time_taken  

print(f"Time taken to process 1000 images: {time_taken:.4f} seconds")
print(f"Throughput: {throughput:.2f} images per second")


