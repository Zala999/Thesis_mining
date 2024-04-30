#This code has been developed on the basis of the work presented in the following github repository: "https://github.com/michal-nahlik/solafune-finding-mining-sites-2024.git"

#Importing the necessary libraries
import os
import random
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2
import timm
from sklearn.metrics import f1_score as sklearn_f1
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

#Loading training data
train_data = pd.read_csv(f'/home/u387021/thesis/mining/data/train/train/answer.csv', names=["file_name", "label"], header=None)

#Function for applying the selected data augmentations and resizing for training set
train_preprocess = v2.Compose([
    v2.ToTensor(), #Convert image data into tensor
    v2.Resize((224, 224)), #Resize the image
    v2.RandomRotation(degrees=90), #Randomly rotate the image by up to 90 degrees
    v2.RandomHorizontalFlip(), #Randomly flip the image horizontally
    v2.RandomVerticalFlip(), #Randomly flip the image vertically
    v2.ToDtype(torch.float32, scale=False),]) #Change the data type to float32

#Function for pre-processing the validation set
val_preprocess = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=False),])


#Function for adding NDVI and NDMI index & normalizing between -1 and 1
def adding_index(sat_image):
    NDMI = (sat_image[..., 7] - sat_image[..., 10]) / (sat_image[..., 7] + sat_image[..., 10] + 1e-10) #Calculates NDMI index
    NDVI = (sat_image[..., 7] - sat_image[..., 3]) / (sat_image[..., 7] + sat_image[..., 3] + 1e-10) #Calculates NDVI index
    sat_image = np.concatenate([sat_image * 2 - 1, np.expand_dims(NDMI, axis=-1),np.expand_dims(NDVI, axis=-1),], axis=-1,) #Adds the indices as new channels
    return sat_image

#Class for training the models
class SatelliteTrainDataset(Dataset):
    def __init__(self, directory, data, transform_pipeline,):
        self.directory = directory #Path to where images are stored
        self.data = data #Defining the data frame
        self.transform_pipeline = transform_pipeline #Transformations to apply to each image

    def __len__(self):
        return len(self.data) #Total number of images in the dataset

    def __getitem__(self, index):
        im = self.data.loc[index]
        sat_image = tifffile.imread(f"{self.directory}{im.file_name}")
        sat_image = adding_index(np.array(sat_image)) #Adding NDVI & NDMI indices
        sat_image = self.transform_pipeline(sat_image) #Applying the specified transformations
        label = im.label #Getting the label
        return {"image": sat_image, "label": torch.tensor(label, dtype=torch.long)}


#Function for optimizing and evaluating the threshold based on F1-score
def optimize_threshold(actual_labels, model_predictions):
    initial_f1 = sklearn_f1(actual_labels, model_predictions > 0.5)
    optimal_f1 = 0
    optimal_threshold = -1
    #Evaluating different thresholds to find the best F1-score
    for threshold_increment in range(100):
        current_threshold = threshold_increment / 100
        current_f1 = sklearn_f1(actual_labels, model_predictions > current_threshold)
        if current_f1 > optimal_f1:
            optimal_f1 = current_f1
            optimal_threshold = current_threshold

    #Calculating confusion matrix elements
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(
        actual_labels.numpy(), model_predictions.numpy() > optimal_threshold).ravel()
    print(f"True Positives: {true_pos}, True Negatives: {true_neg}, False Positives: {false_pos}, False Negatives: {false_neg}")
    return initial_f1, optimal_f1, optimal_threshold


#Setting seed for reproducibility
def set_seed(seed_val=10):
    random.seed(seed_val)
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)

#Function for evaluating the network on validation data and plotting the confusion matrix    
def assess_performance(config, network, loader, current_epoch=-1):
    #Defining the loss function
    criterion = nn.CrossEntropyLoss(weight=config.weights.to(config.device), label_smoothing=0.1)

    #Setting the network to evaluation mode
    network.eval()
    total_validation_loss = 0 #This variable tracks the cumulative loss

    collected_targets = [] #List to store all actual labels from the validation set
    collected_predictions = [] #List to store all model predictions

    loader_size = len(loader) #Total number of batches in the validation loader
    progress_bar = tqdm(enumerate(loader), total=loader_size)  #Progress bar definition
    for step, batch in progress_bar:
        images = batch["image"].to(config.device, non_blocking=True)  #Loading images to configured device
        labels = batch["label"].to(config.device, non_blocking=True)  #Loading labels to configured device

        #Disable gradient calculations
        with torch.no_grad():
            outputs = network(images) #Forward pass to get outputs

        loss = criterion(outputs, labels) #Calculating the loss between outputs and actual labels
        total_validation_loss += loss.item() #Adding current loss to the total loss

        #Storing labels and predictions
        collected_targets.append(labels.detach().cpu())
        collected_predictions.append(outputs.detach().cpu())
        del images, labels, outputs 

    #Concatenating lists of targets and predictions across all batches
    collected_targets = torch.cat(collected_targets, dim=0)
    collected_predictions = torch.cat(collected_predictions, dim=0)
    collected_predictions = F.sigmoid(collected_predictions) #Applying sigmoid to get probabilities

    #Calculating predictions
    binary_predictions = (collected_predictions[:, 1] > 0.5).int()
    #Computing the confusion matrix
    cm = confusion_matrix(collected_targets, binary_predictions)
    #Normalizing the matrix
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #Plotting the matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Purples', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix with Percentages')
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    plt.show()
    plt.savefig('confusion_matrix_swin_small512.png')

    #Calculating validation loss
    total_validation_loss /= loader_size
    #Determining the best threshold for classification using F1-score
    initial_f1, optimal_f1, optimal_threshold = optimize_threshold(collected_targets, collected_predictions[:, 1])

    #Printing results for the current epoch
    print(f'Epoch {current_epoch} validation loss = {total_validation_loss:.4f}, base f1 score (0.5 threshold) = {initial_f1:.4f} (optimal threshold: {optimal_threshold} -> f1 {optimal_f1:.4f})')
    return total_validation_loss, optimal_f1 


#This function conducts a single training cycle across one epoch of data
def train_epoch(configuration, neural_net, loader, opt, lr_scheduler, epoch):
    #Initializing a gradient scaler
    grad_scaler = torch.cuda.amp.GradScaler(enabled=configuration.apex)
    #Defining the loss function with class weighting and label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(weight=configuration.weights.to(configuration.device), label_smoothing=0.1)

    #Setting the neural network to training mode
    neural_net.train()
    cumulative_loss = 0  #Variable to track loss across all batches
    lr_track = []  #List to record learning rate at each batch

    collected_targets = []  #List to store all targets
    collected_predictions = []  #List to store all model predictions

    loader_size = len(loader)  #Total number of batches in the loader
    progress = tqdm(enumerate(loader), total=loader_size)  #Progress bar for feedback
    for step_index, batch in progress:
        images = batch["image"].to(configuration.device, non_blocking=True)  
        labels = batch["label"].to(configuration.device, non_blocking=True)  

        with torch.cuda.amp.autocast(enabled=configuration.apex):
            predictions = neural_net(images)  #Forward pass - computing predictions
            loss = criterion(predictions, labels)  #Computing the loss between predictions and actual labels

        #Scaling loss
        grad_scaler.scale(loss).backward()
        #Applying gradient clipping
        torch.nn.utils.clip_grad_norm_(neural_net.parameters(), max_norm=configuration.clip_val)

        cumulative_loss += loss.item()  #Accumulating loss
        grad_scaler.step(opt)  #Updating model weights
        grad_scaler.update()  #Updating scaler
        opt.zero_grad()  #Zeroing the gradients

        #Determine current learning rate
        current_lr = opt.param_groups[0]['lr'] if lr_scheduler is None else lr_scheduler.get_last_lr()[0]
        if lr_scheduler:
            lr_scheduler.step()  # Updating scheduler

        #Updating the progress bar with current information
        progress.set_description(f"Epoch {epoch} training {step_index+1}/{loader_size} [LR {current_lr:0.6f}] - loss: {cumulative_loss/(step_index+1):.4f}")
        lr_track.append(current_lr)  #Appending the current learning rate to history

        collected_targets.append(labels.detach().cpu())  #Collecting targets for metrics calculation
        collected_predictions.append(predictions.detach().cpu())  #Collecting predictions for metrics calculation
        del images, labels  

    #Getting all collected targets and predictions
    collected_targets = torch.cat(collected_targets, dim=0)
    collected_predictions = torch.cat(collected_predictions, dim=0)
    collected_predictions = F.sigmoid(collected_predictions)  #Applying sigmoid to obtain probabilities

    #Computing the loss and F1-score
    cumulative_loss /= loader_size
    initial_f1, optimal_f1, optimal_threshold = optimize_threshold(collected_targets, collected_predictions[:, 1])

    #Printing the results of the current epoch
    print(f'Epoch {epoch} train loss = {cumulative_loss:.4f}, base f1 score (0.5 threshold) = {initial_f1:.4f} (best threshold: {optimal_threshold} -> f1 {optimal_f1:.4f})')
    return cumulative_loss, optimal_f1, lr_track


#Class for defining parameters
class CFG:
    seed=10 #Sets seed number
    num_folds = 5 #Defining number of folds
    train_folds = [0, 1, 2, 3, 4] 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    apex=True 

    model_name = 'swin_small_patch4_window7_224' #Specifying model name to be used
    epochs = 20 #Defining the number of epochs
    weights =  torch.tensor([0.206, 0.794],dtype=torch.float32) #Class weights to handle imbalance in dataset

    clip_val = 1000. #Value for gradient clipping to prevent exploding gradients
    batch_size = 16 #Defining batch size
    gradient_accumulation_steps = 1 #Number of steps to accumulate gradients before performing a backpropagation

    lr = 1e-4 #Learning rate
    weight_decay=1e-2 #Setting weight decay


#Splitting the data for stratified k fold cross validation
sgkf = StratifiedKFold(n_splits=CFG.num_folds, random_state=CFG.seed, shuffle=True)
for i, (train_index, test_index) in enumerate(sgkf.split(train_data["file_name"].values, train_data["label"].values)):
    train_data.loc[test_index, "fold"] = i


#Iterating over each fold specified in the configuration
for current_fold in CFG.train_folds:

    #Initializing the random seed
    set_seed(CFG.seed)

    #Separating the dataset into training and validation sets
    training_data = train_data[train_data["fold"] != current_fold].reset_index(drop=True)
    validation_data = train_data[train_data["fold"] == current_fold].reset_index(drop=True)

    #Printing data distribution by labels for training and validation sets
    data_distribution = pd.merge(validation_data.groupby("label")["file_name"].count().rename("Validation Count").reset_index(), training_data.groupby("label")["file_name"].count().rename("Training Count").reset_index(), on="label", how="left").T
    print(data_distribution)

    #Preparing datasets and dataloaders for train and val
    train_dataset = SatelliteTrainDataset('/home/u387021/thesis/mining/data/train/train/train/', training_data, transform_pipeline=train_preprocess)
    valid_dataset = SatelliteTrainDataset('/home/u387021/thesis/mining/data/train/train/train/', validation_data, transform_pipeline=val_preprocess)
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=False)
    
    #Initializing the model, optimizer, and learning rate scheduler
    model = timm.create_model(CFG.model_name, in_chans=14, num_classes=2, pretrained=True)
    model.to(CFG.device)

    #Generating the model summary for obtaining the number of parameters
    summary(model, input_size=(1, 14, 224, 224))
    #Generating the FLOP count
    flop_analysis = FlopCountAnalysis(model, torch.randn(1, 14, 224, 224).to(CFG.device))
    print(f"FLOPs: {flop_analysis.total():,}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):_}") 


