import copy
import json
import math
import os
import random
import shutil
import time
import pandas as pd

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data

from scipy.stats import entropy
import sklearn
import copy

import gc
from torch.utils.data import DataLoader


import torchvision.transforms as transforms
from PIL import Image

import Verma.experts as vexp
import Verma.losses as vlos
from Verma.utils import AverageMeter, accuracy
import Verma.resnet50 as vres
from AL.utils import *
from AL.metrics import *
from AL.neural_network import NetSimple

import Dataset.Dataset as ds
import expert as ex

class NIHExpertDatasetMemory():
    def __init__(self, images, filenames, targets, expert_fn, labeled, indices = None, expert_preds = None, param=None, preload=False, preprocess=False, image_container=None):
        """
        Original cifar dataset
        images: images
        targets: labels
        expert_fn: expert function
        labeled: indicator array if images is labeled
        indices: indices in original CIFAR dataset (if this subset is subsampled)
        expert_preds: used if expert_fn or have different expert model
        """
        self.preprocess = preprocess
        self.filenames = filenames
        self.targets = np.array(targets)
        self.expert_fn = expert_fn
        self.labeled = np.array(labeled)
        
        self.image_ids = filenames
        self.preload = False
        self.PATH = param["PATH"]
        
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3]],
                                         std=[x / 255.0 for x in [63.0]])
        self.transform_test = transforms.Compose([transforms.Resize(128), transforms.ToTensor(), normalize])

        self.image_container = image_container

        self.images = images
        if images is not None:
            self.images = images
            self.preload = True
        else:
            self.preload = preload
            if self.preload:
                self.images = []
                self.loadImages()
        
        if expert_preds is not None:
            self.expert_preds = expert_preds
        else:
            if expert_fn is not None:
                self.expert_preds = np.array(expert_fn(self.images, torch.FloatTensor(targets), fnames = self.filenames))
            else:
                self.expert_preds = np.array([-1 for filename in self.filenames])
        for i in range(len(self.expert_preds)):
            if self.labeled[i] == 0:
                self.expert_preds[i] = -1 # not labeled by expert
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.array(list(range(len(self.targets))))
            
    def loadImage(self, idx):
        """
        Load one single image
        """
        if self.image_container is not None:
            return self.image_container.get_image_from_name(self.image_ids[idx])
        else:
            return Image.open(self.PATH + "images/" + self.image_ids[idx]).convert("RGB").resize((244,244))
            
    def getImage(self, idx):
        """
        Returns the image from index idx
        """
        if self.preload:
            return self.images[idx]
        else:
            return self.loadImage(idx)

    def loadImages(self):
        """
        Load all images
        """
        if self.image_container is not None:
            self.images = self.image_container.get_images_from_name(self.image_ids)
            if self.preprocess:
                print("Preprocessed")
                #self.images = [self.transformImage(img) for img in self.images]
        else:
            for idx in range(len(self.image_ids)):
                if self.preprocess:
                    self.images.append(self.transformImage(self.loadImage(idx)))
                else:
                    self.images.append(self.loadImage(idx))

    def transformImage(self, img):
        return self.transform_test(img)
    
    
    def __getitem__(self, index):
        """Take the index of item and returns the image, label, expert prediction and index in original dataset"""
        label = self.targets[index]
        img = self.getImage(index)
        if self.preprocess:
            image = img
        else:
            image = self.transformImage(img)
        #image = self.transform_test(self.images[index])
        filename = self.filenames[index]
        expert_pred = self.expert_preds[index]
        indice = self.indices[index]
        labeled = self.labeled[index]
        return torch.FloatTensor(image), label, expert_pred, indice, labeled, str(filename)

    def __len__(self):
        return len(self.targets)

def sampleIndices(n, k, all_indices, experten, seed = None):
    if seed is not None:
        set_seed(seed)
    same_indices = random.sample(all_indices, k)
    diff_indices = []
    used_indices = same_indices
    indices = []
    if k == n:
        for expert in experten:
            indices.append(same_indices)
    if k < n:
        for expert in experten:
            temp_indices = []
            count = 0 # To avoid infinity loop
            while len(temp_indices) < (n - k):
                count += 1
                temp = random.sample(all_indices, 1)
                if temp not in used_indices:
                    temp_indices = temp_indices + temp
                    used_indices = used_indices + temp
                if count >= 1000:
                    temp = random.sample(used_indices, n-k-len(temp_indices))
                    if isinstance(temp, list):
                        temp_indices = temp_indices + temp
                    else:
                        temp_indices.append(temp)
                    break
            indices.append(same_indices + temp_indices)
    return indices

def getIndicesWithoutLabel(all_indices, labeled_indices):
    temp = all_indices
    for indices in labeled_indices:
        temp = [x for x in temp if x not in indices]
    return temp

def get_least_confident_points(model, data_loader, budget):
    '''
    based on entropy score get points, can chagnge, but make sure to get max or min accordingly
    '''
    uncertainty_estimates = []
    indices_all = []
    for data in data_loader:
        images, labels, _, indices, _, filenames = data
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_size = outputs.size()[0]  
            for i in range(0, batch_size):
                output_i =  outputs.data[i].cpu().numpy()
                entropy_i = entropy(output_i)
                #entropy_i = 1 - max(output_i)
                uncertainty_estimates.append(entropy_i)
                indices_all.append(indices[i].item())
    indices_all = np.array(indices_all)
    top_budget_indices = np.argsort(uncertainty_estimates)[-budget:]
    actual_indices = indices_all[top_budget_indices]
    uncertainty_estimates = np.array(uncertainty_estimates)
    return actual_indices

def disagree(array):
    start = array[0]
    for el in array[1:]:
        if start != el:
            return start != el
    return False

def getQbQPoints(expert_models, data_loader, budget):
    """
    Selects n (budget) points with query by committee
    """
    # Get Predictions for all points for all experts
    #global prediction_matrix
    prediction_matrix = None
    indices_all = []
    for data in data_loader:
        images, labels, _, indices, _, filenames = data
        experts_preds = []
        for j, expert_model in enumerate(expert_models):
            with torch.no_grad():
                images = images.to(device)
                outputs_exp = expert_model(images)
                preds = []
                for i in range(outputs_exp.size()[0]):
                    pred_exp = outputs_exp.data[i].cpu().numpy()
                    pred_exp = pred_exp[1]
                    #preds.append(round(pred_exp))
                    preds.append(pred_exp)
                    if (j == 0): #Add the indices only the first time
                        indices_all.append(indices[i].item())
            experts_preds.append(np.array(preds))

        if prediction_matrix is None:
            prediction_matrix = np.swapaxes(np.array(experts_preds), 0, 1)
        else:
            prediction_matrix = np.concatenate((prediction_matrix, np.swapaxes(np.array(experts_preds), 0, 1)), axis=0)
    predictions_matrix = prediction_matrix

    #Get where the experts disagree

    print(predictions_matrix.shape)

    matrixx = [disagree(np.round(row)) for row in predictions_matrix]
    ids = np.array(indices_all)[matrixx]
    
    print("Disagreement on " + str(len(ids)) + " Points")
    if NEPTUNE:
        run["Disagreement Points"].append(len(ids))

    ids = ids[:budget].tolist()

    if len(ids) < budget:
        matrixx = [row for row in predictions_matrix if not disagree(np.round(row))]
        points = np.array([np.sum(np.abs(row - 0.5)) for row in matrixx])

        for row in np.array(matrixx)[points.argsort()[:(budget - len(ids))].tolist()]:
            ids.append(indices_all[np.argwhere(predictions_matrix == row)[0][0]])

    print(ids)

    return random.sample(ids, min(budget, len(ids)))

def getQbQPointsDifference(expert_models, data_loader, budget):
    """
    Selects n (budget) points with query by committee
    """
    # Get Predictions for all points for all experts
    #global prediction_matrix
    prediction_matrix = None
    indices_all = []
    for data in data_loader:
        images, labels, _, indices, _, filenames = data
        experts_preds = []
        for j, expert_model in enumerate(expert_models):
            with torch.no_grad():
                images = images.to(device)
                outputs_exp = expert_model(images)
                preds = []
                for i in range(outputs_exp.size()[0]):
                    pred_exp = outputs_exp.data[i].cpu().numpy()
                    pred_exp = pred_exp[1]
                    #preds.append(round(pred_exp))
                    preds.append(pred_exp)
                    if (j == 0): #Add the indices only the first time
                        indices_all.append(indices[i].item())
            experts_preds.append(np.array(preds))

        if prediction_matrix is None:
            prediction_matrix = np.swapaxes(np.array(experts_preds), 0, 1)
        else:
            prediction_matrix = np.concatenate((prediction_matrix, np.swapaxes(np.array(experts_preds), 0, 1)), axis=0)
    predictions_matrix = prediction_matrix

    #Get where the experts disagree
    print(predictions_matrix.shape)

    matrixx = [row for row in predictions_matrix if disagree(np.round(row))]
    points = np.array([np.sum(np.abs(row - 0.5)) for row in matrixx])

    print("Disagreement on " + str(len(points)) + " Points")
    if NEPTUNE:
        run["Disagreement Points"].append(len(points))

    ids = []
    for row in np.array(matrixx)[points.argsort()[:budget].tolist()]:
        ids.append(indices_all[np.argwhere(predictions_matrix == row)[0][0]])

    if len(ids) < budget:
        matrixx = [row for row in predictions_matrix if not disagree(np.round(row))]
        points = np.array([np.sum(np.abs(row - 0.5)) for row in matrixx])

        for row in np.array(matrixx)[points.argsort()[:(budget - len(ids))].tolist()]:
            ids.append(indices_all[np.argwhere(predictions_matrix == row)[0][0]])

    print(ids)
    
    #print("Disagreement on " + str(len(ids)) + " Points")
    return ids[:budget]

def getExpertModels(indices, experts, train_dataset, val_dataset, test_dataset, param=None, seed=None, fold=None, mod="", image_container=None):
    
    # initialize data, Erhält alle Indizes der Daten
    all_indices = list(range(len(train_dataset.getAllIndices())))
    #all_data_x = train_dataset.getAllImagesNP()[all_indices]
    all_data_filenames = np.array(train_dataset.getAllFilenames())[all_indices]
    all_data_y = np.array(train_dataset.getAllTargets())[all_indices]
    
    print("Complete data generation")

    # Bestimmt die Indizes, welche gelabelt und welche ungelabelt sind

    set_seed(seed)

    #Samples the indices with k same and n-k different images
    """if k is not None:
        indices = sampleIndices(n = param["INITIAL_SIZE"], k = k, all_indices = all_indices, experten = experts, seed = seed)
    else:
        Intial_random_set = random.sample(all_indices, param["INITIAL_SIZE"])
        indices_labeled  = Intial_random_set
        indices_unlabeled= list(set(all_indices) - set(indices_labeled))"""

    gc.collect()

    # train expert model on labeled data
    # Expertenmodell variabel
    
    expert_models = []
    for i, expert in enumerate(experts):

        print("Starting with expert " + str(i))

        Intial_random_set = indices[i]
        indices_labeled  = Intial_random_set

        # Lädt die Datasets für die beschrifteten und unbeschrifteten Daten
        dataset_train_labeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_labeled], all_data_y[indices_labeled], expert.predict , [1]*len(indices_labeled), 
                                                       indices_labeled, param=param, preload=param["PRELOAD"], image_container=image_container)
        #dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], expert.predict , [0]*len(indices_unlabeled), indices_unlabeled, param=param, preload=param["PRELOAD"])
        #dataset_val_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], expert.predict , [1]*len(indices_unlabeled), indices_unlabeled, param=param, preload=param["PRELOAD"])
        # Lädt die Dataloaders
        dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=param["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
        #dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param["BATCH_SIZE"], shuffle=True,  num_workers=0, pin_memory=True)

        dataset_val_unlabeled = NIHExpertDatasetMemory(None, val_dataset.getAllFilenames(), np.array(val_dataset.getAllTargets()), expert.predict , [1]*len(val_dataset.getAllIndices()), 
                                                       val_dataset.getAllIndices(), param=param, preload=param["PRELOAD"], image_container=image_container)
        dataLoaderValUnlabeled = DataLoader(dataset=dataset_val_unlabeled, batch_size=param["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)

        print("Complete dataloader generation")

        gc.collect()
        
        expert_models.append(NetSimple(2, 3, 100, 100, 1000,500).to(device)) 
        run_expert(expert_models[-1], param["EPOCH_TRAIN"], dataLoaderTrainLabeled, dataLoaderValUnlabeled, param=param, id=expert.labelerId, seed=seed, fold=fold, 
                   n_images=param["INITIAL_SIZE"]) 

    print("Experts trained")

    #Returns all indices without any used label
    indices_unlabeled = getIndicesWithoutLabel(all_indices = all_indices, labeled_indices = indices)
    indices_labeled = list(set(all_indices) - set(indices_unlabeled))

    dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], expert.predict , [0]*len(indices_unlabeled), 
                                                     indices_unlabeled, param=param, preload=param["PRELOAD"], image_container=image_container)
    dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    
    data_sizes = []
    error_confidence = []
    data_sizes.append(param["INITIAL_SIZE"])
    
    print("Starting with AL")
    for round in range(param["MAX_ROUNDS"]):

        print(f'\n \n Round {round} \n \n')

        # get points where expert model is least confident on
        #indices_confidence =  random.sample(indices_unlabeled, BATCH_SIZE_AL)
        #indices_confidence = get_least_confident_points(model_expert, dataLoaderTrainUnlabeled, param["BATCH_SIZE_AL"])

        #Try to get better Points
        if mod == "disagreement":
            indices_qbq = getQbQPoints(expert_models, dataLoaderTrainUnlabeled, param["BATCH_SIZE_AL"])
        if mod == "disagreement_diff":
            indices_qbq = getQbQPointsDifference(expert_models, dataLoaderTrainUnlabeled, param["BATCH_SIZE_AL"])
        
        #indices_labeled  = indices_labeled + list(indices_confidence) 
        indices_labeled  = indices_labeled + list(indices_qbq) 
        indices_unlabeled= list(set(all_indices) - set(indices_labeled))
        
        # train model on labeled data
        for j, expert in enumerate(experts):

            dataset_train_labeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_labeled], all_data_y[indices_labeled], expert.predict , [1]*len(indices_labeled), indices_labeled, param=param, preload=param["PRELOAD"], image_container=image_container)
            

            #dataset_val_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], expert.predict , [1]*len(indices_unlabeled), indices_unlabeled, param=param, preload=param["PRELOAD"])
            #dataset_val_unlabeled = NIHExpertDatasetMemory(None, val_dataset.getAllFilenames(), np.array(val_dataset.getAllTargets()), expert.predict , [1]*len(val_dataset.getAllIndices()), val_dataset.getAllIndices(), param=param, preload=param["PRELOAD"])
            dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=param["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
            
            #dataLoaderValUnlabeled = DataLoader(dataset=dataset_val_unlabeled, batch_size=param["BATCH_SIZE"], shuffle=True,  num_workers=0, pin_memory=True)

            dataset_val_unlabeled = NIHExpertDatasetMemory(None, val_dataset.getAllFilenames(), np.array(val_dataset.getAllTargets()), expert.predict , [1]*len(val_dataset.getAllIndices()), val_dataset.getAllIndices(), param=param, preload=param["PRELOAD"], image_container=image_container)
            dataLoaderValUnlabeled = DataLoader(dataset=dataset_val_unlabeled, batch_size=param["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
            
            run_expert(expert_models[j], param["EPOCH_TRAIN"], dataLoaderTrainLabeled, dataLoaderValUnlabeled, param=param, id=expert.labelerId, seed=seed, fold=fold, n_images=param["INITIAL_SIZE"] + (round+1)*param["BATCH_SIZE_AL"])

        dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], expert.predict , [0]*len(indices_unlabeled), indices_unlabeled, param=param, preload=param["PRELOAD"], image_container=image_container)
        dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    
    print("Test Data:")
    dataset_test_unlabeled = NIHExpertDatasetMemory(None, test_dataset.getAllFilenames(), np.array(test_dataset.getAllTargets()), expert.predict , [1]*len(test_dataset.getAllIndices()), test_dataset.getAllIndices(), param=param, preload=param["PRELOAD"], image_container=image_container)
    dataLoaderVal = DataLoader(dataset=dataset_test_unlabeled, batch_size=param["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    met = {}
    for j, expert in enumerate(experts):
        temp = metrics_print_expert(expert_models[j], dataLoaderVal, id=expert.labelerId, seed=seed, fold=fold, n_images=param["INITIAL_SIZE"] + (param["MAX_ROUNDS"] + 5)*param["BATCH_SIZE_AL"], test=True)
        met[expert.labelerId] = temp
    print("AL finished")
    return expert_models, met

def getExpertModel(indices, train_dataset, val_dataset, test_dataset, expert, param=None, seed=None, fold=None, image_container=None):
    
    # initialize data, Erhält alle Indizes der Daten
    all_indices = list(range(len(train_dataset.getAllIndices())))
    all_data_filenames = np.array(train_dataset.getAllFilenames())[all_indices]
    all_data_y = np.array(train_dataset.getAllTargets())[all_indices]

    set_seed(seed)
    
    Intial_random_set = indices
    indices_labeled  = Intial_random_set
    indices_unlabeled= list(set(all_indices) - set(indices_labeled))

    gc.collect()

    # Lädt die Datasets für die beschrifteten und unbeschrifteten Daten
    dataset_train_labeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_labeled], all_data_y[indices_labeled], expert.predict , [1]*len(indices_labeled), indices_labeled, param=param, preload=param["PRELOAD"], image_container=image_container)
    dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], None , [0]*len(indices_unlabeled), indices_unlabeled, param=param, preload=param["PRELOAD"], image_container=image_container)

    dataset_val_unlabeled = NIHExpertDatasetMemory(None, val_dataset.getAllFilenames(), np.array(val_dataset.getAllTargets()), expert.predict , [1]*len(val_dataset.getAllIndices()), val_dataset.getAllIndices(), param=param, preload=param["PRELOAD"], image_container=image_container)
    
    # Lädt die Dataloaders
    dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=param["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param["BATCH_SIZE_VAL"], shuffle=True, num_workers=4, pin_memory=True)
    
    dataLoaderValUnlabeled = DataLoader(dataset=dataset_val_unlabeled, batch_size=param["BATCH_SIZE_VAL"], shuffle=True, num_workers=4, pin_memory=True)
    
    print("Complete dataloader generation")

    gc.collect()

    # train expert model on labeled data
    # Expertenmodell variabel
    model_expert = NetSimple(2, 3, 100, 100, 1000,500).to(device)
    # Trainier Modell um Experten vorherzusagen
    
    run_expert(model_expert, param["EPOCH_TRAIN"], dataLoaderTrainLabeled, dataLoaderValUnlabeled, param=param, id=expert.labelerId, seed=seed, fold=fold, n_images=param["INITIAL_SIZE"]) 
    
    print("Expert trained")


    data_sizes = []
    error_confidence = []
    data_sizes.append(param["INITIAL_SIZE"])
        
    gc.collect()

    #Trainiere Rejector nur noch, wenn notwendig
    
    print("Starting with AL")
    for round in range(param["MAX_ROUNDS"]):

        print(f'\n \n Round {round} \n \n')

        # get points where expert model is least confident on
        #indices_confidence =  random.sample(indices_unlabeled, BATCH_SIZE_AL)
        indices_confidence = get_least_confident_points(model_expert, dataLoaderTrainUnlabeled, param["BATCH_SIZE_AL"])
        indices_labeled  = indices_labeled + list(indices_confidence) 
        indices_unlabeled= list(set(all_indices) - set(indices_labeled))

        dataset_train_labeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_labeled], all_data_y[indices_labeled], expert.predict , [1]*len(indices_labeled), indices_labeled, param=param, preload=param["PRELOAD"], image_container=image_container)
        dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], expert.predict , [0]*len(indices_unlabeled), indices_unlabeled, param=param, preload=param["PRELOAD"], image_container=image_container)

        #dataset_val_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], expert.predict , [1]*len(indices_unlabeled), indices_unlabeled, param=param, preload=param["PRELOAD"])
        #dataset_val_unlabeled = NIHExpertDatasetMemory(None, val_dataset.getAllFilenames(), np.array(val_dataset.getAllTargets()), expert.predict , [1]*len(val_dataset.getAllIndices()), val_dataset.getAllIndices(), param=param, preload=param["PRELOAD"])
        
        dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=param["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
        dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)

        
        # train model on labeled data
        run_expert(model_expert, param["EPOCH_TRAIN"], dataLoaderTrainLabeled, dataLoaderValUnlabeled, param=param, id=expert.labelerId, seed=seed, fold=fold, n_images=param["INITIAL_SIZE"] + (round+1)*param["BATCH_SIZE_AL"])


    print("Test Data:")
    dataset_test_unlabeled = NIHExpertDatasetMemory(None, test_dataset.getAllFilenames(), np.array(test_dataset.getAllTargets()), expert.predict , [1]*len(test_dataset.getAllIndices()), test_dataset.getAllIndices(), param=param, preload=param["PRELOAD"], image_container=image_container)
    dataLoaderVal = DataLoader(dataset=dataset_test_unlabeled, batch_size=param["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    met = metrics_print_expert(model_expert, dataLoaderVal, id=expert.labelerId, seed=seed, fold=fold, n_images=param["INITIAL_SIZE"] + (param["MAX_ROUNDS"] + 5)*param["BATCH_SIZE_AL"], test=True)
    print("AL finished")
    return model_expert, met

def train_expert_confidence(train_loader, model, optimizer, scheduler, epoch, apply_softmax, param=None, id=""):
    """Train for one epoch the model to predict expert agreement with label"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, label, expert_pred, _, _, filenames ) in enumerate(train_loader):
        expert_pred = expert_pred.long()
        expert_pred = (expert_pred == label) *1
        target = expert_pred.to(device)
        input = input.to(device)
        
        # compute output
        output = model(input)

        # compute loss
        
        if apply_softmax:
            loss = my_CrossEntropyLossWithSoftmax(output, target)
        else:
            #loss = my_CrossEntropyLoss(output, target)
            loss = my_CrossEntropyLoss(output, target, cost=param["COST"])
        
        # measure accuracy and record loss
        prec1 = accuracy(output.detach().data, target, topk=(1,))[0]
        #losses.update(loss.data.item(), input.size(0))
        losses.update(loss.detach().item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))
            

def run_expert(model, epochs, train_loader, val_loader, apply_softmax = False, param=None, id=0, seed=None, fold=None, n_images=None):
    '''
    train expert model to predict disagreement with label
    model: WideResNet model or pytorch model (2 outputs)
    epochs: number of epochs to train
    '''

    # define loss function (criterion) and optimizer
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    optimizer = torch.optim.SGD(model.parameters(), 0.001, #0.001
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)

    for epoch in range(0, epochs):
        # train for one epoch
        train_expert_confidence(train_loader, model, optimizer, scheduler, epoch, apply_softmax, param=param)
        if epoch % 10 == 0 and epoch != epochs:
            print("Eval")
            metrics_print_expert(model, val_loader, id, seed=seed, fold=fold)
            
    metrics_print_expert(model, val_loader, id=id, seed=seed, fold=fold, n_images=n_images)

def metrics_print_expert(model, data_loader, defer_net = False, id=0, seed=None, fold=None, n_images=None, test=False):
    '''
    Computes metrics for expert model error prediction
    model: model
    data_loader: data loader
    '''
    correct = 0
    total = 0
    
    #label_list = np.empty(0)
    #predictions_list = np.empty(0)
    label_list = []
    predictions_list = []
    # again no gradients needed
    with torch.no_grad():
        for data in data_loader:
            images, label, expert_pred, _ ,_, filenames = data
            expert_pred = expert_pred.long()
            expert_pred = (expert_pred == label) *1
            #expert_pred = (expert_pred == label).int()
            images, labels = images.to(device), expert_pred.to(device)
            outputs = model(images)
            #_, predictions = torch.max(outputs.data, 1) # maybe no .data
            _, predictions = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            label_list.extend(labels.cpu().numpy())
            predictions_list.extend(predictions.cpu().numpy())
            
            #label_list = np.concatenate((label_list, labels.cpu().numpy()), axis=0)
            #predictions_list = np.concatenate((predictions_list, predictions.cpu().numpy()), axis=0)

    label_list = np.array(label_list)
    predictions_list = np.array(predictions_list)
    
    print('Accuracy of the network on the %d test images: %.3f %%' % (total,
        100 * correct / total))
    
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(label_list, predictions_list, labels=[0, 1]).ravel()

    f1 = sklearn.metrics.f1_score(label_list, predictions_list)

    ac_balanced = sklearn.metrics.balanced_accuracy_score(label_list, predictions_list)

    met = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "f1": f1,
        "accurancy_balanced": ac_balanced,
    }

    if NEPTUNE:
        if test:
            run[f"Test_Seed_{seed}_Fold_{fold}_expert_{id}" + "/tn"].append(tn)
            run[f"Test_Seed_{seed}_Fold_{fold}_expert_{id}" + "/fp"].append(fp)
            run[f"Test_Seed_{seed}_Fold_{fold}_expert_{id}" + "/fn"].append(fn)
            run[f"Test_Seed_{seed}_Fold_{fold}_expert_{id}" + "/tp"].append(tp)

            run[f"Test_Seed_{seed}_Fold_{fold}_expert_{id}" + "/accuracy"].append(100 * correct / total)

            run[f"Test_Seed_{seed}_Fold_{fold}_expert_{id}" + "/f1"].append(f1)

            run[f"Test_Seed_{seed}_Fold_{fold}_expert_{id}" + "/accuracy_balanced"].append(ac_balanced)

        else:
            run[f"Seed_{seed}_Fold_{fold}_expert_{id}" + "/tn"].append(tn, step=n_images)
            run[f"Seed_{seed}_Fold_{fold}_expert_{id}" + "/fp"].append(fp, step=n_images)
            run[f"Seed_{seed}_Fold_{fold}_expert_{id}" + "/fn"].append(fn, step=n_images)
            run[f"Seed_{seed}_Fold_{fold}_expert_{id}" + "/tp"].append(tp, step=n_images)

            run[f"Seed_{seed}_Fold_{fold}_expert_{id}" + "/accuracy"].append(100 * correct / total, step=n_images)

            run[f"Seed_{seed}_Fold_{fold}_expert_{id}" + "/f1"].append(f1, step=n_images)

            run[f"Seed_{seed}_Fold_{fold}_expert_{id}" + "/accuracy_balanced"].append(ac_balanced)
    
    print("Confusion Matrix:")
    print(sklearn.metrics.confusion_matrix(label_list, predictions_list, labels=[0, 1]))
    print("F1 Score: " + str(f1))

    print("Accuracy balanced")
    print(ac_balanced)

    if test:
        return met