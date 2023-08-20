import copy
import json
import math
import os
import random
import shutil
import time
import pandas as pd

import hashlib

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

#import Verma.experts as vexp
import Verma.losses as vlos
from Verma.utils import AverageMeter, accuracy
import Verma.resnet50 as vres
from AL.utils import *
from AL.metrics import *
from AL.neural_network import NetSimple
from AL.neural_network import ResnetPretrained

import Dataset.Dataset as ds
import expert as ex

def set_seed(seed, fold=None, text=None):
    if fold is not None and text is not None:
        s = text + f" + {seed} + {fold}"
        seed = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

        self.transformed_images = {}
            
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

    def getTransformedImage(self, image, image_id):
        """
        Transforms the image
        """
        if image_id not in self.transformed_images.keys():
            self.transformed_images[image_id] = self.transformImage(image)
        return self.transformed_images[image_id]
    
    
    def __getitem__(self, index):
        """Take the index of item and returns the image, label, expert prediction and index in original dataset"""
        label = self.targets[index]
        img = self.getImage(index)
        #if self.preprocess:
        #    image = img
        #else:
        #    image = self.transformImage(img)
        #image = self.transform_test(self.images[index])
        filename = self.filenames[index]
        expert_pred = self.expert_preds[index]
        indice = self.indices[index]
        labeled = self.labeled[index]

        #optimized
        image = self.getTransformedImage(img, filename)
        
        return torch.FloatTensor(image), label, expert_pred, indice, labeled, str(filename)

    def __len__(self):
        return len(self.targets)

def sampleIndices(n, k, all_indices, experten, seed = None, X=None, sample_equal=False, fold=None):
    
    if seed is not None:
        set_seed(seed, fold, text="")
    
    if X is not None and sample_equal:
        all_indices_0 = [indice for indice in all_indices if X["GT"][indice] == 0]
        all_indices_1 = [indice for indice in all_indices if X["GT"][indice] == 1]
        same_indices_0 = random.sample(all_indices_0, round(k/2))
        same_indices_1 = random.sample(all_indices_1, round(k/2))
        same_indices = same_indices_0 + same_indices_1
    else:
        same_indices = random.sample(all_indices, k)
        
    diff_indices = []
    used_indices = same_indices
    indices = {}
    if k == n:
        for labelerId in experten:
            indices[labelerId] = same_indices
    if k < n:
        for labelerId in experten:
            temp_indices = []
            count = 0 # To avoid infinity loop

            ######
            if sample_equal:
                print(f"Indices with GT=0: {n/2} and with GT=1: {n/2}")
                working_indices_gt[0] = all_indices_0
                working_indices_gt[1] = all_indices_1
                print(f"Len GT=0 {len(working_indices_gt[0])} and GT=1 {len(working_indices_gt[1])}")
                for gt in [0, 1]:
                    while len(temp_indices) < (n - round(k/2)):
                        count += 1
                        temp = random.sample(working_indices_gt[gt], 1)
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
            else:
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

            indices[labelerId] = (same_indices + temp_indices)

    for key, inds in indices.items():
        assert len(inds) == n, "Wrong length by sampling"
    return indices

def getIndicesWithoutLabel(all_indices, labeled_indices):
    temp = all_indices
    for key, indices in labeled_indices.items():
        temp = [x for x in temp if x not in indices]
    return temp

def get_least_confident_points(model, data_loader, budget, mod=None):
    '''
    based on entropy score get points, can chagnge, but make sure to get max or min accordingly
    '''

    assert ((mod is not None and (mod == "ssl" or mod == "al"))), "Give al or ssl as mod"
    if mod == "ssl":
        expert = model # only for better readability
    
    uncertainty_estimates = []
    indices_all = []
    for data in data_loader:
        images, labels, _, indices, _, _ = data
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            if mod == "al":
                assert isinstance(model, nn.Module), "expert_model is not type nn.Module"
                outputs = model(images)
            elif mod == "ssl":
                assert isinstance(expert, ex.Expert), "expert is not type Expert"
                features = expert.sslModel.embedded_model.get_embedding(batch=images)
                logits, _ = expert.sslModel.linear_model(features)
                scores = torch.softmax(logits, dim=1)
                outputs = scores
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

def getQbQPoints(expert_models, data_loader, budget, mod=None, param=None):
    """
    Selects n (budget) points with query by committee
    """

    assert ((mod is not None and (mod == "ssl" or mod == "al"))), "Give al or ssl as mod"
    
    # Get Predictions for all points for all experts
    prediction_matrix = None
    indices_all = []
    for data in data_loader:
        images, labels, _, indices, _, filenames = data
        experts_preds = []
        for j, expert_model in enumerate(expert_models):
            with torch.no_grad():
                images = images.to(device)
                if mod == "al":
                    assert isinstance(expert_model, nn.Module), "expert_model is not type nn.Module"
                    outputs_exp = expert_model(images)
                elif mod == "ssl":
                    assert isinstance(expert_models[expert_model], ex.Expert), "expert is not type Expert"
                    expert = expert_models[expert_model] # only for better readability
                    features = expert.sslModel.embedded_model.get_embedding(batch=images)
                    logits, _ = expert.sslModel.linear_model(features)
                    scores = torch.softmax(logits, dim=1)
                    outputs_exp = scores
                
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
    if param["NEPTUNE"]["NEPTUNE"]:
        run = param["NEPTUNE"]["RUN"]
        run["Disagreement Points"].append(len(ids))

    ids = ids[:budget].tolist()

    if len(ids) < budget:
        matrixx = [row for row in predictions_matrix if not disagree(np.round(row))]
        points = np.array([np.sum(np.abs(row - 0.5)) for row in matrixx])

        for row in np.array(matrixx)[points.argsort()[:(budget - len(ids))].tolist()]:
            ids.append(indices_all[np.argwhere(predictions_matrix == row)[0][0]])

    print(ids)

    return random.sample(ids, min(budget, len(ids)))

def getQbQPointsDifference(expert_models, data_loader, budget, mod=None, param=None):
    """
    Selects n (budget) points with query by committee
    """
    # Get Predictions for all points for all experts
    #global prediction_matrix

    assert ((mod is not None and (mod == "ssl" or mod == "al"))), "Give al or ssl as mod"
    
    prediction_matrix = None
    indices_all = []
    for data in data_loader:
        images, labels, _, indices, _, filenames = data
        experts_preds = []
        for j, expert_model in enumerate(expert_models):
            with torch.no_grad():
                images = images.to(device)
                if mod == "al":
                    assert isinstance(expert_model, nn.Module), "expert_model is not type nn.Module"
                    outputs_exp = expert_model(images)
                elif mod == "ssl":
                    assert isinstance(expert_models[expert_model], ex.Expert), "expert is not type Expert"
                    expert = expert_models[expert_model] # only for better readability
                    features = expert.sslModel.embedded_model.get_embedding(batch=images)
                    logits, _ = expert.sslModel.linear_model(features)
                    scores = torch.softmax(logits, dim=1)
                    outputs_exp = scores
                
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
    if param["NEPTUNE"]["NEPTUNE"]:
        run = param["NEPTUNE"]["RUN"]
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

def getExpertModels(indices, experts, train_dataset, val_dataset, test_dataset, param=None, seed=None, fold=None, mod="", learning_mod="", prediction_type="", image_container=None):
    
    # initialize data, Erhält alle Indizes der Daten
    all_indices = list(range(len(train_dataset.getAllIndices())))
    all_data_filenames = np.array(train_dataset.getAllFilenames())[all_indices]
    all_data_y = np.array(train_dataset.getAllTargets())[all_indices]
    
    metrics = {}
    
    set_seed(seed, fold, text="")

    param_al = param["AL"]

    gc.collect()

    # train expert model on labeled data
    # Expertenmodell variabel
    
    expert_models = {}
    for labelerId, expert in experts.items():

        Intial_random_set = indices[labelerId]
        indices_labeled  = Intial_random_set

        # Lädt die Datasets für die beschrifteten und unbeschrifteten Daten
        dataset_train_labeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_labeled], all_data_y[indices_labeled], expert.predict , [1]*len(indices_labeled), 
                                                       indices_labeled, param=param, preload=param_al["PRELOAD"], image_container=image_container)
        dataset_val_unlabeled = NIHExpertDatasetMemory(None, val_dataset.getAllFilenames(), np.array(val_dataset.getAllTargets()), expert.predict , [1]*len(val_dataset.getAllIndices()), 
                                                       val_dataset.getAllIndices(), param=param, preload=param_al["PRELOAD"], image_container=image_container)

        dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=param_al["BATCH_SIZE"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)
        dataLoaderValUnlabeled = DataLoader(dataset=dataset_val_unlabeled, batch_size=param_al["BATCH_SIZE_VAL"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)

        gc.collect()
        
        #expert_models.append(NetSimple(2, 3, 100, 100, 1000,500).to(device))
        model_folder = param["Parent_PATH"]+"/SSL_Working/NIH/Embedded"
        if param["cluster"]:
            model_folder += f"/Seed_{seed}_Fold_{fold}"
            
        expert_models[labelerId] = ResnetPretrained(2, model_folder, type="50").to(device)
        if torch.cuda.device_count() > 1:
            print("Use ", torch.cuda.device_count(), "GPUs!")
            expert_models[labelerId] = nn.DataParallel(expert_models[labelerId])
        dataloaders = (dataLoaderTrainLabeled, dataLoaderValUnlabeled)
        train_metrics, val_metrics = run_expert(expert_models[labelerId], param_al["EPOCH_TRAIN"], dataloaders, param=param, id=expert.labelerId, seed=seed, fold=fold, 
                   n_images=param_al["INITIAL_SIZE"], mod=learning_mod, prediction_type=prediction_type)
        
        metrics[labelerId] = {}
        metrics[labelerId]["Train"] = {}
        metrics[labelerId]["Train"][param_al["INITIAL_SIZE"]] = {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        met = testExpert(expert, val_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Val", model=expert_models[labelerId])
        metrics[labelerId]["Val"] = {
            "Start": met,
        }

        met = testExpert(expert, test_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Test", model=expert_models[labelerId])
        metrics[labelerId]["Test"] = {
            "Start": met,
        }

    #Returns all indices without any used label
    indices_unlabeled = getIndicesWithoutLabel(all_indices = all_indices, labeled_indices = indices)
    indices_labeled = list(set(all_indices) - set(indices_unlabeled))

    dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], None , [0]*len(indices_unlabeled), 
                                                     indices_unlabeled, param=param, preload=param_al["PRELOAD"], image_container=image_container)
    dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param_al["BATCH_SIZE_VAL"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)
    
    for round in range(param_al["ROUNDS"]):

        print(f'\n \n Round {round} \n \n')

        #Try to get better Points
        if mod == "disagreement":
            indices_qbq = getQbQPoints(expert_models.values(), dataLoaderTrainUnlabeled, param_al["LABELS_PER_ROUND"], mod=learning_mod, param=param)
        if mod == "disagreement_diff":
            indices_qbq = getQbQPointsDifference(expert_models.values(), dataLoaderTrainUnlabeled, param_al["LABELS_PER_ROUND"], mod=learning_mod, param=param)
        
        #indices_labeled  = indices_labeled + list(indices_confidence) 
        indices_labeled  = indices_labeled + list(indices_qbq) 
        indices_unlabeled= list(set(all_indices) - set(indices_labeled))
        
        # train model on labeled data
        for labelerId, expert in experts.items():

            dataset_train_labeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_labeled], all_data_y[indices_labeled], expert.predict , [1]*len(indices_labeled), 
                                                           indices_labeled, param=param, preload=param_al["PRELOAD"], image_container=image_container)
            dataset_val_unlabeled = NIHExpertDatasetMemory(None, val_dataset.getAllFilenames(), np.array(val_dataset.getAllTargets()), expert.predict , 
                                                           [1]*len(val_dataset.getAllIndices()), val_dataset.getAllIndices(), param=param, preload=param_al["PRELOAD"], 
                                                           image_container=image_container)

            dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=param_al["BATCH_SIZE"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)
            dataLoaderValUnlabeled = DataLoader(dataset=dataset_val_unlabeled, batch_size=param_al["BATCH_SIZE_VAL"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)

            dataloaders = (dataLoaderTrainLabeled, dataLoaderValUnlabeled)
            n_images = param_al["INITIAL_SIZE"] + (round+1)*param_al["LABELS_PER_ROUND"]
            train_metrics, val_metrics = run_expert(expert_models[labelerId], param_al["EPOCH_TRAIN"], dataloaders, param=param, id=expert.labelerId, seed=seed, fold=fold, 
                                                    n_images=n_images, mod=learning_mod, prediction_type=prediction_type)
            metrics[labelerId]["Train"][n_images] = {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

        
        dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], None , [0]*len(indices_unlabeled), 
                                                         indices_unlabeled, param=param, preload=param_al["PRELOAD"], image_container=image_container)
        dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param_al["BATCH_SIZE"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)

    
    dataset_test_unlabeled = NIHExpertDatasetMemory(None, test_dataset.getAllFilenames(), np.array(test_dataset.getAllTargets()), expert.predict , [1]*len(test_dataset.getAllIndices()), 
                                                    test_dataset.getAllIndices(), param=param, preload=param_al["PRELOAD"], image_container=image_container)
    dataLoaderVal = DataLoader(dataset=dataset_test_unlabeled, batch_size=param_al["BATCH_SIZE"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)
    met_test = {}
    for labelerId, expert in experts.items():
        temp = metrics_print_expert(expert_models[labelerId], dataLoaderVal, id=expert.labelerId, seed=seed, fold=fold, 
                                    n_images=param_al["INITIAL_SIZE"] + param_al["ROUNDS"]*param_al["LABELS_PER_ROUND"], step="Test", mod=learning_mod, prediction_type=prediction_type, param=param)
        met_test[expert.labelerId] = temp

        met = testExpert(expert, val_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Val", model=expert_models[labelerId])
        metrics[labelerId]["Val"]["End"] = met

        met = testExpert(expert, test_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Test", model=expert_models[labelerId])
        metrics[labelerId]["Test"]["End"] = met

    return expert_models, met_test, metrics

def getExpertModel(indices, train_dataset, val_dataset, test_dataset, expert, param=None, seed=None, fold=None, image_container=None, learning_mod="al", prediction_type="right"):

    param_al = param["AL"]
    
    # initialize data, Erhält alle Indizes der Daten
    all_indices = list(range(len(train_dataset.getAllIndices())))
    all_data_filenames = np.array(train_dataset.getAllFilenames())[all_indices]
    all_data_y = np.array(train_dataset.getAllTargets())[all_indices]

    set_seed(seed, fold, text="")

    metrics = {}
    
    Intial_random_set = indices
    indices_labeled  = Intial_random_set
    indices_unlabeled= list(set(all_indices) - set(indices_labeled))

    # Lädt die Datasets für die beschrifteten und unbeschrifteten Daten
    dataset_train_labeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_labeled], all_data_y[indices_labeled], expert.predict , [1]*len(indices_labeled), indices_labeled, 
                                                   param=param, preload=param_al["PRELOAD"], image_container=image_container)
    dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], None , [0]*len(indices_unlabeled), indices_unlabeled, 
                                                     param=param, preload=param_al["PRELOAD"], image_container=image_container)
    dataset_val_unlabeled = NIHExpertDatasetMemory(None, val_dataset.getAllFilenames(), np.array(val_dataset.getAllTargets()), expert.predict , [1]*len(val_dataset.getAllIndices()), 
                                                   val_dataset.getAllIndices(), param=param, preload=param_al["PRELOAD"], image_container=image_container)
    
    # Lädt die Dataloaders
    dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=param_al["BATCH_SIZE"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)
    dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param_al["BATCH_SIZE_VAL"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)    
    dataLoaderValUnlabeled = DataLoader(dataset=dataset_val_unlabeled, batch_size=param_al["BATCH_SIZE_VAL"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)
    
    # train expert model on labeled data
    #model_expert = NetSimple(2, 3, 100, 100, 1000,500).to(device)
    model_folder = param["Parent_PATH"]+"/SSL_Working/NIH/Embedded"
    if param["cluster"]:
        model_folder += f"/Seed_{seed}_Fold_{fold}"
        
    model_expert = ResnetPretrained(2, model_folder, type="50").to(device)
    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), "GPUs!")
        model_expert = nn.DataParallel(model_expert)
    # Trainier Modell um Experten vorherzusagen

    dataloaders = (dataLoaderTrainLabeled, dataLoaderValUnlabeled)
    train_metrics, val_metrics = run_expert(model_expert, param_al["EPOCH_TRAIN"], dataloaders, param=param, id=expert.labelerId, seed=seed, fold=fold, 
                                            n_images=param_al["INITIAL_SIZE"], mod=learning_mod, prediction_type=prediction_type)

    metrics["Train"] = {}
    metrics["Train"][param_al["INITIAL_SIZE"]] = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    met = testExpert(expert, val_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Val", model=model_expert)
    metrics["Val"] = {
        "Start": met,
    }

    met = testExpert(expert, test_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Test", model=model_expert)
    metrics["Test"] = {
        "Start": met,
    }

    for round in range(param_al["ROUNDS"]):

        print(f'\n \n Round {round} \n \n')

        # get points where expert model is least confident on
        indices_confidence = get_least_confident_points(model_expert, dataLoaderTrainUnlabeled, param_al["LABELS_PER_ROUND"], mod=learning_mod)
        indices_labeled  = indices_labeled + list(indices_confidence) 
        indices_unlabeled= list(set(all_indices) - set(indices_labeled))

        dataset_train_labeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_labeled], all_data_y[indices_labeled], expert.predict , [1]*len(indices_labeled), 
                                                       indices_labeled, param=param, preload=param_al["PRELOAD"], image_container=image_container)
        dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], None , [0]*len(indices_unlabeled), 
                                                         indices_unlabeled, param=param, preload=param_al["PRELOAD"], image_container=image_container)

        
        dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=param_al["BATCH_SIZE"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)
        dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param_al["BATCH_SIZE_VAL"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)

        
        # train model on labeled data
        n_images = param_al["INITIAL_SIZE"] + (round+1)*param_al["LABELS_PER_ROUND"]
        dataloaders = (dataLoaderTrainLabeled, dataLoaderValUnlabeled)
        train_metrics, val_metrics = run_expert(model_expert, param_al["EPOCH_TRAIN"], dataloaders, param=param, id=expert.labelerId, seed=seed, fold=fold, n_images = n_images, 
                                                mod=learning_mod, prediction_type=prediction_type)

        metrics["Train"][n_images] = {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }


    dataset_test_unlabeled = NIHExpertDatasetMemory(None, test_dataset.getAllFilenames(), np.array(test_dataset.getAllTargets()), expert.predict , [1]*len(test_dataset.getAllIndices()), 
                                                    test_dataset.getAllIndices(), param=param, preload=param_al["PRELOAD"], image_container=image_container)
    dataLoaderVal = DataLoader(dataset=dataset_test_unlabeled, batch_size=param_al["BATCH_SIZE_VAL"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)
    met_test = metrics_print_expert(model_expert, dataLoaderVal, id=expert.labelerId, seed=seed, fold=fold, 
                               n_images=param_al["INITIAL_SIZE"] + param_al["ROUNDS"]*param_al["LABELS_PER_ROUND"], step="Test", mod=learning_mod, prediction_type=prediction_type, param=param)

    met = testExpert(expert, val_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Val", model=model_expert)
    metrics["Val"]["End"] = met

    met = testExpert(expert, test_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Test", model=model_expert)
    metrics["Test"]["End"] = met

    return model_expert, met_test, metrics

def getExpertModelNormal(indices, train_dataset, val_dataset, test_dataset, expert, param=None, seed=None, fold=None, image_container=None, learning_mod="al", prediction_type="right"):

    param_al = param["AL"]
    
    # initialize data, Erhält alle Indizes der Daten
    all_indices = list(range(len(train_dataset.getAllIndices())))
    all_data_filenames = np.array(train_dataset.getAllFilenames())[all_indices]
    all_data_y = np.array(train_dataset.getAllTargets())[all_indices]

    set_seed(seed, fold, text="")

    metrics = {}
    
    Intial_random_set = indices
    indices_labeled  = Intial_random_set
    indices_unlabeled= list(set(all_indices) - set(indices_labeled))

    # Lädt die Datasets für die beschrifteten und unbeschrifteten Daten
    dataset_train_labeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_labeled], all_data_y[indices_labeled], expert.predict , [1]*len(indices_labeled), indices_labeled, 
                                                   param=param, preload=param_al["PRELOAD"], image_container=image_container)
    dataset_val_unlabeled = NIHExpertDatasetMemory(None, val_dataset.getAllFilenames(), np.array(val_dataset.getAllTargets()), expert.predict , [1]*len(val_dataset.getAllIndices()), 
                                                   val_dataset.getAllIndices(), param=param, preload=param_al["PRELOAD"], image_container=image_container)
    
    # Lädt die Dataloaders
    dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=param_al["BATCH_SIZE"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)    
    dataLoaderValUnlabeled = DataLoader(dataset=dataset_val_unlabeled, batch_size=param_al["BATCH_SIZE_VAL"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)
    
    # train expert model on labeled data
    #model_expert = NetSimple(2, 3, 100, 100, 1000,500).to(device)
    model_folder = param["Parent_PATH"]+"/SSL_Working/NIH/Embedded"
    if param["cluster"]:
        model_folder += f"/Seed_{seed}_Fold_{fold}"
    
    model_expert = ResnetPretrained(2, model_folder, type="50").to(device)
    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), "GPUs!")
        model_expert = nn.DataParallel(model_expert)
    # Trainier Modell um Experten vorherzusagen

    dataloaders = (dataLoaderTrainLabeled, dataLoaderValUnlabeled)
    train_metrics, val_metrics = run_expert(model_expert, param_al["EPOCH_TRAIN"], dataloaders, param=param, id=expert.labelerId, seed=seed, fold=fold, 
                                            n_images=param["LABELED"], mod=learning_mod, prediction_type=prediction_type) 
    metrics["Train"] = {}
    metrics["Train"][param["LABELED"]] = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics
    }
        
    gc.collect()

    dataset_test_unlabeled = NIHExpertDatasetMemory(None, test_dataset.getAllFilenames(), np.array(test_dataset.getAllTargets()), expert.predict , [1]*len(test_dataset.getAllIndices()), 
                                                    test_dataset.getAllIndices(), param=param, preload=param_al["PRELOAD"], image_container=image_container)
    dataLoaderVal = DataLoader(dataset=dataset_test_unlabeled, batch_size=param_al["BATCH_SIZE_VAL"], shuffle=True, num_workers=param["num_worker"], pin_memory=True)
    met_test = metrics_print_expert(model_expert, dataLoaderVal, id=expert.labelerId, seed=seed, fold=fold, 
                               n_images=param["LABELED"], step="Test", mod=learning_mod, prediction_type=prediction_type, param=param)

    met = testExpert(expert, val_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Val", model=model_expert)
    metrics["Val"] = {
        "End": met,
    }

    met = testExpert(expert, test_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Test", model=model_expert)
    metrics["Test"] = {
        "End": met,
    }

    return model_expert, met_test, metrics

def train_expert_confidence(train_loader, optimizer, scheduler, epoch, apply_softmax, param=None, id="", model=None, expert=None, mod=None, prediction_type="target"):
    """Train for one epoch the model to predict expert agreement with label"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    assert ((expert is None) or (model is None)), "Can't pass expert and model"
    assert ((expert is not None) or (model is not None)), "Need model or expert"
    assert (mod is not None) and (mod == "al" or mod == "ssl"), "You have to pass a mod (al or ssl)"
    assert (prediction_type == "target" or prediction_type == "right"), "You have to pass a prediction_type (target or right)"

    #Variables for faster if statements
    mod_al = False
    mod_ssl = False
    
    # switch to train mode
    if mod == "al":
        mod_al = True
        model.train()
    elif mod == "ssl":
        mod_ssl = True
        expert.sslModel.linear_model.train()
    
    

    end = time.time()
    for i, (input, label, expert_pred, _, _, filenames ) in enumerate(train_loader):
        expert_pred = expert_pred.long()
        target, input = expert_pred.to(device), input.to(device)

        if prediction_type == "right":
            expert_pred = (expert_pred == label) *1
        
        if mod_al:
            # compute output
            output = model(input)
        elif mod_ssl:
            features = expert.sslModel.embedded_model.get_embedding(batch=input)
            logits, _ = expert.sslModel.linear_model(features)
            scores = torch.softmax(logits, dim=1)
            #preds = torch.argmax(scores, dim=1).cpu().tolist()
            output = scores
        
        # compute loss
        if apply_softmax:
            loss = my_CrossEntropyLossWithSoftmax(output, target)
        else:
            if prediction_type == "right":
                loss = my_CrossEntropyLoss(output, target, cost=param["AL"]["COST"])
            else:
                loss = my_CrossEntropyLoss(output, target)
        
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        #losses.update(loss.data.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
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
            

def run_expert(model, epochs, dataloaders, apply_softmax = False, param=None, id=0, seed=None, fold=None, n_images=None, expert=None, mod="", prediction_type="target"):
    '''
    train expert model to predict disagreement with label
    model: WideResNet model or pytorch model (2 outputs)
    epochs: number of epochs to train
    '''

    # define loss function (criterion) and optimizer
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    train_loader, val_loader = dataloaders

    assert mod == "al" or mod == "ssl", "Mod should be al or ssl"

    if mod == "ssl":
        optimizer = torch.optim.SGD(expert.sslModel.linear_model.parameters(), 0.001, #0.001
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    elif mod == "al":
        optimizer = torch.optim.SGD(model.parameters(), 0.001, #0.001
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)

    train_metrics = []
    for epoch in range(0, epochs):
        # train for one epoch
        train_expert_confidence(train_loader=train_loader, model=model, expert=expert, optimizer=optimizer, scheduler=scheduler, epoch=epoch, apply_softmax=apply_softmax, param=param, 
                                mod=mod, prediction_type=prediction_type)
        #if epoch % 10 == 0 and epoch != epochs and epoch != 0:
        #   print("Eval")
        #    metrics_print_expert(model, val_loader, id, seed=seed, fold=fold, mod=mod, prediction_type=prediction_type, expert=expert, param=param)
        met = metrics_print_expert(model, data_loader=val_loader, id=id, seed=seed, fold=fold, n_images=n_images, mod=mod, prediction_type=prediction_type, expert=expert, param=param, 
                                   print_result=False, step="Train", epoch=epoch)
        train_metrics.append(met)
            
    val_metrics = metrics_print_expert(model, data_loader=val_loader, id=id, seed=seed, fold=fold, n_images=n_images, expert=expert, mod=mod, prediction_type=prediction_type, param=param, step="Val")

    return train_metrics, val_metrics

def metrics_print_expert(model, data_loader, expert=None, defer_net = False, id=0, seed=None, fold=None, n_images=None, mod="", prediction_type="", param=None, 
                         print_result=True, step="", epoch=None):
    '''
    Computes metrics for expert model error prediction
    model: model
    data_loader: data loader
    '''

    assert (mod is not None) and (mod == "al" or mod == "ssl" or mod == "perfect"), "You have to pass a mod (al or ssl)"
    assert (prediction_type == "target" or prediction_type == "right"), "You have to pass a prediction_type (target or right)"
    if param["NEPTUNE"]["NEPTUNE"]:
        assert step != "", "Need step"
    
    correct = 0
    total = 0
    
    label_list = []
    predictions_list = []
    # again no gradients needed
    cou = 3
    with torch.no_grad():
        for data in data_loader:
            images, label, expert_pred, _ ,_, filenames = data
            expert_pred = expert_pred.long()
            if prediction_type == "right":
                expert_pred = (expert_pred == label) *1
            images, labels = images.to(device), expert_pred.to(device)

            if mod == "al":
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
            elif mod == "ssl":
                features = expert.sslModel.embedded_model.get_embedding(batch=images)
                logits, _ = expert.sslModel.linear_model(features)
                scores = torch.softmax(logits, dim=1)
                predictions = torch.argmax(scores, dim=1)#.cpu()#.tolist()
                #preds, preds2 = torch.max(scores, 1)
                #output = scores
            elif mod == "perfect":
                predictions = expert_pred.to(device)

            if cou == 1:
                print("###########")
                print("Predictions:")
                print(predictions)
                #print(preds)
                #print(preds2)
                cou = 2

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            label_list.extend(labels.cpu().numpy())
            predictions_list.extend(predictions.cpu().numpy())
            

                             
    label_list = np.array(label_list)
    predictions_list = np.array(predictions_list)

    accurancy = 100 * correct / total

    if print_result: 
        print(mod)
        print(prediction_type)
        print('Accuracy of the network on the %d test images: %.3f %%' % (total, accurancy))
    
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(label_list, predictions_list, labels=[0, 1]).ravel()

    f1 = sklearn.metrics.f1_score(label_list, predictions_list)

    ac_balanced = sklearn.metrics.balanced_accuracy_score(label_list, predictions_list)

    met = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accurancy": accurancy,
        "f1": f1,
        "accurancy_balanced": ac_balanced,
    }

    if param["NEPTUNE"]["NEPTUNE"]:
        if step == "Train":
            output = f"Train/Images_{n_images}"
            neptune_steps = epoch
        elif step == "Val":
            output = "Val"
            neptune_steps = n_images
        elif step == "Test":
            output = "Test"
            neptune_steps = n_images

        run = param["NEPTUNE"]["RUN"]

        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/tn"].append(tn, step=neptune_steps)
        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/fp"].append(fp, step=neptune_steps)
        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/fn"].append(fn, step=neptune_steps)
        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/tp"].append(tp, step=neptune_steps)

        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/accuracy"].append(accurancy, step=neptune_steps)

        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/f1"].append(f1, step=neptune_steps)

        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/accuracy_balanced"].append(ac_balanced, step=neptune_steps)

    if print_result:
        print("Confusion Matrix:")
        print(sklearn.metrics.confusion_matrix(label_list, predictions_list, labels=[0, 1]))
        print("F1 Score: " + str(f1))

        print("Accuracy balanced")
        print(ac_balanced)

    return met

def testExpert(expert, dataset, image_container, param, mod, prediction_type, seed, fold, data_name, model=None):
    
    #test_dataset = ds.NIHDataset(dataset, preload=False, preprocess=False, param=param, image_container=image_container)
    
    final_dataset = NIHExpertDatasetMemory(None, dataset.getAllFilenames(), np.array(dataset.getAllTargets()), expert.predict , [1]*len(dataset.getAllIndices()), 
                                                       dataset.getAllIndices(), param=param, preload=True, image_container=image_container)

    data_loader = DataLoader(dataset=final_dataset, batch_size=128, shuffle=True, num_workers=param["num_worker"], pin_memory=True)

    if param["NEPTUNE"]["NEPTUNE"]:
        run = param["NEPTUNE"]["RUN"]
        param["NEPTUNE"]["RUN"] = None
        param_neptune_off = copy.deepcopy(param)
        param["NEPTUNE"]["RUN"] = run
        param_neptune_off["NEPTUNE"]["NEPTUNE"] = False
    else:
        param_neptune_off = copy.deepcopy(param)
    if mod == "al":
        metrics = metrics_print_expert(model=model, data_loader=data_loader, expert=None, id=expert.labelerId, mod=mod, prediction_type=prediction_type, param=param_neptune_off, 
                                       print_result=False)
    elif mod == "ssl":
        metrics = metrics_print_expert(model=None, data_loader=data_loader, expert=expert, id=expert.labelerId, mod=mod, prediction_type=prediction_type, param=param_neptune_off, 
                                       print_result=False)
    elif mod == "perfect":
        metrics = metrics_print_expert(model=None, data_loader=data_loader, expert=expert, id=expert.labelerId, mod=mod, prediction_type=prediction_type, param=param_neptune_off, 
                                       print_result=False)

    if param["NEPTUNE"]["NEPTUNE"]:
        output = data_name + "_Start_End"
        run = param["NEPTUNE"]["RUN"]
        id=expert.labelerId
        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/tn"].append(metrics["tn"])
        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/fp"].append(metrics["fp"])
        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/fn"].append(metrics["fn"])
        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/tp"].append(metrics["tp"])

        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/accuracy"].append(metrics["accurancy"])

        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/f1"].append(metrics["f1"])

        run[f"Seed_{seed}/Fold_{fold}/Expert_{id}/" + output + "/accuracy_balanced"].append(metrics["accurancy_balanced"])

    return metrics