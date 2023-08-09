#!/usr/bin/env python
# coding: utf-8

import copy
import json
import math
import os
import random
import shutil
import time
import pandas as pd
import time

import pickle
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
import sys

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

import Dataset.Dataset as ds

import ssl_functions as ssl
import active_learning as al
from active_learning import NIHExpertDatasetMemory

import expert as expert_module
import verma as verm
import hemmer as hm

import neptune

import json
import shutil


# In[2]:


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


# In[3]:




with open('neptune_config.json', 'r') as f:
    config = json.load(f)

config_neptune = config["neptune"]


# In[4]:



def cleanTrainDir(path):
    shutil.rmtree(path)


# In[21]:


def getExpertModelSSL_AL(dataManager, expert, labelerId, param=None, seed=None, fold=None, learning_mod="ssl", prediction_type="target"):

    if param["SETTING"] == "SSL_AL_SSL":
        learning_type = "ssl"
    elif param["SETTING"] == "SSL_AL":
        learning_type = "sl"

    assert learning_type != "", "Need to define how experts should be trained with new AL data (sl or ssl)"
    
    nih_dataloader = dataManager.getKFoldDataloader(seed)

    expert_train, expert_val, expert_test = nih_dataloader.get_dataset_for_folder(fold)
    image_container = nih_dataloader.get_ImageContainer()
    train_dataset = ds.NIHDataset(expert_train, preload=False, preprocess=False, param=param, image_container=image_container)
    val_dataset = ds.NIHDataset(expert_val, preload=False, preprocess=False, param=param, image_container=image_container)
    test_dataset = ds.NIHDataset(expert_test, preload=False, preprocess=False, param=param, image_container=image_container)

    sslDataset = dataManager.getSSLDataset(seed)
    usedFilenames = sslDataset.getLabeledFilenames(labelerId, fold)
    
    # initialize data, Erhält alle Indizes der Daten
    all_indices = list(range(len(train_dataset.getAllIndices())))
    all_data_filenames = np.array(train_dataset.getAllFilenames())[all_indices]
    all_data_y = np.array(train_dataset.getAllTargets())[all_indices]

    used_indices = [index for index in all_indices if all_data_filenames[index] in usedFilenames]
    indices = used_indices

    print("Len overlapping used indices: " + str(len(used_indices)))

    metrics = {}

    met = al.testExpert(expert, val_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Val")
    metrics["Val"] = {
        "Start": met,
    }

    met = al.testExpert(expert, test_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Test")
    metrics["Test"] = {
        "Start": met,
    }

    metrics["Train"] = {}

    set_seed(seed, fold, text="")
    
    Intial_random_set = indices
    indices_labeled  = Intial_random_set
    indices_unlabeled= list(set(all_indices) - set(indices_labeled))

    # Lädt die Datasets für die beschrifteten und unbeschrifteten Daten
    dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], None , [0]*len(indices_unlabeled), indices_unlabeled, param=param, preload=param["AL"]["PRELOAD"], image_container=image_container)
    dataset_val_unlabeled = NIHExpertDatasetMemory(None, val_dataset.getAllFilenames(), np.array(val_dataset.getAllTargets()), expert.predict , [1]*len(val_dataset.getAllIndices()), val_dataset.getAllIndices(), param=param, preload=param["AL"]["PRELOAD"], image_container=image_container)
    
    # Lädt die Dataloaders
    dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param["AL"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    dataLoaderValUnlabeled = DataLoader(dataset=dataset_val_unlabeled, batch_size=param["AL"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    
    for round in range(param["AL"]["ROUNDS"]):

        print(f'\n \n Round {round} \n \n')

        # get points where expert model is least confident on
        indices_confidence = al.get_least_confident_points(expert, dataLoaderTrainUnlabeled, param["AL"]["LABELS_PER_ROUND"], mod="ssl")
        indices_labeled  = indices_labeled + list(indices_confidence) 
        indices_unlabeled= list(set(all_indices) - set(indices_labeled))

        dataset_train_labeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_labeled], all_data_y[indices_labeled], expert.predict , [1]*len(indices_labeled), indices_labeled, param=param, preload=param["AL"]["PRELOAD"], image_container=image_container)
        dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], expert.predict , [0]*len(indices_unlabeled), indices_unlabeled, param=param, preload=param["AL"]["PRELOAD"], image_container=image_container)

        dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=param["AL"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
        dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param["AL"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)

        if learning_type == "ssl": #If the experts should be trained with ssl
            sslDataset = dataManager.getSSLDataset(seed)
            sslDataset.addNewLabels(all_data_filenames[list(indices_confidence)], fold, expert.labelerId)
            emb_model, model = ssl.getExpertModelSSL(labelerId=expert.labelerId, sslDataset=sslDataset, seed=seed, fold_idx=fold, n_labeled=None, embedded_model=None, param=param, neptune_param=param["NEPTUNE"], added_epochs=(round+1)*param["AL"]["SSL_EPOCHS"])
            expert.setModel(expert_module.SSLModel(emb_model, model), mod="SSL")


            #TODO: Test experts and get metrics
            n_images = param["AL"]["INITIAL_SIZE"] + (round+1)*param["AL"]["LABELS_PER_ROUND"]

            train_metrics = al.metrics_print_expert(model=None, expert=expert, data_loader=dataLoaderTrainLabeled, id=expert.labelerId, seed=seed, fold=fold, n_images=n_images, step="Train", param=param, mod="ssl", prediction_type="target", print_result=False)
            val_metrics = al.metrics_print_expert(model=None, expert=expert, data_loader=dataLoaderValUnlabeled, id=expert.labelerId, seed=seed, fold=fold, n_images=n_images, step="Val", param=param, mod="ssl", prediction_type="target")

            metrics["Train"][n_images] = {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

        elif learning_type == "sl": #supervised learning
        
            # train model on labeled data
            dataloaders = (dataLoaderTrainLabeled, dataLoaderValUnlabeled)
            n_images = param["AL"]["INITIAL_SIZE"] + (round+1)*param["AL"]["LABELS_PER_ROUND"]
            train_metrics, val_metrics = al.run_expert(model=None, expert=expert, epochs=param["AL"]["EPOCH_TRAIN"], dataloaders=dataloaders, param=param, id=expert.labelerId, seed=seed, fold=fold, n_images=n_images, mod="ssl", prediction_type="target")
        
            metrics["Train"][n_images] = {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
    
    dataset_test_unlabeled = NIHExpertDatasetMemory(None, test_dataset.getAllFilenames(), np.array(test_dataset.getAllTargets()), expert.predict , [1]*len(test_dataset.getAllIndices()), test_dataset.getAllIndices(), param=param, preload=param["AL"]["PRELOAD"], image_container=image_container)
    dataLoaderVal = DataLoader(dataset=dataset_test_unlabeled, batch_size=param["AL"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    met_test = al.metrics_print_expert(model=None, expert=expert, data_loader=dataLoaderVal, id=expert.labelerId, seed=seed, fold=fold, n_images=param["AL"]["INITIAL_SIZE"] + param["AL"]["ROUNDS"]*param["AL"]["LABELS_PER_ROUND"], step="Test", param=param, mod="ssl", prediction_type="target")

    met = al.testExpert(expert, val_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Val")
    metrics["Val"]["End"] = met

    met = al.testExpert(expert, test_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Test")
    metrics["Test"]["End"] = met
    
    #metrics["Test"] = met
    print("AL finished")
    return met_test, metrics


# In[6]:


def getExpertModelsSSL_AL(dataManager, experts, param, seed, fold, learning_mod="ssl", prediction_type="target"):

    if param["SETTING"] == "SSL_AL_SSL":
        learning_type = "ssl"
    elif param["SETTING"] == "SSL_AL":
        learning_type = "sl"

    assert learning_type != "", "Need to define how experts should be trained with new AL data (sl or ssl)"

    nih_dataloader = dataManager.getKFoldDataloader(seed)

    expert_train, expert_val, expert_test = nih_dataloader.get_dataset_for_folder(fold)
    image_container = nih_dataloader.get_ImageContainer()
    train_dataset = ds.NIHDataset(expert_train, preload=False, preprocess=False, param=param, image_container=image_container)
    val_dataset = ds.NIHDataset(expert_val, preload=False, preprocess=False, param=param, image_container=image_container)
    test_dataset = ds.NIHDataset(expert_test, preload=False, preprocess=False, param=param, image_container=image_container)

    sslDataset = dataManager.getSSLDataset(seed)
    usedFilenames = []
    for labelerId in param["LABELER_IDS"]:
        temp = usedFilenames + sslDataset.getLabeledFilenames(labelerId, fold)
    usedFilenames = temp
    
    
    # initialize data, Erhält alle Indizes der Daten
    all_indices = list(range(len(train_dataset.getAllIndices())))
    all_data_filenames = np.array(train_dataset.getAllFilenames())[all_indices]
    all_data_y = np.array(train_dataset.getAllTargets())[all_indices]

    unused_indices = [index for index in all_indices if all_data_filenames[index] not in usedFilenames]
    
    metrics = {}
    for labelerId, expert in experts.items():
        metrics[labelerId] = {}

        met = al.testExpert(expert, val_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Val")
        metrics[labelerId]["Val"] = {
            "Start": met,
        }

        met = al.testExpert(expert, test_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Test")
        metrics[labelerId]["Test"] = {
            "Start": met,
        }

        metrics[labelerId]["Train"] = {}

    set_seed(seed, fold, text="")

    gc.collect()

    indices_unlabeled = unused_indices
    indices_labeled = list(set(all_indices) - set(indices_unlabeled))

    dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], experts[param["LABELER_IDS"][0]].predict , [0]*len(indices_unlabeled), indices_unlabeled, param=param, preload=param["AL"]["PRELOAD"], image_container=image_container)
    dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param["AL"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    
    for round in range(param["AL"]["ROUNDS"]):

        print(f'\n \n Round {round} \n \n')

        #Try to get better Points
        if param["MOD"] == "disagreement":
            indices_qbq = al.getQbQPoints(experts, dataLoaderTrainUnlabeled, param["AL"]["LABELS_PER_ROUND"], mod="ssl", param=param)
        if param["MOD"] == "disagreement_diff":
            indices_qbq = al.getQbQPointsDifference(experts, dataLoaderTrainUnlabeled, param["AL"]["LABELS_PER_ROUND"], mod="ssl", param=param)
        
        #indices_labeled  = indices_labeled + list(indices_confidence) 
        indices_labeled  = indices_labeled + list(indices_qbq) 
        indices_unlabeled= list(set(all_indices) - set(indices_labeled))     
        
        # train model on labeled data
        for labelerId, expert in experts.items():

            #Val Dataset, needed for SSL and AL
            dataset_val_unlabeled = NIHExpertDatasetMemory(None, val_dataset.getAllFilenames(), np.array(val_dataset.getAllTargets()), expert.predict , [1]*len(val_dataset.getAllIndices()), val_dataset.getAllIndices(), param=param, preload=param["AL"]["PRELOAD"], image_container=image_container)
            dataLoaderValUnlabeled = DataLoader(dataset=dataset_val_unlabeled, batch_size=param["AL"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)

            #Create train dataset
            dataset_train_labeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_labeled], all_data_y[indices_labeled], expert.predict , [1]*len(indices_labeled), indices_labeled, param=param, preload=param["AL"]["PRELOAD"], image_container=image_container)
            dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=param["AL"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)

            if learning_type == "ssl": #If the experts should be trained with ssl
                sslDataset = dataManager.getSSLDataset(seed)
                sslDataset.addNewLabels(all_data_filenames[list(indices_qbq)], fold, labelerId)
                emb_model, model = ssl.getExpertModelSSL(labelerId=labelerId, sslDataset=sslDataset, seed=seed, fold_idx=fold, n_labeled=None, embedded_model=None, param=param, neptune_param=param["NEPTUNE"], added_epochs=(round+1)*param["AL"]["SSL_EPOCHS"])
                experts[labelerId].setModel(expert_module.SSLModel(emb_model, model), mod="SSL")


                #TODO: Test experts and get metrics
                n_images = param["AL"]["INITIAL_SIZE"] + (round+1)*param["AL"]["LABELS_PER_ROUND"]

                train_metrics = al.metrics_print_expert(model=None, expert=expert, data_loader=dataLoaderTrainLabeled, id=expert.labelerId, seed=seed, fold=fold, n_images=n_images, step="Train", param=param, mod="ssl", prediction_type="target")
                val_metrics = al.metrics_print_expert(model=None, expert=expert, data_loader=dataLoaderValUnlabeled, id=expert.labelerId, seed=seed, fold=fold, n_images=n_images, step="Val", param=param, mod="ssl", prediction_type="target")

                metrics[labelerId]["Train"][n_images] = {
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }

                
            elif learning_type == "sl": # If the experts sould be trained with supervised learning

                dataloaders = (dataLoaderTrainLabeled, dataLoaderValUnlabeled)
                n_images = param["AL"]["INITIAL_SIZE"] + (round+1)*param["AL"]["LABELS_PER_ROUND"]
                train_metrics, val_metrics = al.run_expert(model=None, expert=expert, epochs=param["AL"]["EPOCH_TRAIN"], dataloaders=dataloaders, param=param, id=expert.labelerId, seed=seed, fold=fold, n_images=n_images, mod="ssl", prediction_type="target")

                metrics[labelerId]["Train"][n_images] = {
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics
                }
        
        dataset_train_unlabeled = NIHExpertDatasetMemory(None, all_data_filenames[indices_unlabeled], all_data_y[indices_unlabeled], expert.predict , [0]*len(indices_unlabeled), indices_unlabeled, param=param, preload=param["AL"]["PRELOAD"], image_container=image_container)
        dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=param["AL"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    
    dataset_test_unlabeled = NIHExpertDatasetMemory(None, test_dataset.getAllFilenames(), np.array(test_dataset.getAllTargets()), expert.predict , [1]*len(test_dataset.getAllIndices()), test_dataset.getAllIndices(), param=param, preload=param["AL"]["PRELOAD"], image_container=image_container)
    dataLoaderVal = DataLoader(dataset=dataset_test_unlabeled, batch_size=param["AL"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    met_test = {}
    for labelerId, expert in experts.items():
        temp = al.metrics_print_expert(model=None, expert=expert, data_loader=dataLoaderVal, id=expert.labelerId, seed=seed, fold=fold, n_images=param["AL"]["INITIAL_SIZE"] + param["AL"]["ROUNDS"]*param["AL"]["LABELS_PER_ROUND"], step="Test", param=param, mod="ssl", prediction_type="target")
        met_test[expert.labelerId] = temp

        met = al.testExpert(expert, val_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Val")
        metrics[labelerId]["Val"]["End"] = met

        met = al.testExpert(expert, test_dataset, image_container, param, learning_mod, prediction_type, seed, fold, data_name="Test")
        metrics[labelerId]["Test"]["End"] = met
        
    return met_test, metrics

# In[7]:


def getExpertsSSL_AL(dataManager, param, fold, seed):

    mod = "ssl"
    prediction_type = param["EXPERT_PREDICT"]

    sslDataset = dataManager.getSSLDataset(seed)

    sslDataset.createLabeledIndices(labelerIds=param["LABELER_IDS"], n_L=param["AL"]["INITIAL_SIZE"], k=round(param["AL"]["INITIAL_SIZE"]*param["OVERLAP"]/100), seed=seed, sample_equal=param["SAMPLE_EQUAL"])

    train_dataloader, val_dataloader, test_dataloader = sslDataset.get_data_loader_for_fold(fold)
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)

    embedded_model = ssl.create_embedded_model(dataloaders, param, param["NEPTUNE"], fold=fold, seed=seed)

    experts = {}
    for labelerId in param["LABELER_IDS"]:
        nih_expert = expert_module.Expert(dataset = dataManager.getBasicDataset(), labeler_id=labelerId)
        emb_model, model = ssl.getExpertModelSSL(labelerId=labelerId, sslDataset=sslDataset, seed=seed, fold_idx=fold, n_labeled=None, embedded_model=embedded_model, param=param, neptune_param=param["NEPTUNE"])
        nih_expert.setModel(expert_module.SSLModel(emb_model, model), mod="SSL")
        experts[labelerId] = nih_expert
    metrics = {}
    if param["MOD"] == "confidence":
        for i, labelerId in enumerate(param["LABELER_IDS"]):
            met, metrics_return = getExpertModelSSL_AL(dataManager=dataManager, expert=experts[labelerId], labelerId=labelerId, param=param, seed=seed, fold=fold, learning_mod="ssl", prediction_type=param["EXPERT_PREDICT"])
            metrics[labelerId] = metrics_return
    elif param["MOD"] == "disagreement" or param["MOD"] == "disagreement_diff":
        met, metrics = getExpertModelsSSL_AL(dataManager, experts, param, seed, fold, learning_mod="ssl", prediction_type=param["EXPERT_PREDICT"])
        
    return experts, metrics


# In[8]:


def getExpertsSSL(dataManager, param, fold, seed):

    sslDataset = dataManager.getSSLDataset(seed)

    mod = "ssl"
    prediction_type = param["EXPERT_PREDICT"]

    sslDataset.createLabeledIndices(labelerIds=param["LABELER_IDS"], n_L=param["LABELED"], k=round(param["LABELED"]*param["OVERLAP"]/100), seed=seed, sample_equal=param["SAMPLE_EQUAL"])

    train_dataloader, val_dataloader, test_dataloader = sslDataset.get_data_loader_for_fold(fold)
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)

    ssl.create_embedded_model(dataloaders, param, param["NEPTUNE"], fold=fold, seed=seed)

    torch.cuda.empty_cache()
    gc.collect()

    experts = {}
    for labelerId in param["LABELER_IDS"]:
        nih_expert = expert_module.Expert(dataset = dataManager.getBasicDataset(), labeler_id=labelerId)
        emb_model, model = ssl.getExpertModelSSL(labelerId=labelerId, sslDataset=sslDataset, seed=seed, fold_idx=fold, n_labeled=None, embedded_model=None, param=param, neptune_param=param["NEPTUNE"])
        nih_expert.setModel(expert_module.SSLModel(emb_model, model), mod="SSL")
        experts[labelerId] = nih_expert

    nih_dataloader = dataManager.getKFoldDataloader(seed)
    expert_train, expert_val, expert_test = nih_dataloader.get_dataset_for_folder(fold)
    image_container = nih_dataloader.get_ImageContainer()

    val_dataset = ds.NIHDataset(expert_val, preload=False, preprocess=False, param=param, image_container=image_container)
    test_dataset = ds.NIHDataset(expert_test, preload=False, preprocess=False, param=param, image_container=image_container)

    metrics = {}
    for labelerId, expert in experts.items():
        metrics[labelerId] = {}

        met = al.testExpert(expert, val_dataset, image_container, param, mod, prediction_type, seed, fold, data_name="Val")
        metrics[labelerId]["Val"] = {
            "End": met,
        }

        met = al.testExpert(expert, test_dataset, image_container, param, mod, prediction_type, seed, fold, data_name="Test")
        metrics[labelerId]["Test"] = {
            "End": met
        }
        
    return experts, metrics


# In[9]:


def setupEmbeddedModel(dataManager, param, fold, seed):
    sslDataset = dataManager.getSSLDataset(seed)

    mod = "ssl"
    prediction_type = param["EXPERT_PREDICT"]

    train_dataloader, val_dataloader, test_dataloader = sslDataset.get_data_loader_for_fold(fold)
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)

    ssl.create_embedded_model(dataloaders, param, param["NEPTUNE"], fold=fold, seed=seed)

    torch.cuda.empty_cache()
    gc.collect()


# In[10]:


def getExpertsAL(dataManager, param, fold_idx, seed):
    nih_dataloader = dataManager.getKFoldDataloader(seed)
    expert_train, expert_val, expert_test = nih_dataloader.get_dataset_for_folder(fold_idx)
    image_container = nih_dataloader.get_ImageContainer()
    expert_train_dataset = ds.NIHDataset(expert_train, preload=False, preprocess=False, param=param, image_container=image_container)
    expert_val_dataset = ds.NIHDataset(expert_val, preload=False, preprocess=False, param=param, image_container=image_container)
    expert_test_dataset = ds.NIHDataset(expert_test, preload=False, preprocess=False, param=param, image_container=image_container)

    setupEmbeddedModel(dataManager, param, fold_idx, seed)
    #Get init labeled indices with k same images and n-k different images
    #k=None means random indieces
    k = param["OVERLAP"]
    all_indices = list(range(len(expert_train_dataset.getAllIndices())))
    #If no k is set than it selects one randomly
    k = round(param["AL"]["INITIAL_SIZE"]*k/100)
    if param["NEPTUNE"]["NEPTUNE"]:
        run["param/overlap_k"] = k
    indices = al.sampleIndices(n = param["AL"]["INITIAL_SIZE"], k = k, all_indices = all_indices, experten = list(param["LABELER_IDS"]), seed = seed, fold=fold_idx)

    if param["NEPTUNE"]["NEPTUNE"]:
        run[f"Seed_{seed}/Fold_{fold_idx}/Experts/Indices"] = indices

    print("Random indices:")
    print(indices)

    experts = {}
    metrics = {}
    for i, labelerId in enumerate(list(param["LABELER_IDS"])):
        nih_expert = expert_module.Expert(dataset = dataManager.getBasicDataset(), labeler_id=labelerId)
        experts[labelerId] = nih_expert
        if param["MOD"] == "confidence":
            expert_model, met_test, metric = al.getExpertModel(indices[labelerId], expert_train_dataset, expert_val_dataset, expert_test_dataset, nih_expert, param, seed, fold_idx, image_container=image_container, learning_mod="al", prediction_type=param["EXPERT_PREDICT"])
            nih_expert.setModel(expert_model, mod="AL")
            metrics[labelerId] = metric
    if param["MOD"] == "disagreement" or param["MOD"]=="disagreement_diff":
        expert_models, met, metrics = al.getExpertModels(indices, experts, expert_train_dataset, expert_val_dataset, expert_test_dataset, param, seed, fold_idx, mod=param["MOD"], image_container=image_container, learning_mod="al", prediction_type=param["EXPERT_PREDICT"])
        for labelerId, expert in experts.items():
            expert.setModel(expert_models[labelerId], mod="AL")

    return experts, metrics


# In[11]:


def getExpertsNormal(dataManager, param, fold_idx, seed):
    nih_dataloader = dataManager.getKFoldDataloader(seed)
    expert_train, expert_val, expert_test = nih_dataloader.get_dataset_for_folder(fold_idx)
    image_container = nih_dataloader.get_ImageContainer()
    expert_train_dataset = ds.NIHDataset(expert_train, preload=False, preprocess=False, param=param, image_container=image_container)
    expert_val_dataset = ds.NIHDataset(expert_val, preload=False, preprocess=False, param=param, image_container=image_container)
    expert_test_dataset = ds.NIHDataset(expert_test, preload=False, preprocess=False, param=param, image_container=image_container)

    setupEmbeddedModel(dataManager, param, fold_idx, seed)
    
    #Get init labeled indices with k same images and n-k different images
    #k=None means random indieces
    k = param["OVERLAP"]
    all_indices = list(range(len(expert_train_dataset.getAllIndices())))
    #If no k is set than it selects one randomly
    k = round(param["LABELED"]*k/100)
    if param["NEPTUNE"]["NEPTUNE"]:
        run["param/overlap_k"] = k
    indices = al.sampleIndices(n = param["LABELED"], k = k, all_indices = all_indices, experten = list(param["LABELER_IDS"]), seed = seed, fold=fold_idx)

    if param["NEPTUNE"]["NEPTUNE"]:
        run[f"Seed_{seed}/Fold_{fold_idx}/Experts/Indices"] = indices

    print("Random indices:")
    print(indices)

    experts = {}
    #Create the experts
    metrics = {}
    for i, labelerId in enumerate(list(param["LABELER_IDS"])):
        nih_expert = expert_module.Expert(dataset = dataManager.getBasicDataset(), labeler_id=labelerId)
        experts[labelerId] = nih_expert

        model, met, metric = al.getExpertModelNormal(indices[labelerId], expert_train_dataset, expert_val_dataset, expert_test_dataset, nih_expert, param, seed, fold_idx, image_container=image_container, learning_mod="al", prediction_type=param["EXPERT_PREDICT"])
        nih_expert.setModel(model, mod="AL")
        metrics[labelerId] = metric

    return experts, metrics


# In[12]:


def getExpertsPerfect(dataManager, param, fold, seed):

    experts = {}
    for i, labelerId in enumerate(list(param["LABELER_IDS"])):
        nih_expert = expert_module.Expert(dataset = dataManager.getBasicDataset(), labeler_id=labelerId)
        experts[labelerId] = nih_expert


    sslDataset = dataManager.getSSLDataset(seed)

    mod = "perfect"
    prediction_type = param["EXPERT_PREDICT"]

    torch.cuda.empty_cache()
    gc.collect()

    nih_dataloader = dataManager.getKFoldDataloader(seed)
    expert_train, expert_val, expert_test = nih_dataloader.get_dataset_for_folder(fold)
    image_container = nih_dataloader.get_ImageContainer()

    val_dataset = ds.NIHDataset(expert_val, preload=False, preprocess=False, param=param, image_container=image_container)
    test_dataset = ds.NIHDataset(expert_test, preload=False, preprocess=False, param=param, image_container=image_container)

    metrics = {}
    for labelerId, expert in experts.items():
        metrics[labelerId] = {}

        met = al.testExpert(expert, val_dataset, image_container, param, mod, prediction_type, seed, fold, data_name="Val")
        metrics[labelerId]["Val"] = {
            "End": met,
        }

        met = al.testExpert(expert, test_dataset, image_container, param, mod, prediction_type, seed, fold, data_name="Test")
        metrics[labelerId]["Test"] = {
            "End": met
        }

    return experts, metrics


# In[13]:


def getExperts(dataManager, param, seed, fold):
      
    #Creates expert models for the choosen method
    if param["SETTING"] == "PERFECT":
        experts, metrics = getExpertsPerfect(dataManager, param, fold, seed)
    if param["SETTING"] == "AL":
        experts, metrics = getExpertsAL(dataManager, param, fold, seed)
    elif param["SETTING"] == "SSL":
        experts, metrics = getExpertsSSL(dataManager, param, fold, seed)
    elif param["SETTING"] == "SSL_AL" or param["SETTING"] == "SSL_AL_SSL":
        experts, metrics = getExpertsSSL_AL(dataManager, param, fold, seed)
    elif param["SETTING"] == "NORMAL":
        experts, metrics = getExpertsNormal(dataManager, param, fold, seed)

    return experts, metrics


# In[14]:


def L2D_Verma(train_loader, val_loader, test_loader, full_dataloader, expert_fns, param, seed, fold_idx, experts):
    num_experts = len(expert_fns)
            
    model = model = vres.ResNet50_defer(int(param["n_classes"]) + num_experts)

    metrics_train_all, metrics_val_all, metrics_test, metrics_full, metrics_pretrain_all = verm.train(model, train_loader, val_loader, test_loader, expert_fns, param, seed=seed, experts=experts, 
                                                                                fold=fold_idx, full_dataloader=full_dataloader, param=param)

    return metrics_train_all, metrics_val_all, metrics_test, metrics_full, metrics_pretrain_all


# In[15]:


def one_run(dataManager, run_param):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expert_metrics = {}
    verma_metrics = {}
    hemmer_metrics = {}

    for seed in run_param["SEEDS"]:
        print("Seed: " + str(seed))
        

        expert_metrics[seed] = {}
        verma_metrics[seed] = {}
        hemmer_metrics[seed] = {}

        for fold_idx in range(run_param["K"]):
        #for fold_idx in range(4):

            if os.path.isdir(f'{run_param["Parent_PATH"]}/SSL_Working'):
                cleanTrainDir(f'{run_param["Parent_PATH"]}/SSL_Working')

            if seed != "":
                set_seed(seed, fold_idx, text="")

            print("/n")
            print(f"Seed: {seed} - Fold: {fold_idx} \n")

            if os.path.isdir(f'{run_param["Parent_PATH"]}/SSL_Working/NIH/EmbeddingCM_bin'):
                cleanTrainDir(f'{run_param["Parent_PATH"]}SSL_Working/NIH/EmbeddingCM_bin')

            neptune = {
                "SEED": seed,
                "FOLD": fold_idx,
            }

            torch.cuda.empty_cache()
            gc.collect()

            experts, expert_metric = getExperts(dataManager, run_param, seed, fold_idx)
            expert_metrics[seed][fold_idx] = expert_metric

            torch.cuda.empty_cache()
            gc.collect()

            #print(f"Got {len(experts)} experts")

            nih_dataloader = dataManager.getKFoldDataloader(seed=seed)

            train_loader, val_loader, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)
            full_dataloader = nih_dataloader.getFullDataloader()

            expert_fns = []
            print(run_param["SETTING"])
            for labelerId, expert in experts.items():
                if run_param["SETTING"] == "AL":
                    expert.init_model_predictions(full_dataloader, mod="AL")
                    expert_fns.append(expert.predict_model_predefined_al)
                elif run_param["SETTING"] == "SSL":
                    expert.init_model_predictions(full_dataloader, mod="SSL")
                    expert_fns.append(expert.predict_model_predefined_ssl)
                elif (run_param["SETTING"] == "SSL_AL" or run_param["SETTING"] == "SSL_AL_SSL"):
                    expert.init_model_predictions(full_dataloader, mod="SSL")
                    expert_fns.append(expert.predict_model_predefined_ssl)
                elif run_param["SETTING"] == "NORMAL":
                    expert.init_model_predictions(full_dataloader, mod="AL")
                    expert_fns.append(expert.predict_model_predefined_al)
                elif run_param["SETTING"] == "PERFECT":
                    expert_fns.append(expert.predict)

            metrics_train_all, metrics_val_all, metrics_test_all, metrics_full_all, metrics_pretrain_all = L2D_Verma(train_loader, val_loader, test_loader, full_dataloader, expert_fns, run_param, seed, fold_idx, experts=experts)

            verma_metrics[seed][fold_idx] = {
                "train": metrics_train_all,
                "val": metrics_val_all,
                "test": metrics_test_all,
                "full": metrics_full_all,
                "pretrain": metrics_pretrain_all,
            }
            
            system_accuracy, classifier_coverage, all_train_metrics, all_val_metrics, all_test_metrics, all_full_metrics = hm.L2D_Hemmer(train_loader, val_loader, test_loader, full_dataloader, expert_fns, run_param, seed, fold_idx, experts)

            hemmer_metrics[seed][fold_idx] = {
                "train": all_train_metrics,
                "val": all_val_metrics,
                "test": all_test_metrics,
                "full": all_full_metrics,
            }

            
    return expert_metrics, verma_metrics, hemmer_metrics
            


# In[16]:


def run_experiment(param):
    run_param = copy.deepcopy(param)

    runs = None

    expert_metrics_all = []

    #with open('Metrics_Folder/Metrics_97.pickle', 'rb') as handle:
    #    expert_metrics_all = pickle.load(handle)

    #runs = [{i:run[i] for i in run if i not in ["expert metrics", "verma", "hemmer"]} for run in expert_metrics_all]

    count = 0

    #Every pair of labeler ids
    for labeler_ids in param["LABELER_IDS"]:
        run_param["LABELER_IDS"] = labeler_ids
        run_param["labeler_ids"] = convert_ids_to_string(labeler_ids)
        

        dataManager = ds.DataManager(path=param["PATH"], target=param["TARGET"], param=run_param, seeds=param["SEEDS"])
        dataManager.createData()

        for init_size in param["AL"]["INITIAL_SIZE"]:
            run_param["AL"]["INITIAL_SIZE"] = init_size

            for labels_per_round in param["AL"]["LABELS_PER_ROUND"]:
                run_param["AL"]["LABELS_PER_ROUND"] = labels_per_round

                for rounds in param["AL"]["ROUNDS"]:
                    run_param["AL"]["ROUNDS"] = rounds

                    labeled = init_size + rounds * labels_per_round

                    run_param["LABELED"] = labeled

                    if (labeled >= 128): #Prevents from large amount of data
                        continue

                    for cost in param["AL"]["COST"]:
                        run_param["AL"]["COST"] = cost
                        run_param["AL"]["cost"] = convert_cost_to_string(cost)

                        for overlap in param["OVERLAP"]:
                            run_param["OVERLAP"] = overlap

                            for setting in param["SETTING"]:
                                run_param["SETTING"] = setting
                        
                                for mod in param["MOD"]:
                                    run_param["MOD"] = mod

                                    if ((setting == "AL"  or setting=="SSL_AL" or setting=="SSL_AL_SSL") and (mod not in ["confidence", "disagreement", "disagreement_diff"])):
                                        continue

                                    if (setting == "SSL" and mod != "ssl"):
                                        continue

                                    if (setting == "NORMAL" and mod != "normal"):
                                        continue

                                    for expert_predict in param["EXPERT_PREDICT"]:
                                        run_param["EXPERT_PREDICT"] = expert_predict

                                        if ((setting == "SSL" or setting == "SSL_AL" or setting == "SSL_AL_SSL") and (expert_predict == "right")):
                                            continue

                                        if (expert_predict == "target") and (cost != param["AL"]["COST"][0]):
                                            continue
                                        if (expert_predict == "target"):
                                            run_param["AL"]["cost"] = convert_cost_to_string((0, 0))

                                        for sample_equal in param["SAMPLE_EQUAL"]:
                                            run_param["SAMPLE_EQUAL"] = sample_equal

                                            for epochs_pretrain in param["epochs_pretrain"]:
                                                run_param["epochs_pretrain"] = epochs_pretrain

                                                metrics_save = {}
                                                metrics_save["labeler_ids"] = labeler_ids
                                                metrics_save["init_size"] = init_size
                                                metrics_save["labels_per_round"] = labels_per_round
                                                metrics_save["rounds"] = rounds
                                                metrics_save["labeled"] = labeled
                                                metrics_save["cost"] = cost
                                                metrics_save["overlap"] = overlap
                                                metrics_save["setting"] = setting
                                                metrics_save["mod"] = mod
                                                metrics_save["expert_predict"] = expert_predict
                                                metrics_save["sample_equal"] = sample_equal
                                                metrics_save["epochs_pretrain"] = epochs_pretrain

                                                if runs is not None:
                                                    if metrics_save in runs:
                                                        continue
                                    
                            
                                                NEPTUNE = param["NEPTUNE"]["NEPTUNE"]
                                                if param["NEPTUNE"]["NEPTUNE"]:
                                                    global run
                                                    run = neptune.init_run(
                                                        project=config_neptune["project"],
                                                        api_token=config_neptune["api_token"],
                                                        #custom_run_id="AL_" + 
                                                    )
                                                    run["param"] = run_param
                                                    run_param["NEPTUNE"]["RUN"] = run

                                                print("\n #####################################################################################")
                                                print("\n \n \n NEW RUN \n")
                                                print("Initial size: " + str(init_size))
                                                print("Batch size: " + str(labels_per_round))
                                                print("Max rounds: " + str(rounds))
                                                print("Labeled: " + str(labeled))
                                                print("Cost: " + str(cost))
                                                print("Setting: " + str(setting))
                                                print("Mod: " + str(mod))
                                                print("Overlap: " + str(overlap))
                                                print("Prediction Type " + str(expert_predict))
                                                print("Sample equal " + str(sample_equal))
                                                print("epochs_pretrain " + str(epochs_pretrain))

                                            


                                                start_time = time.time()
                                                expert_metrics, verma_metrics, hemmer_metrics = one_run(dataManager, run_param)
                                                print("--- %s seconds ---" % (time.time() - start_time))

                                                metrics_save["expert metrics"] = expert_metrics
                                                metrics_save["verma"] = verma_metrics
                                                metrics_save["hemmer"] = hemmer_metrics
                                                expert_metrics_all.append(metrics_save)
                                                with open(f'{param["Parent_PATH"]}/Metrics_Folder/Metrics_{count}.pickle', 'wb') as handle:
                                                    pickle.dump(expert_metrics_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
                                                count += 1
                                                if param["NEPTUNE"]["NEPTUNE"]:
                                                    run["metrics"] = metrics_save

                                                    run.stop()

    return expert_metrics_all


# In[17]:


def convert_cost_to_string(tp):
    return "(" + str(tp[0]) + ", " + str(tp[1]) + ")"

def convert_ids_to_string(ids):
    return f"{ids[0]}, {ids[1]}"

def convert_list_to_string(li):
    result = "["
    for el in li[:-2]:
        result = result + str(el)
    result = result + "]"
    return 


#CUDA_LAUNCH_BLOCKING=1
torch.backends.cudnn.benchmark = True


# In[ ]:
def main(args):

    path = args[0]

    if "liebschner" not in path:
        return

    param = {
        "PATH": f"{path}/Datasets/NIH/",
        "Parent_PATH": path,
        "TARGET": "Airspace_Opacity",
        "LABELER_IDS": [[4323195249, 4295232296]],
        "K": 10, #Number of folds
        "SEEDS": [1, 2, 3, 4, 42], #Seeds for the experiments
        "GT": True, # Determines if the classifier gets all data with GT Label or only the labeld data
        "MOD": ["confidence", "disagreement", "disagreement_diff", "ssl", "normal"], #Determines the experiment modus

        "OVERLAP": [0, 100],
        "SAMPLE_EQUAL": [False, True],

        "SETTING": ["AL", "SSL", "SSL_AL", "NORMAL", "SSL_AL_SSL"],

        "NUM_EXPERTS": 2,
        "NUM_CLASSES": 2,

        "EXPERT_PREDICT": ["right", "target"],

        "AL": { #Parameter for Active Learning
            "INITIAL_SIZE": [4, 8, 16, 32], #
            "EPOCH_TRAIN": 40, #
            "n_dataset": 2, #Number Classes
            "BATCH_SIZE": 4,
            "BATCH_SIZE_VAL": 32,
            "ROUNDS": [2, 4, 8],
            "LABELS_PER_ROUND": [4, 8, 16],
            "EPOCHS_DEFER": 10,
            "COST": [(0, 0), (5, 0)], #Cost for Cost sensitiv learning
            #"TRAIN REJECTOR": False,
            "PRELOAD": True,
            "PREPROCESS": True,
            "SSL_EPOCHS": 3
        
        },
        "SSL": {
            "PREBUILD": False,
            #"TRAIN_BATCH_SIZE": 128,
            "TRAIN_BATCH_SIZE": 254,
            "TEST_BATCH_SIZE": 254,
            "N_EPOCHS": 5, #number of training epoches
            "BATCHSIZE": 16, #train batch size of labeled samples
            #"N_IMGS_PER_EPOCH": 32768, #number of training images for each epoch
            "N_IMGS_PER_EPOCH": 4381*1, #number of training images for each epoch
        },
        "L2D": { # Parameter for Learning to defer
            "TRAIN_BATCH_SIZE": 128,
            "TEST_BATCH_SIZE": 128,
            "PRELOAD": True,
            "PREBUILD": True,
            "EPOCHS": 50,
            "VERMA": {},
            "HEMMER": {
                "EPOCHS": 50,
                "LR": 5e-3,
                "USE_LR_SCHEDULER": False,
                "DROPOUT": 0.00,
                "NUM_HIDDEN_UNITS": 30,
            },
        
        },
        "NEPTUNE": {
            "NEPTUNE": True,
        },
        "EMBEDDED": {
            "ARGS": {
                'dataset': "nih",
                'model': "resnet50",
                'num_classes': 2,
                'batch': 128,
                'lr': 0.001,
            },
            "EPOCHS": 30,
        },
    
    
        "epochs_pretrain": [0],
        "batch_size": 64,
        "alpha": 1.0, #scaling parameter for the loss function, default=1.0
        "epochs": 50,
        "patience": 25, #number of patience steps for early stopping the training
        "expert_type": "MLPMixer", #specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict
        "n_classes": 2, #K for K class classification
        "k": 0, #
        "n_experts": 2, #
        "lr": 0.001, #learning rate
        "weight_decay": 5e-4, #
        "warmup_epochs": 5, #
        #"loss_type": "softmax", #surrogate loss type for learning to defer
        "loss_type": "ova",
        "ckp_dir": f"{path}/Models", #directory name to save the checkpoints
        "experiment_name": "multiple_experts", #specify the experiment name. Checkpoints will be saved with this name
    }

    expert_metrics_all = run_experiment(param)

if __name__ == "__main__":
    main(sys.argv[1:])

# In[ ]:




# Store data (serialize)
#with open('Metrics.pickle', 'wb') as handle:
#    pickle.dump(expert_metrics_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load data (deserialize)
#with open('filename.pickle', 'rb') as handle:
#    unserialized_data = pickle.load(handle)

