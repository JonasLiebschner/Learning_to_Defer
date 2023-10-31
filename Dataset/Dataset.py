import tarfile
import shutil
import urllib
import os
import copy

import hashlib

import random
from itertools import chain
from typing import Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.v2 as transforms2
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from tabulate import tabulate

from sklearn.model_selection import KFold
from SSL.datasets import transform as T
from SSL.datasets.randaugment import RandomAugment
from SSL.datasets.sampler import RandomSampler, BatchSampler

import warnings


class BasicDataset:
    """
    Contains the main Dataset with GT Label and Expert Label for every Image, sorted by file name
    """
    def __init__(self, Path, target):
        df = pd.read_csv(Path + "labels.csv")
        ids = df["Reader ID"].unique()
        result = df[["Patient ID", "Image ID", target + "_GT_Label"]].drop_duplicates().rename(columns={target + "_GT_Label": 'GT'})
        for reader_id in ids:
            temp = df[df["Reader ID"] == reader_id][["Image ID", target + "_Expert_Label"]].rename(columns={target + "_Expert_Label": str(reader_id)})
            result = result.join(temp.set_index('Image ID'), on='Image ID')
        self.data = result.fillna(-1).reset_index(drop=True)

        print("Number of images of the whole dataset: " + str(len(self.data["Image ID"].values)))

    def getExpert(self, id):
        """
        Returns the data for the given expert
        """
        return self.data[["Patient ID", "Image ID", "GT", str(id)]].copy()

    def getData(self):
        """
        Returns all data
        """
        return self.data
    
    def getDataForLabelers(self, labelerIds):
        """
        Returns the data with ["Patient ID", "Image ID", "GT", [labelerIds]]
        """
        temp = self.data[["Patient ID", "Image ID", "GT"]].copy()
        for labelerId in labelerIds:
            temp[str(labelerId)] = self.data[str(labelerId)]
        return temp

class NIHDataset:
    """
    """
    def __init__(self, data: pd.DataFrame, transformation=None, preload=False, preprocess=False, param=None, image_container=None, size=(128, 128)):
        self.data = data
        self.image_ids = data["Image ID"].values
        self.targets = data["GT"].values

        if transformation == None:
            self.tfms = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.tfms = transformation
            
        self.param = param
        self.PATH = param["PATH"]

        self.images = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.preload = preload
        self.preprocess = preprocess

        self.image_list = np.empty(0)
        
        self.current = 0
        self.high = len(self.image_ids)

        if (image_container is None):
            print("None image container")

        self.image_container = image_container
        self.size = size

        self.transformed_images = {}
        
        if ((self.preload) or (self.image_container is not None)):
            self.loadImages()
            
    def __iter__(self):
        return self
    
    def __next__(self):
        self.current += 1
        if self.current < self.high:
            return self.__getitem__(self.current)
        raise StopIteration
            
    def loadImage(self, idx):
        """
        Load one single image
        """
        if self.image_container is not None:
            self.preload = True
            return self.image_container.get_image_from_name(self.image_ids[idx])
        else:
            return Image.open(self.PATH + "images/" + self.image_ids[idx]).convert("RGB").resize(self.size)
            
    def getImage(self, idx):
        """
        Returns the image from index idx
        """
        if self.preload:
            return self.images[idx]
        else:
            print("wrong")
            return self.loadImage(idx)

    def loadImages(self):
        """
        Load all images
        """
        if self.image_container is not None:
            self.preload = True
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
        #print("Loading complete")
        
    def transformImage(self, image):
        """
        Transforms the image
        """
        return self.tfms(image)#.to(self.device)

    def getTransformedImage(self, image, image_id):
        """
        Transforms the image
        """
        if image_id not in self.transformed_images.keys():
            self.transformed_images[image_id] = self.tfms(image)
        return self.transformed_images[image_id]
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        filename, target = self.image_ids[index], self.targets[index]
        img = self.getImage(index)
        
        if not self.preprocess:
            #img = self.transformImage(img)
            img = self.getTransformedImage(img, filename)#.to(self.device)
        return img, target, filename

    def __len__(self) -> int:
        #return len(self.images)
        return len(self.image_ids)

    """
    Functions for Verma active learning
    """
    def getAllImagesNP(self):
        """
        Returns all images from the Dataset
        """
        if not self.preload:
            self.preload = True
            self.loadImages()
        if self.image_list.size == 0:
            image_liste = []
            for img in self.images:
                #np_img = np.moveaxis(np.array(img), -1, 0)
                np_img = np.array(img)
                image_liste.append(np_img)
            self.image_list = np.array(image_liste)
        return self.image_list

    def getAllImages(self):
        """
        Returns all images from the Dataset
        """
        if not self.preload:
            self.preload = True
            self.loadImages()
        return self.images

    def getAllTargets(self):
        """
        Returns all targets
        """
        return self.targets

    def getAllFilenames(self):
        """
        Returns all filenames
        """
        return self.image_ids.astype(str)

    def getAllIndices(self):
        return self.data.index

class NIH_K_Fold_Dataloader:
    def __init__(self, dataset, k=10, labelerIds=[4323195249 , 4295194124], train_batch_size=8, test_batch_size=8,
                 seed=42, fraction=1.0, maxLabels=800, preload=False, preprocess=False, prebuild=False, param=None):
        self.dataset = dataset.getData()
        print("Full length: " + str(len(self.dataset)))
        self.k = k
        self.labelerIds = labelerIds
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed
        self.k_fold_datasets = []
        self.k_fold_patient_ids = []
        self.preload = preload
        self.preprocess = preprocess
        self.param = param
        self.prebuild = prebuild

        self.num_workers = param["num_worker"]

        #TODO: Implement support for non overlapping Labels

        ##
        self.common = True
        if self.common:
            self.common_image_ids = self.dataset["Image ID"].values.tolist()
            names = ["Patient ID", "Image ID", "GT"]
            for labelerId in self.labelerIds:
                temp = self.dataset[self.dataset[str(labelerId)] != -1]["Image ID"].values.tolist()
                self.common_image_ids = np.intersect1d(self.common_image_ids, temp)
                names.append(str(labelerId))
            self.data = self.dataset[self.dataset["Image ID"].isin(self.common_image_ids)][names]

        #Performance
        patient_ids = self.data["Patient ID"].unique()
        num_patient_images = self.data.drop_duplicates(subset=["Image ID"]).groupby(by="Patient ID", as_index=False).count()["Image ID"]
        self.patient_performance = pd.DataFrame({"Patient ID": patient_ids, "Num Patient Images": num_patient_images})
                     
        for labeler_id in self.labelerIds:
            temp = self.data[["Patient ID", "Image ID", "GT", str(labeler_id)]]
            temp["Expert_Correct"] = self.data["GT"] == self.data[str(labeler_id)]
            sum = temp[["Patient ID", "Expert_Correct"]].groupby(by="Patient ID", as_index=False).sum()
            sum.columns = ["Patient ID", f'{labeler_id}_num_correct']
            self.patient_performance = pd.merge(self.patient_performance, sum, left_on="Patient ID", right_on="Patient ID")
            self.patient_performance[f'{labeler_id}_perf'] = self.patient_performance[f'{labeler_id}_num_correct'] / self.patient_performance['Num Patient Images']

        target_temp = self.patient_performance[f'{labelerIds[0]}_perf'].astype(str)
        for labeler_id in labelerIds[1:]:
            target_temp = target_temp + "_" + self.patient_performance[f'{labeler_id}_perf'].astype(str)
        self.patient_performance["target"] = target_temp 

        self.expert_labels = self.data
        self._init_k_folds(maxLabels=maxLabels)

        if self.prebuild:
            self.buildDataloaders()

    def _init_k_folds(self, fraction=1.0, maxLabels=800):
        self.labels = self.expert_labels.drop_duplicates(subset=["Image ID"])
        self.labels = self.labels.fillna(0)
        self.labels = self.labels[["Patient ID", "Image ID", "GT"]]
        
        self.labels["Image ID"] = self.labels["Image ID"].astype('category')

        self.image_container = ImageContainer(path=self.param["PATH"], img_ids=self.labels["Image ID"], preload=True, transform=None, preprocess=False, img_size=(128, 128))

        kf_cv = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)

        # _ sind train indizes, fold_test_idxs ist liste der test indizes
        fold_data_idxs = [fold_test_idxs for (_, fold_test_idxs) in kf_cv.split(self.patient_performance["Patient ID"].values, self.patient_performance["target"].values)]

       
        for fold_idx in range(len(fold_data_idxs)):
            
            test = round(self.k*0.1)
            val = round(self.k*0.2)
            train = self.k - test - val
            
            test_folds_idxs = [(fold_idx + i) % self.k for i in range(test)]
            test_fold_data_idxs = [fold_data_idxs[test_fold_idx] for test_fold_idx in test_folds_idxs]
            test_fold_data_idxs = list(chain.from_iterable(test_fold_data_idxs))
            
            #test_fold_idx = fold_idx # Nummer des Folds
            #test_fold_data_idxs = fold_data_idxs[test_fold_idx] # Array der Test Indizes

            # use next 2 folds for validation set
            val_folds_idxs = [(fold_idx + test + i) % self.k for i in range(val)]
            val_fold_data_idxs = [fold_data_idxs[val_fold_idx] for val_fold_idx in val_folds_idxs]
            val_fold_data_idxs = list(chain.from_iterable(val_fold_data_idxs))

            # use next 7 folds for training set
            train_folds_idxs = [(fold_idx + (test + val) + i) % self.k for i in range(train)]
            #print(train_folds_idxs)
            train_folds_data_idxs = [fold_data_idxs[train_fold_idx] for train_fold_idx in train_folds_idxs]
            train_folds_data_idxs = list(chain.from_iterable(train_folds_data_idxs))

            
            train_patient_ids = self.patient_performance["Patient ID"].iloc[train_folds_data_idxs]
            #train_patient_ids = self.patient_performance["Patient ID"].iloc[train_folds_data_idxs].sample(n=min(maxLabels,len(train_folds_data_idxs)))
            val_patient_ids = self.patient_performance["Patient ID"].iloc[val_fold_data_idxs]
            test_patient_ids = self.patient_performance["Patient ID"].iloc[test_fold_data_idxs]

            #expert_train = self.labels[self.labels["Patient ID"].isin(train_patient_ids)]
            expert_train = self.labels[self.labels["Patient ID"].isin(train_patient_ids)]
            expert_train = self.labels[self.labels["Patient ID"].isin(train_patient_ids)].sample(n=min(maxLabels,len(expert_train)))
            expert_val = self.labels[self.labels["Patient ID"].isin(val_patient_ids)]
            expert_test = self.labels[self.labels["Patient ID"].isin(test_patient_ids)]

            # check that patients are not shared across training, validation and test split
            overlap = expert_train[expert_train["Patient ID"].isin(expert_val["Patient ID"])]
            assert len(overlap) == 0, "Train and Val Patient Ids overlap"

            overlap = expert_train[expert_train["Patient ID"].isin(expert_test["Patient ID"])]
            assert len(overlap) == 0, "Train and Test Patient Ids overlap"

            overlap = expert_val[expert_val["Patient ID"].isin(expert_test["Patient ID"])]
            assert len(overlap) == 0, "Val and Test Patient Ids overlap"

            expert_train = expert_train[["Image ID", "GT"]]
            expert_val = expert_val[["Image ID", "GT"]]
            expert_test = expert_test[["Image ID", "GT"]]
            
            print("Length of train + test + val: " + str(len(expert_train) + len(expert_val) + len(expert_test)))

            self.k_fold_datasets.append((expert_train, expert_val, expert_test))

    def get_data_loader_for_fold(self, fold_idx):
        if self.prebuild:
            return self.loaders[fold_idx][0], self.loaders[fold_idx][1], self.loaders[fold_idx][2]
        else:
            return self.create_Dataloader_for_Fold(fold_idx)

    def get_dataset_for_folder(self, fold_idx):
        expert_train, expert_val, expert_test = self.k_fold_datasets[fold_idx]

        return expert_train, expert_val, expert_test

    def create_Dataloader_for_Fold(self, idx):
        expert_train, expert_val, expert_test = self.k_fold_datasets[idx]

        expert_train_dataset = NIHDataset(expert_train, preload=self.preload, preprocess=self.preprocess, param=self.param, image_container=self.image_container)
        expert_val_dataset = NIHDataset(expert_val, preload=self.preload, preprocess=self.preprocess, param=self.param, image_container=self.image_container)
        expert_test_dataset = NIHDataset(expert_test, preload=self.preload, preprocess=self.preprocess, param=self.param, image_container=self.image_container)

        train_loader = torch.utils.data.DataLoader(dataset=expert_train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True, drop_last=False, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(dataset=expert_val_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers, shuffle=True, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=expert_test_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers, shuffle=True, drop_last=False, pin_memory=True)
        return train_loader, val_loader, test_loader

    def buildDataloaders(self):
        self.loaders = []
        for i in range(self.k):
            train_loader, val_loader, test_loader = self.create_Dataloader_for_Fold(i)
            loader_set = [train_loader, val_loader, test_loader]
            self.loaders.append(loader_set)
            print("Loaded set number " + str(i))

    def getFullDataloader(self):
        full_dataset = NIHDataset(self.labels[["Image ID", "GT"]], preload=self.preload, preprocess=self.preprocess, param=self.param, image_container=self.image_container)
        return torch.utils.data.DataLoader(dataset=full_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True, drop_last=False, pin_memory=True)

    def get_ImageContainer(self):
        return self.image_container
    
    def getData(self):
        return self.labels[self.labels["Patient ID"].isin(self.patient_performance["Patient ID"])]



class ImageContainer:
    def __init__(self, path, img_ids, preload=True, transform=None, preprocess=False, img_size=(128, 128)):
        self.PATH = path
        self.image_ids = img_ids.values
        self.preload = preload
        self.preprocess = preprocess
        self.img_size = img_size
        self.images = []

        if self.preload:
            self.loadImages()
        

    def loadImages(self):
        for idx in range(len(self.image_ids)):
            self.images.append(self.loadImage(idx))

            if idx % 200 == 0:
                print("Loaded image number: " + str(idx))
            
            if self.preprocess:
                self.images[idx] = self.transformImage(self.images[idx])

    def loadImage(self, idx):
        """
        Load one single image
        """
        return Image.open(self.PATH + "images/" + self.image_ids[idx]).convert("RGB").resize(self.img_size)
            
    def get_image_from_id(self, idx):
        """
        Returns the image from index idx
        """
        if self.preload:
            return self.images[idx]
        else:
            return self.loadImage(idx)

    def get_image_from_name(self, fname):
        if self.preload:
            return self.images[np.where(self.image_ids == fname)]
        else:
            return self.get_image_from_id(np.where(self.image_ids == fname))

    def get_images_from_name(self, fnames):
        if self.preload:
            return [self.images[np.where(self.image_ids == fname)[0][0]] for fname in fnames]
        else:
            return [self.get_image_from_id(np.where(self.image_ids == fname)[0][0]) for fname in fnames]

    def get_images_from_name_np(self, fnames):
        if self.preload:
            return [np.array(self.images[np.where(self.image_ids == fname)[0][0]]) for fname in fnames]
        else:
            return [np.array(self.get_image_from_id(np.where(self.image_ids == fname)[0][0])) for fname in fnames]
    
def setupLabels(PATH_Labels):
    path_to_test_Labels = PATH_Labels + "test_labels.csv"
    path_to_val_Labels = PATH_Labels + "validation_labels.csv"
    test_labels = pd.read_csv(path_to_test_Labels)
    val_labels = pd.read_csv(path_to_val_Labels)

    ground_truth_labels = pd.concat([test_labels,val_labels])
    ground_truth_labels["Fracture_Label"] = ground_truth_labels["Fracture"].map(dict(YES=1, NO=0))
    ground_truth_labels["Pneumothorax_Label"] = ground_truth_labels["Pneumothorax"].map(dict(YES=1, NO=0))
    ground_truth_labels["Airspace_Opacity_Label"] = ground_truth_labels["Airspace opacity"].map(dict(YES=1, NO=0))
    ground_truth_labels["Nodule_Or_Mass_Label"] = ground_truth_labels["Nodule or mass"].map(dict(YES=1, NO=0))

    path_to_individual_reader = PATH_Labels + "individual_readers.csv"
    individual_readers = pd.read_csv(path_to_individual_reader)

    individual_readers["Fracture_Expert_Label"] = individual_readers["Fracture"].map(dict(YES=1, NO=0))
    individual_readers["Pneumothorax_Expert_Label"] = individual_readers["Pneumothorax"].map(dict(YES=1, NO=0))
    individual_readers["Airspace_Opacity_Expert_Label"] = individual_readers["Airspace opacity"].map(dict(YES=1, NO=0))
    individual_readers["Nodule_Or_Mass_Expert_Label"] = individual_readers["Nodule/mass"].map(dict(YES=1, NO=0))

    individual_readers["Fracture_GT_Label"] = individual_readers["Image ID"].map(pd.Series(ground_truth_labels["Fracture_Label"].values,index=ground_truth_labels["Image Index"]).to_dict())
    individual_readers["Pneumothorax_GT_Label"] = individual_readers["Image ID"].map(pd.Series(ground_truth_labels["Pneumothorax_Label"].values,index=ground_truth_labels["Image Index"]).to_dict())
    individual_readers["Airspace_Opacity_GT_Label"] = individual_readers["Image ID"].map(pd.Series(ground_truth_labels["Airspace_Opacity_Label"].values,index=ground_truth_labels["Image Index"]).to_dict())
    individual_readers["Nodule_Or_Mass_GT_Label"] = individual_readers["Image ID"].map(pd.Series(ground_truth_labels["Nodule_Or_Mass_Label"].values,index=ground_truth_labels["Image Index"]).to_dict())

    individual_readers["Fracture_Correct"] = (individual_readers['Fracture_Expert_Label']==individual_readers['Fracture_GT_Label']).astype(int)
    individual_readers["Pneumothorax_Correct"] = (individual_readers['Pneumothorax_Expert_Label']==individual_readers['Pneumothorax_GT_Label']).astype(int)
    individual_readers["Airspace_Opacity_Correct"] = (individual_readers['Airspace_Opacity_Expert_Label']==individual_readers['Airspace_Opacity_GT_Label']).astype(int)
    individual_readers["Nodule_Or_Mass_Correct"] = (individual_readers['Nodule_Or_Mass_Expert_Label']==individual_readers['Nodule_Or_Mass_GT_Label']).astype(int)

    individual_readers.to_csv(PATH_Labels + "labels.csv")
    
def downloadData(PATH):
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]
    files = os.listdir(PATH + "images/")
    for idx, link in enumerate(links):
        fn = PATH + "images/" + 'images_%02d.tar.gz' % (idx+1)
        if ('images_%02d.tar.gz' % (idx+1)) in files:
            continue
        print('downloading'+fn+'...')
        urllib.request.urlretrieve(link, fn)  # download the zip file

    print("Download complete. Please check the checksums")
    
def unpackData(PATH):
    Path = PATH + "images/"
    files = os.listdir(Path)
    for filename in files:
        if ".tar.gz" not in filename:
            continue
        # open file
        file = tarfile.open(Path + filename)
        
        # extracting file
        file.extractall(Path + filename[:-7])

        file.close()
        
def moveData(PATH):
    Path = PATH + "images/"
    directories = os.listdir(Path)
    for direc in directories:
        if "png" in direc:
            continue
        if "tar.gz" in direc:
            continue
        if "checkpoint" in direc:
            continue
        if "images" not in direc:
            continue
        filenames = os.listdir(Path + direc + "/images/")
        for filename in filenames:
            shutil.move(Path + direc + "/images/" + filename, Path + filename)
        shutil.rmtree(Path + direc)
        
def downloadLabels(Path):
    links = [
        "https://storage.googleapis.com/gcs-public-data--healthcare-nih-chest-xray-labels/four_findings_expert_labels/individual_readers.csv",
        "https://storage.googleapis.com/gcs-public-data--healthcare-nih-chest-xray-labels/four_findings_expert_labels/test_labels.csv",
        "https://storage.googleapis.com/gcs-public-data--healthcare-nih-chest-xray-labels/four_findings_expert_labels/validation_labels.csv"
    ]
    urllib.request.urlretrieve(links[0], Path + "individual_readers.csv")
    urllib.request.urlretrieve(links[1], Path + "test_labels.csv")
    urllib.request.urlretrieve(links[2], Path + "validation_labels.csv")
    
    
class DataManager():
    """
    Class to contain and manage all data for all experiments
    
    This class contains all functions to get data and dataloaders for every step of the experiment
    
    It also implements a k fold
    """
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def __init__(self, path, target, param, seeds):
        
        self.path = path
        self.target = target
        self.param = param
        self.seeds = seeds
        
        self.labeler_ids = param["LABELER_IDS"]
        
        self.basicDataset = BasicDataset(Path=self.path, target=self.target)
        
        self.fullImageContainer = ImageContainer(path=self.path, img_ids=self.basicDataset.getData()["Image ID"], preload=True, transform=None, preprocess=False, img_size=(128, 128))
        
        #self.createData()
    
    def createData(self):
        
        self.kFoldDataloaders = {}
        self.SSLDatasets = {}
        
        for seed in self.seeds:
            self.set_seed(seed)
        
            self.kFoldDataloaders[seed] = NIH_K_Fold_Dataloader(
                    dataset = self.basicDataset,
                    k = self.param["K"],
                    labelerIds = self.labeler_ids,
                    train_batch_size = self.param["L2D"]["TRAIN_BATCH_SIZE"],
                    test_batch_size = self.param["L2D"]["TEST_BATCH_SIZE"],
                    seed = seed,
                    #maxLabels = maxL,
                    preprocess = False,
                    preload = self.param["L2D"]["PRELOAD"],
                    prebuild = self.param["L2D"]["PREBUILD"],
                    param = self.param
                )
            
            self.SSLDatasets[seed] = self.SSLDataset = SSLDataset(dataset=self.basicDataset, kFoldDataloader=self.kFoldDataloaders[seed], imageContainer=self.fullImageContainer, 
                                                                  labeler_ids=self.labeler_ids, param=self.param, seed=seed, prebuild = self.param["SSL"]["PREBUILD"])
        
         
    def getKFoldDataloader(self, seed):
        return self.kFoldDataloaders[seed]
    
    def getSSLDataset(self, seed):
        return self.SSLDatasets[seed]

    def getBasicDataset(self):
        return self.basicDataset
    
    
class SSLDataset():
    def __init__(self, dataset, kFoldDataloader, imageContainer, labeler_ids, param, seed, prebuild=False):
        self.basicDataset = dataset
        self.kFoldDataloader = kFoldDataloader
        self.imageContainer = imageContainer
        self.labeler_ids = labeler_ids
        self.param = param
        self.seed = seed
        self.set_seed(self.seed)
        
        self.k_fold_datasets = []
        self.k_fold_datasets_labeled = []
        
        self.prebuild=prebuild
        self.preload = False
        self.preprocess = False

        self.num_workers = param["num_worker"]
        
        self.unpack_param()
        self.setup()

        if self.prebuild:
            self.buildDataloaders()
        
    def unpack_param(self):
        self.k = self.param["K"]
        #self.overlap_k = round(self.param["LABELED"]*self.param["OVERLAP"]/100)
        #self.n_labels = self.param["LABELED"]
        self.train_batch_size = self.param["SSL"]["TRAIN_BATCH_SIZE"]
        self.test_batch_size = self.param["SSL"]["TEST_BATCH_SIZE"]
        
    def set_seed(self, seed, fold=None, text=None):
        if fold is not None and text is not None:
            s = text + f" + {seed} + {fold}"
            seed = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
    def setup(self):

        self.data = self.basicDataset.getDataForLabelers(self.labeler_ids).astype('category').drop_duplicates(subset=['Image ID', "GT"], keep='first')
        
        self.labels = self.data.drop("Patient ID", axis=1)
        
        self.used_image_ids = self.kFoldDataloader.getData().drop("Patient ID", axis=1)
        
        self.unused_data = self.data[~self.data["Image ID"].isin(self.used_image_ids["Image ID"])]
        self.used_data = self.data[self.data["Image ID"].isin(self.used_image_ids["Image ID"])]
        
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
       
        #TODO: Is Patient ID better than Image ID, are splitted patients a problem?
        
        split_target = "Image ID"
        #split_target = "Patient ID"
        
        self.unused_data.head(10)
        
        split = kf.split(self.unused_data["Image ID"], self.unused_data["GT"])
        
        fold_data_idxs = [fold_test_idxs for (_, fold_test_idxs) in split]
        
        #fold_data_idxs = [fold_test_idxs for (_, fold_test_idxs) in kf_cv.split(self.patient_performance["Patient ID"].values, self.patient_performance["target"].values)]

            
        #for j, (train_index, test_index) in enumerate(kf.split(self.unused_data["Image ID"], self.unused_data["GT"])):
        #for j, (train_index, test_index) in enumerate(split):
        for fold_idx in range(len(fold_data_idxs)):
            
            #fold_idx = j
            
            #print(j)
            
            test = round(self.k*0.1)
            val = round(self.k*0.2)
            train = self.k - test - val
            
            test_folds_idxs = [(fold_idx + i) % self.k for i in range(test)]
            test_fold_data_idxs = [fold_data_idxs[test_fold_idx] for test_fold_idx in test_folds_idxs]
            test_fold_data_idxs = list(chain.from_iterable(test_fold_data_idxs))
            
            #test_fold_idx = fold_idx # Nummer des Folds
            #test_fold_data_idxs = fold_data_idxs[test_fold_idx] # Array der Test Indizes

            # use next 2 folds for validation set
            val_folds_idxs = [(fold_idx + test + i) % self.k for i in range(val)]
            val_fold_data_idxs = [fold_data_idxs[val_fold_idx] for val_fold_idx in val_folds_idxs]
            val_fold_data_idxs = list(chain.from_iterable(val_fold_data_idxs))

            # use next 7 folds for training set
            train_folds_idxs = [(fold_idx + (test + val) + i) % self.k for i in range(train)]
            #print(train_folds_idxs)
            train_folds_data_idxs = [fold_data_idxs[train_fold_idx] for train_fold_idx in train_folds_idxs]
            train_folds_data_idxs = list(chain.from_iterable(train_folds_data_idxs))

            
            train_patient_ids = self.unused_data[split_target].iloc[train_folds_data_idxs]
            #train_patient_ids = self.patient_performance["Patient ID"].iloc[train_folds_data_idxs].sample(n=min(maxLabels,len(train_folds_data_idxs)))
            val_patient_ids = self.unused_data[split_target].iloc[val_fold_data_idxs]
            test_patient_ids = self.unused_data[split_target].iloc[test_fold_data_idxs]
            
            
            #Add the data from the other k_fold
            used_train, used_val, used_test = self.kFoldDataloader.get_dataset_for_folder(fold_idx = fold_idx)
            
            
            train_patient_ids = pd.concat([train_patient_ids, used_train["Image ID"]], ignore_index=True)
            val_patient_ids = pd.concat([val_patient_ids, used_val["Image ID"]], ignore_index=True)
            test_patient_ids = pd.concat([test_patient_ids, used_test["Image ID"]], ignore_index=True)

            expert_train = self.labels[self.labels[split_target].isin(train_patient_ids)]
            #expert_train = self.labels[self.labels[split_target].isin(train_patient_ids)].sample(n=min(maxLabels,len(expert_train)))
            expert_val = self.labels[self.labels[split_target].isin(val_patient_ids)]
            expert_test = self.labels[self.labels[split_target].isin(test_patient_ids)]
            


            # check that patients are not shared across training, validation and test split
            overlap = expert_train[expert_train[split_target].isin(expert_val[split_target])]
            assert len(overlap) == 0, "Train and Val Patient Ids overlap"

            overlap = expert_train[expert_train[split_target].isin(expert_test[split_target])]
            assert len(overlap) == 0, "Train and Test Patient Ids overlap"

            overlap = expert_val[expert_val[split_target].isin(expert_test[split_target])]
            assert len(overlap) == 0, "Val and Test Patient Ids overlap"

            #expert_train = expert_train[["Image ID", "GT"]]
            #expert_val = expert_val[["Image ID", "GT"]]
            #expert_test = expert_test[["Image ID", "GT"]]

            self.k_fold_datasets.append((expert_train, expert_val, expert_test))
            self.k_fold_datasets_labeled.append((used_train, used_val, used_test))
            #print("Added")
        
        #self.createLabeledIndices(labelerIds=self.labeler_ids, n_L=self.n_labels, k=self.overlap_k, seed=self.seed)
            
    def sampleIndices(self, n, k, data, experten, seed = None, sample_equal=False, fold=None):
        """
        Creates indices for which data are labeled for each expert
        
        n - number of labels
        k - number of shared labeled data
        data - the data (x and y)
        experten - list of labelerIds
        seed
        
        """
        #Set seed
        if seed is not None:
            self.set_seed(seed, fold, text="")

        if k > n:
            k = n
            warnings.warn("k was bigger than n")
            
        data = data.reset_index(drop=True)
            
        #Get all indices
        all_indices = indices = [j for j in range(len(data))]

        #print(f"Len all indices {len(all_indices)}")
        
        #Get which indices are labeled from which expert
        experts_indices = {}
        common_indices = all_indices
        for expert in experten:
            experts_indices[expert] = [j for j in all_indices if (data[str(expert)][j] != -1)]
            common_indices = set(common_indices).intersection(experts_indices[expert])
        common_indices = list(common_indices)
        #common_indices.sort()

        #print(f"Len common indices {len(common_indices)}")

        #Sample the shared indices
        if sample_equal:
            indices_0 = [ind for ind in common_indices if data["GT"][ind] == 0]
            indices_1 = [ind for ind in common_indices if data["GT"][ind] == 1]
            same_indices = random.sample(indices_0, round(k/2))
            same_indices += random.sample(indices_1, round(k/2))
            #print(f"Indices with GT=0: {k/2} and with GT=1: {k/2}")
            pass
        else:
            same_indices = random.sample(common_indices, k)
        diff_indices = []
        used_indices = same_indices
        indices = {}
        if k == n:
            for expert in experten:
                indices[expert] = same_indices
        if k < n: #If there are not shared indices
            for expert in experten:
                working_indices = experts_indices[expert]
                temp_indices = []
                count = 0 # To avoid infinity loop
                working_indices_gt = {}
                if sample_equal:
                    #print(f"Indices with GT=0: {n/2} and with GT=1: {n/2}")
                    working_indices_gt[0] = [ind for ind in working_indices if data["GT"][ind] == 0]
                    working_indices_gt[1] = [ind for ind in working_indices if data["GT"][ind] == 1]
                    #print(f"Len GT=0 {len(working_indices_gt[0])} and GT=1 {len(working_indices_gt[1])}")
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
                        temp = random.sample(working_indices, 1)
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
                indices[expert] = (same_indices + temp_indices)
        return indices
            
    def createLabeledIndices(self, labelerIds, n_L, k, seed=0, sample_equal=False):
        """
        Creates the labeled indices for all folds for every expert
        
        n_L - number of labeled images
        k - number of shared labeled images
        seed - random seed to init the random function
        """
        #Generate random seeds for every loop
        self.labeled_indices = []
        self.addedIndices = []
        for i in range(self.k):
            train_data, _, _ = self.k_fold_datasets[i]            
            sampled_indices = self.sampleIndices(n=n_L, k=k, data=train_data, experten=labelerIds, seed=seed, sample_equal=sample_equal, fold=i)
            #print(sampled_indices)
            self.labeled_indices.append(sampled_indices)

            #Set empty additional indices for every fold and every expert. Important for AL because indices could be added later
            self.addedIndices.append({})
            for labelerId in labelerIds:
                self.addedIndices[-1][labelerId] = []
        
    def getLabeledIndices(self, labelerId, fold_idx):
        return self.labeled_indices[fold_idx][labelerId] + self.addedIndices[fold_idx][labelerId]

    def addNewLabels(self, filenames, fold_idx, labelerId):
        """
        Add new indeces for labeled images for ssl in combination with active learning

        filenames contains the names of the new labeled images
        fold_idx is the current fold to spezify where the indices should be set
        """
        train_data, _, _ = self.getDatasetsForExpert(labelerId, fold_idx)
        X = np.array(train_data["Image ID"])
        indices = [np.where(X == item)[0] for item in filenames]
        assert set(filenames) == set(np.array(X)[indices].flatten()), "Filenames don't match" #Check if indices are correct
        #self.addedIndices[fold_idx][labelerId] = indices
        self.addedIndices[fold_idx][labelerId] += np.array([list(idx) for idx in indices]).flatten().tolist()
        print(self.addedIndices[fold_idx][labelerId])
        
    def getDatasetsForExpert(self, labelerId, fold_idx):
        print("Index: " + str(fold_idx))
        working_train, working_val, working_test = self.k_fold_datasets[fold_idx]
        working_train = working_train[["Image ID", str(labelerId)]]
        working_val = working_val[["Image ID", str(labelerId)]]
        working_test = working_test[["Image ID", str(labelerId)]]
        
        return working_train, working_val, working_test
    
    def getTrainDataset(self, labelerId, fold_idx):
        train_data, _, _ = self.getDatasetsForExpert(labelerId, fold_idx)
        gt_train, _, _ = self.k_fold_datasets[fold_idx]
        
        X = np.array(train_data["Image ID"])
        y = np.array(train_data[str(labelerId)])
        gt = np.array(gt_train["GT"])
        
        all_indices = [i for i in range(len(y))]
        labeled_indices = self.getLabeledIndices(labelerId, fold_idx)
        
        data_x = [X[ind] for ind in labeled_indices]
        label_x = [y[ind] for ind in labeled_indices]
        gt_x = [gt[ind] for ind in labeled_indices]
        
        data_u = [X[ind] for ind in all_indices if (ind not in labeled_indices)]
        label_u = [y[ind] for ind in all_indices if (ind not in labeled_indices)]
        gt_u = [gt[ind] for ind in all_indices if (ind not in labeled_indices)]
        
        return data_x, label_x, gt_x, data_u, label_u, gt_u
    
    def getValDataset(self, labelerId, fold_idx):
        _, val_data, _ = self.getDatasetsForExpert(labelerId, fold_idx)
        _, val_gt, _ = self.k_fold_datasets[fold_idx]
        X = np.array(val_data["Image ID"])
        y = np.array(val_data[str(labelerId)])
        gt = np.array(val_gt["GT"])
        
        return X, y, gt
    
    def getTestDataset(self, labelerId, fold_idx):
        _, _, test_data = self.getDatasetsForExpert(labelerId, fold_idx)
        _, _, test_gt = self.k_fold_datasets[fold_idx]
        X = np.array(test_data["Image ID"])
        y = np.array(test_data[str(labelerId)])
        gt = np.array(test_gt["GT"])
        
        return X, y, gt
        
    
    def get_train_loader_interface(self, expert, batch_size, mu, n_iters_per_epoch, L, method='comatch', imsize=(128, 128), fold_idx=0, pin_memory=False):
        labeler_id = expert.labeler_id
        
        data_x, label_x, gt_x, data_u, label_u, gt_u = self.getTrainDataset(labeler_id, fold_idx)
        
        #print(f'Label check: {Counter(label_x)}')
        print("Labels: " + str(len(label_x)))
        ds_x = NIH_SSL_Dataset(
            data=data_x,
            labels=label_x,
            gt=gt_x,
            mode='train_x',
            image_container=self.imageContainer,
            imsize=imsize
        )
        sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
        #batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
        batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=False)  # yield a batch of samples one time
        dl_x = torch.utils.data.DataLoader(
            ds_x,
            batch_sampler=batch_sampler_x,
            num_workers=0,
            pin_memory=True
        )
        if data_u is None:
            return dl_x
        else:
            ds_u = NIH_SSL_Dataset(
                data=data_u,
                labels=label_u,
                gt=gt_u,
                mode='train_u_%s'%method,
                image_container=self.imageContainer,
                imsize=imsize
            )
            sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
            batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
            dl_u = torch.utils.data.DataLoader(
                ds_u,
                batch_sampler=batch_sampler_u,
                num_workers=4,
                pin_memory=False
            )
            return dl_x, dl_u
        
    def get_val_loader_interface(self, expert, batch_size, num_workers, pin_memory=True, imsize=(128, 128), fold_idx=0):
        """Get data loader for the validation set

        :param expert: Synthetic cifar expert
        :param batch_size: Batch size
        :param num_workers: Number of workers
        :param pin_memory: Pin memory
        :param imsize: Size of images

        :return: Dataloader
        """
        labeler_id = expert.labeler_id
        data, labels, gt = self.getValDataset(labeler_id, fold_idx)

        ds = NIH_SSL_Dataset(
            data=data,
            labels=labels,
            gt=gt,
            mode='val',
            imsize=imsize,
            image_container=self.imageContainer
        )
        dl = torch.utils.data.DataLoader(
            ds,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )
        return dl
    
    def get_test_loader_interface(self, expert, batch_size, num_workers, pin_memory=True, imsize=(128, 128), fold_idx=0):
        """Get data loader for the validation set

        :param expert: Synthetic cifar expert
        :param batch_size: Batch size
        :param num_workers: Number of workers
        :param pin_memory: Pin memory
        :param imsize: Size of images

        :return: Dataloader
        """
        labeler_id = expert.labeler_id
        data, labels, gt = self.getTestDataset(labeler_id, fold_idx)

        ds = NIH_SSL_Dataset(
            data=data,
            labels=labels,
            gt=gt,
            mode='test',
            imsize=imsize,
            image_container=self.imageContainer
        )
        dl = torch.utils.data.DataLoader(
            ds,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )
        return dl

    
    def getLabeledFilenames(self, labelerId, fold_idx):
        train_data, _, _ = self.getDatasetsForExpert(labelerId, fold_idx)
        X = np.array(train_data["Image ID"])
        labeled_indices = self.getLabeledIndices(labelerId, fold_idx)
        return X[labeled_indices].tolist()
        
    """
    Functions to get the whole dataset as dataloaders
    """
    def get_data_loader_for_fold(self, fold_idx):
        if self.prebuild:
            return self.loaders[fold_idx][0], self.loaders[fold_idx][1], self.loaders[fold_idx][2]
        else:
            return self.create_Dataloader_for_Fold(fold_idx)

    def get_dataset_for_folder(self, fold_idx):
        expert_train, expert_val, expert_test = self.k_fold_datasets[fold_idx]

        return expert_train, expert_val, expert_test

    def create_Dataloader_for_Fold(self, idx):
        expert_train, expert_val, expert_test = self.k_fold_datasets[idx]

        expert_train_dataset = NIHDataset(expert_train, preload=self.preload, preprocess=self.preprocess, param=self.param, image_container=self.imageContainer)
        expert_val_dataset = NIHDataset(expert_val, preload=self.preload, preprocess=self.preprocess, param=self.param, image_container=self.imageContainer)
        expert_test_dataset = NIHDataset(expert_test, preload=self.preload, preprocess=self.preprocess, param=self.param, image_container=self.imageContainer)

        train_loader = torch.utils.data.DataLoader(dataset=expert_train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True, drop_last=False, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(dataset=expert_val_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers, shuffle=True, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=expert_test_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers, shuffle=True, drop_last=False, pin_memory=True)
        return train_loader, val_loader, test_loader

    def buildDataloaders(self):
        self.loaders = []
        for i in range(self.k):
            train_loader, val_loader, test_loader = self.create_Dataloader_for_Fold(i)
            loader_set = [train_loader, val_loader, test_loader]
            self.loaders.append(loader_set)
            print("Loaded set number " + str(i))

    def getFullDataloader(self):
        full_dataset = NIHDataset(self.labels[["Image ID", "GT"]], preload=self.preload, preprocess=self.preprocess, param=self.param, image_container=self.imageContainer)
        return torch.utils.data.DataLoader(dataset=full_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True, drop_last=False, pin_memory=True)

    def get_ImageContainer(self):
        return self.imageContainer
    
    def getData(self):
        return self.labels[self.labels["Patient ID"].isin(self.patient_performance["Patient ID"])]
    
class NIH_SSL_Dataset(Dataset):
    """Class representing the NIH dataset

    :param data: Images
    :param labels: Labels
    :param mode: Mode
    :param imsize: Image size

    :ivar data: Images
    :ivar labels: Labels
    :ivar mode: Mode
    """
    def __init__(self, data, labels, gt, mode, image_container=None, imsize=(128, 128)) -> None:
        self.image_ids = data
        self.labels = labels
        self.gt = gt
        self.mode = mode
        self.image_container = image_container

        self.preprocess=False
        self.preload=True

        self.images = []
        
        self.loadImages()

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        trans_weak = T.Compose([
            T.Resize(imsize),
            T.PadandRandomCrop(border=4, cropsize=imsize),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean, std),
        ])
        trans_strong0 = T.Compose([
            T.Resize(imsize),
            #transforms.Resize(imsize[0]),
            
            T.PadandRandomCrop(border=4, cropsize=imsize),
            T.RandomHorizontalFlip(p=0.5),
            #transforms.RandomHorizontalFlip(p=0.5),
            RandomAugment(2, 10),
            #Normal way
            #T.Normalize(mean, std),
            #T.ToTensor(),

            #Optimization
            transforms.ToTensor(),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean, std),
        ])
        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.ToTensor(),
            transforms.RandomResizedCrop(imsize, scale=(0.2, 1.), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms2.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            #T.Normalize(mean, std),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_comatch':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_fixmatch':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            self.trans = T.Compose([
                T.Resize(imsize),
                #T.Normalize(mean, std),
                #T.ToTensor(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            
    def loadImages(self):
        """
        Load all images
        """
        if self.image_container is not None:
            self.images = copy.deepcopy(self.image_container.get_images_from_name_np(self.image_ids))
        else:
            print("No image container")
            for idx in range(len(self.image_ids)):
                if self.preprocess:
                    self.images.append(self.transformImage(self.loadImage(idx)))
                else:
                    self.images.append(self.loadImage(idx))

    def __getitem__(self, index: int):
        filename, label, gt = self.image_ids[index], self.labels[index], self.gt[index]
        im = self.images[index]
        return self.trans(im), label, filename, gt

    def __len__(self) -> int:
        return len(self.images)
    
class TwoCropsTransform:
    """Take 2 random augmentations of one image

    :param trans_weak: Transform for the weak augmentation
    :param trans_strong: Transform for the strong augmentation

    :ivar trans_weak: Transform for the weak augmentation
    :ivar trans_strong: Transform for the strong augmentation
    """

    def __init__(self, trans_weak, trans_strong):
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong

    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]


class ThreeCropsTransform:
    """Take 3 random augmentations of one image

    :param trans_weak: Transform for the weak augmentation
    :param trans_strong0: Transform for the first strong augmentation
    :param trans_strong1: Transform for the second strong augmentation

    :ivar trans_weak: Transform for the weak augmentation
    :ivar trans_strong0: Transform for the first strong augmentation
    :ivar trans_strong1: Transform for the second strong augmentation
    """

    def __init__(self, trans_weak, trans_strong0, trans_strong1):
        self.trans_weak = trans_weak
        self.trans_strong0 = trans_strong0
        self.trans_strong1 = trans_strong1

    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong0(x)
        x3 = self.trans_strong1(x)
        return [x1, x2, x3]