import tarfile
import shutil
import urllib
import os

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
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from tabulate import tabulate

from sklearn.model_selection import KFold
from datasets import transform as T
from datasets.randaugment import RandomAugment
from datasets.sampler import RandomSampler, BatchSampler

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

    def getExpert(self, id):
        """
        Returns the data for the given expert
        """
        return result["Image ID", "GT", str(id)]

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

        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = None
        
        self.preload = preload
        self.preprocess = preprocess

        self.image_list = np.empty(0)
        
        self.current = 0
        self.high = len(self.image_ids)

        if (image_container is None):
            print("None image container")

        self.image_container = image_container
        self.size = size
        
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
            print("wrong")
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
        #print("transformed")
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self.tfms(image)#.to(self.device)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        filename, target = self.image_ids[index], self.targets[index]
        img = self.getImage(index)
        
        if not self.preprocess:
            img = self.transformImage(img)
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
    def __init__(self, dataset, k=10, labelerIds=[4323195249, 4295194124], train_batch_size=8, test_batch_size=8,
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

        self.num_workers = 4

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
                    train_batch_size = self.param["TRAIN_BATCH_SIZE"],
                    test_batch_size = self.param["TEST_BATCH_SIZE"],
                    seed = seed,
                    #maxLabels = maxL,
                    preprocess = False,
                    preload = self.param["PRELOAD"],
                    prebuild = self.param["PREBUILD"],
                    param = self.param
                )
            
            self.SSLDatasets[seed] = self.SSLDataset = SSLDataset(dataset=self.basicDataset, kFoldDataloader=self.kFoldDataloaders[seed], 
                                                                 imageContainer=self.fullImageContainer, labeler_ids=self.labeler_ids, param=self.param, seed=seed)
        
         
    def getKFoldDataloader(self, seed):
        return self.kFoldDataloaders[seed]
    
    def getSSLDataset(self, seed):
        return self.SSLDatasets[seed]
    
    
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

        self.num_workers = 4
        
        self.unpack_param()
        self.setup()

        if self.prebuild:
            self.buildDataloaders()
        
    def unpack_param(self):
        self.k = self.param["K"]
        self.overlap_k = self.param["OVERLAP K"]
        self.n_labels = self.param["NUMBER LABELS"]
        self.train_batch_size = self.param["TRAIN_BATCH_SIZE"]
        self.test_batch_size = self.param["TEST_BATCH_SIZE"]
        
    def set_seed(self, seed):
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
            print("Added")
        
        self.createLabeledIndices(labelerIds=self.labeler_ids, n_L=self.n_labels, k=self.overlap_k, seed=self.seed)
            
    def sampleIndices(self, n, k, data, experten, seed = None):
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
            self.set_seed(seed)
            
        data = data.reset_index(drop=True)
            
        #Get all indices
        all_indices = indices = [j for j in range(len(data))]
        
        #Get which indices are labeled from which expert
        experts_indices = {}
        common_indices = all_indices
        for expert in experten:
            experts_indices[expert] = [j for j in all_indices if (data[str(expert)][j] != -1)]
            common_indices = set(common_indices).intersection(experts_indices[expert])
        common_indices = list(common_indices)
        common_indices.sort()

        #Sample the shared indices
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
            
    def createLabeledIndices(self, labelerIds, n_L, k, seed=0):
        """
        Creates the labeled indices for all folds for every expert
        
        n_L - number of labeled images
        k - number of shared labeled images
        seed - random seed to init the random function
        """
        if seed is not None:
            self.set_seed(seed)
        #Generate random seeds for every loop
        seeds = [random.randint(0, 10000) for k in range(self.k)]
        self.labeled_indices = []
        for i in range(self.k):
            train_data, _, _ = self.k_fold_datasets[i]            
            sampled_indices = self.sampleIndices(n=n_L, k=k, data=train_data, experten=labelerIds, seed=seeds[i])
            #print(sampled_indices)
            self.labeled_indices.append(sampled_indices)
        
    def getLabeledIndices(self, labelerId, fold_idx):
        return self.labeled_indices[fold_idx][labelerId]
        
    def getDatasetsForExpert(self, labelerId, fold_idx):
        print("Index: " + str(fold_idx))
        working_train, working_val, working_test = self.k_fold_datasets[fold_idx]
        working_train = working_train[["Image ID", str(labelerId)]]
        working_val = working_val[["Image ID", str(labelerId)]]
        working_test = working_test[["Image ID", str(labelerId)]]
        
        return working_train, working_val, working_test
    
    def getTrainDataset(self, labelerId, fold_idx):
        train_data, _, _ = self.getDatasetsForExpert(labelerId, fold_idx)
        X = np.array(train_data["Image ID"])
        y = np.array(train_data[str(labelerId)])
        
        all_indices = [i for i in range(len(y))]
        labeled_indices = self.getLabeledIndices(labelerId, fold_idx)
        
        data_x = [X[ind] for ind in labeled_indices]
        label_x = [y[ind] for ind in labeled_indices]
        
        data_u = [X[ind] for ind in all_indices if (ind not in labeled_indices)]
        label_u = [y[ind] for ind in all_indices if (ind not in labeled_indices)]
        
        return data_x, label_x, data_u, label_u
    
    def getValDataset(self, labelerId, fold_idx):
        _, val_data, _ = self.getDatasetsForExpert(labelerId, fold_idx)
        X = np.array(val_data["Image ID"])
        y = np.array(val_data[str(labelerId)])
        
        return X, y
    
    def getTestDataset(self, labelerId, fold_idx):
        _, _, test_data = self.getDatasetsForExpert(labelerId, fold_idx)
        X = np.array(test_data["Image ID"])
        y = np.array(test_data[str(labelerId)])
        
        return X, y
        
    
    def get_train_loader_interface(self, expert, batch_size, mu, n_iters_per_epoch, L, method='comatch', imsize=(128, 128), fold_idx=0):
        labeler_id = expert.labeler_id
        
        data_x, label_x, data_u, label_u = self.getTrainDataset(labeler_id, fold_idx)
        
        #print(f'Label check: {Counter(label_x)}')
        print("Labels: " + str(len(label_x)))
        ds_x = NIH_SSL_Dataset(
            data=data_x,
            labels=label_x,
            mode='train_x',
            image_container=self.imageContainer,
            imsize=imsize
        )
        sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
        batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
        dl_x = torch.utils.data.DataLoader(
            ds_x,
            batch_sampler=batch_sampler_x,
            num_workers=4,
            pin_memory=True
        )
        if data_u is None:
            return dl_x
        else:
            ds_u = NIH_SSL_Dataset(
                data=data_u,
                labels=label_u,
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
                pin_memory=True
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
        data, labels = self.getValDataset(labeler_id, fold_idx)

        ds = NIH_SSL_Dataset(
            data=data,
            labels=labels,
            mode='test',
            imsize=imsize,
            image_container=self.imageContainer
        )
        dl = torch.utils.data.DataLoader(
            ds,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
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
        data, labels = self.getTestDataset(labeler_id, fold_idx)

        ds = NIH_SSL_Dataset(
            data=data,
            labels=labels,
            mode='test',
            imsize=imsize,
            image_container=self.imageContainer
        )
        dl = torch.utils.data.DataLoader(
            ds,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return dl

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
    def __init__(self, data, labels, mode, image_container=None, imsize=(128, 128)) -> None:
        self.image_ids = data
        self.labels = labels
        self.mode = mode
        self.image_container = image_container

        self.preprocess=False
        self.preload=True

        self.images = []
        
        self.loadImages()

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        trans_weak = T.Compose([
            T.Resize((imsize, imsize)),
            T.PadandRandomCrop(border=4, cropsize=(imsize, imsize)),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        trans_strong0 = T.Compose([
            T.Resize((imsize, imsize)),
            T.PadandRandomCrop(border=4, cropsize=(imsize, imsize)),
            T.RandomHorizontalFlip(p=0.5),
            RandomAugment(2, 10),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(imsize, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
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
                T.Resize((imsize, imsize)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            
    def loadImages(self):
        """
        Load all images
        """
        if self.image_container is not None:
            self.images = self.image_container.get_images_from_name(self.image_ids)
        else:
            print("No image container")
            for idx in range(len(self.image_ids)):
                if self.preprocess:
                    self.images.append(self.transformImage(self.loadImage(idx)))
                else:
                    self.images.append(self.loadImage(idx))

    def __getitem__(self, index: int):
        filename, label = self.image_ids[index], self.labels[index]
        im = self.images[index]
        return self.trans(im), label, filename

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