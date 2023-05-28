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

class BasicDataset():
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
        return self.data

class NIHDataset():
    """
    """
    def __init__(self, data: pd.DataFrame, transformation=None, preload=False, preprocess=False, param=None):
        self.data = data
        self.image_ids = data["Image ID"].values
        self.targets = data["GT"].values

        if transformation == None:
            self.tfms = transforms.Compose(
            [
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.tfms = transformation
            
        self.param = param
        self.PATH = param["PATH"]

        self.images = []
        
        self.preload = preload
        self.preprocess = preprocess
        
        if self.preload:
            self.loadImages()
            
    def loadImage(self, idx):
        """
        Load one single image
        """
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
        count = 0
        for idx in range(len(self.image_ids)):
            if count % 1000 == 0:
                print("Loaded " + str(count) + " images")
            count += 1
            if self.preprocess:
                self.images.append(self.transformImage(self.loadImage(idx)))
            else:
                self.images.append(self.loadImage(idx))
        print("Loading complete")
        
    def transformImage(self, image):
        """
        Transforms the image
        """
        return self.tfms(image).to(device)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        filename, target = self.image_ids[index], self.targets[index]
        img = self.getImage(index)
        if not self.preprocess:
            img = self.transformImage(img)
        return img, target, filename

    def __len__(self) -> int:
        #return len(self.images)
        return len(self.image_ids)

class NIH_K_Fold_Dataloader:
    def __init__(self, dataset, k=10, labelerIds=[4323195249, 4295194124], train_batch_size=8, test_batch_size=8,
                 seed=42, fraction=1.0, maxLabels=800, preload=False, preprocess=False, param=None):
        self.dataset = dataset.getData()
        self.k = k
        self.labelerIds = labelerIds
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed
        self.k_fold_datasets = []
        self.k_fold_patient_ids = []
        self.PATH=param["PATH"]
        self.preload = preload
        self.preprocess = preprocess
        self.param = param

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
        self.patient_ids = self.data["Patient ID"].unique()
        num_patient_images = self.data.drop_duplicates(subset=["Image ID"]).groupby(by="Patient ID", as_index=False).count()["Image ID"]
        self.patient_performance = pd.DataFrame({"Patient ID": patient_ids, "Num Patient Images": num_patient_images})
                     
        for labeler_id in self.labelerIds:
            temp = self.data[["Patient ID", "Image ID", "GT", str(labeler_id)]]
            temp["Expert_Correct"] = expert_labels["GT"] == self.data[str(labeler_id)]
            sum = temp[["Patient ID", "Expert_Correct"]].groupby(by="Patient ID", as_index=False).sum()
            sum.columns = ["Patient ID", f'{labeler_id}_num_correct']
            self.patient_performance = pd.merge(self.patient_performance, sum, left_on="Patient ID", right_on="Patient ID")
            self.patient_performance[f'{labeler_id}_perf'] = self.patient_performance[f'{labeler_id}_num_correct'] / self.patient_performance['Num Patient Images']

        target_temp = patient_performance[f'{labelerIds[0]}_perf'].astype(str)
        for labeler_id in labelerIds[1:]:
            target_temp = target_temp + "_" + patient_performance[f'{labeler_id}_perf'].astype(str)
        self.patient_performance["target"] = target_temp 

        self.expert_labels = self.data
        self._init_k_folds(maxLabels=maxLabels)

    def _init_k_folds(self, fraction=1.0, maxLabels=800):
        self.labels = self.expert_labels.drop_duplicates(subset=["Image ID"])
        self.labels = self.labels.fillna(0)
        self.labels = self.labels[["Patient ID", "Image ID", "GT"]]

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

            self.k_fold_datasets.append((expert_train, expert_val, expert_test))

    def get_data_loader_for_fold(self, fold_idx):
        expert_train, expert_val, expert_test = self.k_fold_datasets[fold_idx]

        expert_train_dataset = NIH_Dataset(self.PATH, expert_train, preload=self.preload, preprocess=self.preprocess)
        expert_val_dataset = NIH_Dataset(self.PATH, expert_val, preload=self.preload, preprocess=self.preprocess)
        expert_test_dataset = NIH_Dataset(self.PATH, expert_test, preload=self.preload, preprocess=self.preprocess)

        train_loader = torch.utils.data.DataLoader(dataset=expert_train_dataset, batch_size=self.train_batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset=expert_val_dataset, batch_size=self.test_batch_size, shuffle=True, drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=expert_test_dataset, batch_size=self.test_batch_size, shuffle=True, drop_last=False)
        return train_loader, val_loader, test_loader

    def get_dataset_for_folder(self, fold_idx):
        expert_train, expert_val, expert_test = self.k_fold_datasets[fold_idx]

        return expert_train, expert_val, expert_test
    
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