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
import urllib

import tarfile

import sys


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
    if not os.path.exists(PATH + "images/"):
        os.makedirs(PATH + "images/")
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

def main(args):
    path = args[0]

    print(f"Path for data: {path}")

    if path[-1] != "/":
        path += "/"

    answer = input("Continue?")
    if answer.lower() in ["y","yes"]:
        downloadData(path)
        unpackData(path)
        moveData(path)
        downloadLabels(path)
    elif answer.lower() in ["n","no"]:
        return


if __name__ == "__main__":
    main(sys.argv[1:])