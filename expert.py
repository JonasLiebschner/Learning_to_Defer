import numpy as np
import pandas as pd

import copy
import json
import math
import os
import random
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data

import sklearn

class Expert:
    def __init__(self, dataset, labeler_id, modus, param=None, nLabels=800, prob=0.5):
        self.labelerId = labeler_id
        self.dataset = dataset
        self.data = dataset.getData()[["Image ID", str(self.labelerId)]]
        self.gt = dataset.getData()[["Image ID", "GT"]]
        self.gt_values = self.gt["GT"].unique()
        #self.data["Image ID"] = self.data["Image ID"].astype('category')
        self.nLabels = nLabels
        self.param = param
        self.prob = prob
        self.modus = modus
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.predictions = self.data
        self.predictions["Image ID"] = self.predictions["Image ID"].astype('category').copy()

        self.predictions = self.predictions.set_index("Image ID")
            
        self.prebuild_predictions = []
        self.prebuild_filenames = []

        self.prebuild_predictions_ssl = []
        self.prebuild_filenames_ssl = []

        self.prebuild_predictions_al = []
        self.prebuild_filenames_al = []

        self.predictions_dict = self.predictions.to_dict().get(str(self.labelerId))

    def predict(self, img, target, fnames):
        """
        img: the input image
        target: the GT label
        fname: filename (id for the image)
        """
        #return np.array([self.predictions[self.predictions["Image ID"] == image_id][str(self.labelerId)].values for image_id in fnames]).ravel()
        #return np.array([self.predictions.loc[self.predictions["Image ID"] == image_id, str(self.labelerId)].values[0] for image_id in fnames])

        #test_array = np.array([self.predictions.loc[self.predictions["Image ID"] == image_id, str(self.labelerId)].values[0] for image_id in fnames])
        #new_array = np.array(self.predictions.set_index("Image ID").loc[fnames, str(self.labelerId)])


        #assert (test_array == new_array).all()
        #return new_array
        if torch.is_tensor(fnames):
            fnames = fnames.tolist()
        #return np.array(self.predictions.loc[fnames, str(self.labelerId)])
        #old = np.array(self.predictions.loc[fnames, str(self.labelerId)])
        new = np.array([self.predictions_dict.get(fname) for fname in fnames])

        #assert np.array_equal(old,new), f"Predictions are not equal, Old: {old}, New {new}"

        return new

    def predictSLL(self, img, target, fnames):
        outputs = self.getSSLOutput(img, target, fnames)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

    def getSSLOutput(self, img, target, fnames):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            outputs, _ = self.sslModel.predict(img)
            return outputs

    def predictAL(self, img, target, fnames):
        return self.predictWithModel(img, target, fnames)

    def getALOutputs(self, img, target, fnames):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            outputs = self.alModel(img)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def get_model_prediction(self, model, img, target, fnames):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def loadModel(self, path, mod):
        model = torch.load(path)
        if mod == "SLL":
            self.sslModel = model
        elif mod == "AL":
            self.alModel = model

    def init_model_predictions(self, train_dataloader, mod, prediction_type):
        if mod == "SSL":
            for i, (input, target, hpred) in enumerate(train_dataloader):
                result = self.predictSLL(input.to(self.device), target, hpred).tolist()
                if prediction_type == "target":
                    self.prebuild_predictions_ssl += result
                elif prediction_type == "right":
                    target_prediction = []
                    for i in range(len(result)):
                        gt = self.gt[self.gt["Image ID"] == hpred[i]]["GT"].item()
                        if result[i] == 1:
                            target_prediction.append(gt)
                        else:
                            sample_list = self.gt_values.copy()
                            sample_list = np.delete(sample_list, np.where(sample_list == gt))
                            target_prediction.append(random.sample(list(sample_list), k=1)[0])
                    self.prebuild_predictions_ssl += target_prediction
                
                self.prebuild_filenames_ssl += hpred
            self.prebuild_ssl_dict = {filename.item(): prediction for filename, prediction in zip(self.prebuild_filenames_ssl, self.prebuild_predictions_ssl)}
        elif mod == "AL":
            for i, (input, target, hpred) in enumerate(train_dataloader):
                result = self.predictAL(input.to(self.device), target, hpred).tolist()
                if prediction_type == "target":
                    self.prebuild_predictions_al += result
                elif prediction_type == "right":
                    target_prediction = []
                    for i in range(len(result)):
                        gt = self.gt[self.gt["Image ID"] == hpred[i]]["GT"].item()
                        if result[i] == 1:
                            target_prediction.append(gt)
                        else:
                            sample_list = self.gt_values.copy()
                            sample_list = np.delete(sample_list, np.where(sample_list == gt))
                            target_prediction.append(random.sample(list(sample_list), k=1)[0])
                    self.prebuild_predictions_al += target_prediction
                    
                self.prebuild_filenames_al += hpred
            self.prebuild_al_dict = {filename.item(): prediction for filename, prediction in zip(self.prebuild_filenames_al, self.prebuild_predictions_al)}
            #self.filename_prediction_dict_al = {filename.item(): prediction for filename, prediction in zip(self.prebuild_filenames_ssl, self.prebuild_predictions_ssl)}

        # Create a dictionary to store filename-prediction pairs
        
    
    def predict_model_predefined(self, img, target, filenames, mod):
        if mod == "SSL":
            #return [self.prebuild_predictions_ssl[self.prebuild_filenames_ssl.index(filename)] for filename in filenames]
            return [self.prebuild_ssl_dict[filename.item()] for filename in filenames]
        elif mod == "AL":
            #return [self.prebuild_predictions_al[self.prebuild_filenames_al.index(filename)] for filename in filenames]
            return [self.prebuild_al_dict[filename.item()] for filename in filenames]

    def predict_model_predefined_al(self, img, target, filenames):
        #return [self.prebuild_predictions_al[self.prebuild_filenames_al.index(filename)] for filename in filenames]
        return [self.prebuild_al_dict[filename.item()] for filename in filenames]

    def predict_model_predefined_ssl(self, img, target, filenames):
        #return [self.prebuild_predictions_ssl[self.prebuild_filenames_ssl.index(filename)] for filename in filenames]
        return [self.prebuild_ssl_dict[filename.item()] for filename in filenames]

    def getModel(self, mod):
        if mod == "SSL":
            return self.sslModel
        elif mod == "AL":
            return self.alModel

    def saveModel(self, path, name, mod):
        if mod == "SSL":
            print("Not implemented")
            pass
            torch.save(self.sslModel, path + "/" + name + "_" + str(labeler_id))
        elif mod == "AL":
            torch.save(self.alModel, path + "/" + name + "_" + str(labeler_id))

    def setModel(self, model, mod):
        if mod == "SSL":
            self.sslModel = model
        elif mod == "AL":
            self.alModel = model

    """
    Old functions, here for compatibility and wraped from the new functions
    """
        
    def predictWithModel(self, img, target, filename):
        """
        Checks with the model if the expert would be correct
        If it predicts 1 than it returns the true label
        If it predicts 0 than is returns the opposit label
        """
        predicted = self.get_model_prediction(self.alModel, img, target, filename)
        #if prediction_type == "target":
        #    return redicted
        #elif prediction_type == "right"
        #result = []
        #target = target.cpu().detach().numpy()
        #for i, pred in enumerate(predicted):
        #    if pred == 1:
        #        result.append(target[i])
        #    else:
        #        result.append(1 - target[i])
        #return result
        return predicted

class SSLModel():
    def __init__(self, embedded_model, linear_model):
        super().__init__()
        self.embedded_model = embedded_model
        self.linear_model = linear_model

    def predict(self, imgs):
        embedding = self.embedded_model.get_embedding(batch=imgs)
        return self.linear_model(embedding)