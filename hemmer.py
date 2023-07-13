import expert as ex
import Dataset.Dataset as ds
#import NIH.hemmer_functions as nhf

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

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights

class NihAverageExpert:
    def __init__(self, expert_fns=[]):
        self.expert_fns = expert_fns
        self.num_experts = len(self.expert_fns)

    def predict(self, filenames):
        all_experts_predictions = [expert_fn(filenames) for expert_fn in self.expert_fns]
        predictions = [None] * len(filenames)

        for idx, expert_predictions in enumerate(all_experts_predictions):
            predictions[idx::self.num_experts] = expert_predictions[idx::self.num_experts]

        return predictions

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def L2D_Hemmer(train_loader, val_loader, test_loader, full_dataloader, expert_fns, param, seed, fold_idx, experts):

    data_loaders = (train_loader, val_loader, test_loader, full_dataloader)

    system_accuracy, classifier_coverage, all_train_metrics, all_val_metrics, all_test_metrics = run_team_performance_optimization("Our Approach", seed, data_loaders, expert_fns, param=param, fold=fold_idx, experts=experts)
    
    #jsf_accuracy, jsf_coverage = run_team_performance_optimization("Joint Sparse Framework", seed, data_loaders, expert_fns, param=param, fold=fold_idx)

    #if param["NEPTUNE"]["NEPTUNE"]:
    #    run[f"Seed_{seed}/hemmer_approach_accuracy"].append(our_approach_accuracy)
    #    run[f"Seed_{seed}/hemmer_approach_coverage"].append(our_approach_coverage)
    # 
    #    run[f"seed_{seed}/jsf_accuracy"].append(jsf_accuracy)
    #    run[f"seed_{seed}/jsf_accuracy"].append(jsf_coverage)

    print("-"*40)

    return system_accuracy, classifier_coverage, all_train_metrics, all_val_metrics, all_test_metrics


def run_team_performance_optimization(method, seed, data_loaders, expert_fns, param=None, fold=None, experts=None):
    print(f'Team Performance Optimization with {method}')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if method == "Joint Sparse Framework":
        loss_fn = joint_sparse_framework_loss
        allocation_system_activation_function = "sigmoid"


    elif method == "Our Approach":
        loss_fn = our_loss
        allocation_system_activation_function = "softmax"

    labelerIds = []
    for labelerId, expert in experts.items():
        labelerIds.append(labelerId)

    feature_extractor = Resnet().to(device)

    overall_allocation_system_decisions = []
    overall_system_preds = []
    overall_targets = []

    classifier = Network(output_size=param["NUM_CLASSES"], softmax_sigmoid="softmax", param=param).to(device)

    allocation_system = Network(output_size=param["NUM_EXPERTS"] + 1, softmax_sigmoid=allocation_system_activation_function, param=param).to(device)

    train_loader, val_loader, test_loader, full_loader = data_loaders

    parameters = list(classifier.parameters()) + list(allocation_system.parameters())
    optimizer = torch.optim.Adam(parameters, lr=param["L2D"]["HEMMER"]["LR"], betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, param["L2D"]["HEMMER"]["EPOCHS"] * len(train_loader))

    best_val_system_accuracy = 0
    best_val_system_loss = 100
    best_metrics = None

    all_train_metrics = {}
    all_val_metrics = {}
    all_test_metrics = {}

    for epoch in tqdm(range(1, param["L2D"]["HEMMER"]["EPOCHS"] + 1)):
        train_one_epoch(epoch, feature_extractor, classifier, allocation_system, train_loader, optimizer, scheduler, expert_fns, loss_fn, param)

        train_system_accuracy, train_system_loss, train_system_preds, train_allocation_system_decisions, train_targets, train_metrics = evaluate_one_epoch(epoch, feature_extractor, 
                                                                                                                                                           classifier, allocation_system, 
                                                                                                                                                           train_loader, expert_fns, 
                                                                                                                                                           loss_fn, param=param)

        val_system_accuracy, val_system_loss, val_system_preds, val_allocation_system_decisions, val_targets, val_metrics = evaluate_one_epoch(epoch, feature_extractor, classifier, 
                                                                                                                                               allocation_system, val_loader, 
                                                                                                                                               expert_fns, loss_fn, param=param)
        test_system_accuracy, test_system_loss, test_system_preds, test_allocation_system_decisions, test_targets, test_metrics = evaluate_one_epoch(epoch, feature_extractor, 
                                                                                                                                                     classifier, allocation_system, 
                                                                                                                                                     test_loader, 
                                                                                                                                                     expert_fns, loss_fn, param=param)

        if method == "Joint Sparse Framework":
            if val_system_accuracy > best_val_system_accuracy:
                best_val_system_accuracy = val_system_accuracy
                best_epoch_system_preds = test_system_preds
                best_epoch_allocation_system_decisions = test_allocation_system_decisions
                best_epoch_targets = test_targets

        elif method == "Our Approach":
            if val_system_loss < best_val_system_loss:
                best_val_system_loss = val_system_loss
                best_epoch_system_preds = test_system_preds
                best_epoch_allocation_system_decisions = test_allocation_system_decisions
                best_epoch_targets = test_targets

        if param["NEPTUNE"]["NEPTUNE"]:
            run = param["NEPTUNE"]["RUN"]

            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Train" + "/system_accuracy"].append(train_system_accuracy)
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Train" + "/train_system_loss"].append(train_system_loss)
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Train" + "/Classifier Accuracy"].append(train_metrics["Classifier Accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Train" + "/Classifier Task Subset Accuracy"].append(train_metrics["Classifier Task Subset Accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Train" + "/Classifier Coverage"].append(train_metrics["Classifier Coverage"])
            for expert_idx in range(len(expert_fns)):
                run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Train/Expert_{labelerIds[expert_idx]}" + "/expert_accuracy"].append(metrics[f'Expert {expert_idx+1} Accuracy'])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Train/Expert_{labelerIds[expert_idx]}" + "/expert_task_subset_accuracy"].append(metrics[f'Expert {expert_idx+1} Task Subset Accuracy'])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Train/Expert_{labelerIds[expert_idx]}" + "/expert_coverage"].append(metrics[f'Expert {expert_idx+1} Coverage'])

            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Val" + "/system_accuracy"].append(val_system_accuracy)
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Val" + "/train_system_loss"].append(val_system_loss)
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Val" + "/Classifier Accuracy"].append(val_metrics["Classifier Accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Val" + "/Classifier Task Subset Accuracy"].append(val_metrics["Classifier Task Subset Accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Val" + "/Classifier Coverage"].append(val_metrics["Classifier Coverage"])

            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Test" + "/system_accuracy"].append(test_system_accuracy)
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Test" + "/train_system_loss"].append(test_system_loss)
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Test" + "/Classifier Accuracy"].append(test_metrics["Classifier Accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Test" + "/Classifier Task Subset Accuracy"].append(test_metrics["Classifier Task Subset Accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Hemmer/Test" + "/Classifier Coverage"].append(test_metrics["Classifier Coverage"])

        all_train_metrics[epoch] = {}
        all_val_metrics[epoch] = {}
        all_test_metrics[epoch] = {}

        all_train_metrics[epoch]["system_accuracy"] = train_system_accuracy
        all_train_metrics[epoch]["train_system_loss"] = train_system_loss
        all_train_metrics[epoch]["Classifier Accuracy"] = train_metrics["Classifier Accuracy"]
        all_train_metrics[epoch]["Classifier Task Subset Accuracy"] = train_metrics["Classifier Task Subset Accuracy"]
        all_train_metrics[epoch]["Classifier Coverage"] = train_metrics["Classifier Coverage"]

        all_val_metrics[epoch]["system_accuracy"] = val_system_accuracy
        all_val_metrics[epoch]["train_system_loss"] = val_system_loss
        all_val_metrics[epoch]["Classifier Accuracy"] = val_metrics["Classifier Accuracy"]
        all_val_metrics[epoch]["Classifier Task Subset Accuracy"] = val_metrics["Classifier Task Subset Accuracy"]
        all_val_metrics[epoch]["Classifier Coverage"] = val_metrics["Classifier Coverage"]

        all_test_metrics[epoch]["system_accuracy"] = test_system_accuracy
        all_test_metrics[epoch]["train_system_loss"] = test_system_loss
        all_test_metrics[epoch]["Classifier Accuracy"] = test_metrics["Classifier Accuracy"]
        all_test_metrics[epoch]["Classifier Task Subset Accuracy"] = test_metrics["Classifier Task Subset Accuracy"]
        all_test_metrics[epoch]["Classifier Coverage"] = test_metrics["Classifier Coverage"]

        

    overall_system_preds.extend(list(best_epoch_system_preds))
    overall_allocation_system_decisions.extend(list(best_epoch_allocation_system_decisions))
    overall_targets.extend(list(best_epoch_targets))

    system_accuracy = get_accuracy(overall_system_preds, overall_targets)
    classifier_coverage = np.sum([1 for dec in overall_allocation_system_decisions if dec==0])
    
    return system_accuracy, classifier_coverage, all_train_metrics, all_val_metrics, all_test_metrics

def train_one_epoch(epoch, feature_extractor, classifier, allocation_system, train_loader, optimizer, scheduler, expert_fns, loss_fn, param=None):
    feature_extractor.eval()
    classifier.train()
    allocation_system.train()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, (batch_input, batch_targets, batch_filenames) in enumerate(train_loader):
        batch_targets = batch_targets.to(device)
        batch_input = batch_input.to(device)

        expert_batch_preds = np.empty((param["NUM_EXPERTS"], len(batch_targets)))
        for idx, expert_fn in enumerate(expert_fns):
            expert_batch_preds[idx] = np.array(expert_fn(batch_input, batch_targets, batch_filenames))

        batch_features = feature_extractor(batch_input.to(device))
        batch_outputs_classifier = classifier(batch_features)
        batch_outputs_allocation_system = allocation_system(batch_features)

        batch_loss = loss_fn(epoch=epoch, classifier_output=batch_outputs_classifier, allocation_system_output=batch_outputs_allocation_system,
                                expert_preds=expert_batch_preds, targets=batch_targets, param=param)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if param["L2D"]["HEMMER"]["USE_LR_SCHEDULER"]:
            scheduler.step()

def evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, data_loader, expert_fns, loss_fn, param=None):
    feature_extractor.eval()
    classifier.eval()
    allocation_system.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier_outputs = torch.tensor([]).to(device)
    allocation_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)

    inputs_list = []
    targets_list = []
    filenames = []

    with torch.no_grad():
        for i, (batch_input, batch_targets, batch_filenames) in enumerate(data_loader):

            inputs_list.extend(batch_input)
            targets_list.extend(list(batch_targets.numpy()))
            filenames.extend(batch_filenames)
            
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_classifier_outputs = classifier(batch_features)
            batch_allocation_system_outputs = allocation_system(batch_features)

            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))
            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_targets))

    expert_preds = np.empty((param["NUM_EXPERTS"], len(targets)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(inputs_list, targets_list, filenames))

    classifier_outputs = classifier_outputs.cpu().numpy()
    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    classifier_preds = np.argmax(classifier_outputs, 1)
    preds = np.vstack((classifier_preds, expert_preds)).T
    system_preds = preds[range(len(preds)), allocation_system_decisions.astype(int)]

    system_accuracy, system_loss, metrics = get_metrics(epoch, allocation_system_outputs, classifier_outputs, expert_preds, targets, loss_fn, param)

    return system_accuracy, system_loss, system_preds, allocation_system_decisions, targets, metrics

            


def joint_sparse_framework_loss(epoch, classifier_output, allocation_system_output, expert_preds, targets, param=None):
    # Input:
    #   epoch: int = current epoch (used for epoch-dependent weighting of allocation system loss)
    #   classifier_output: softmax probabilities as class probabilities,  nxm matrix with n=batch size, m=number of classes
    #   allocation_system_output: sigmoid outputs as expert weights,  nx(m+1) matrix with n=batch size, m=number of experts + 1 for machine
    #   expert_preds: nxm matrix with expert predictions with n=number of experts, m=number of classes
    #   targets: targets as 1-dim vector with n length with n=batch_size

    # loss for allocation system 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set up zero-initialized tensor to store weighted team predictions
    batch_size = len(targets)
    weighted_team_preds = torch.zeros((batch_size, param["NUM_CLASSES"])).to(classifier_output.device)

    # for each team member add the weighted prediction to the team prediction
    # start with machine
    weighted_team_preds = weighted_team_preds + allocation_system_output[:, 0].reshape(-1, 1) * classifier_output
    # continue with human experts
    for idx in range(param["NUM_EXPERTS"]):
        one_hot_expert_preds = torch.tensor(np.eye(param["NUM_CLASSES"])[expert_preds[idx].astype(int)]).to(classifier_output.device)
        weighted_team_preds = weighted_team_preds + allocation_system_output[:, idx + 1].reshape(-1, 1) * one_hot_expert_preds

    # calculate team probabilities using softmax
    team_probs = nn.Softmax(dim=1)(weighted_team_preds)

    # alpha2 is 1-epoch^0.5 (0.5 taken from code of preprint paper) <--- used for experiments
    alpha2 = 1 - (epoch ** -0.5)
    alpha2 = torch.tensor(alpha2).to(classifier_output.device)

    # weight the negative log likelihood loss with alpha2 to get team loss
    log_team_probs = torch.log(team_probs + 1e-7)
    allocation_system_loss = nn.NLLLoss(reduction="none")(log_team_probs, targets.long())
    allocation_system_loss = torch.mean(alpha2 * allocation_system_loss)

    # loss for classifier

    alpha1 = 1
    log_classifier_output = torch.log(classifier_output + 1e-7)
    classifier_loss = nn.NLLLoss(reduction="none")(log_classifier_output, targets.long())
    classifier_loss = alpha1 * torch.mean(classifier_loss)

    # combine both losses
    system_loss = classifier_loss + allocation_system_loss

    return system_loss

def our_loss(epoch, classifier_output, allocation_system_output, expert_preds, targets, param=None):
    # Input:
    #   epoch: int = current epoch (not used, just to have the same function parameters as with JSF loss)
    #   classifier_output: softmax probabilities as class probabilities,  nxm matrix with n=batch size, m=number of classes
    #   allocation_system_output: softmax outputs as weights,  nx(m+1) matrix with n=batch size, m=number of experts + 1 for machine
    #   expert_preds: nxm matrix with expert predictions with n=number of experts, m=number of classes
    #   targets: targets as 1-dim vector with n length with n=batch_size
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, param["NUM_CLASSES"])).to(classifier_output.device) # set up zero-initialized tensor to store team predictions
    team_probs = team_probs + allocation_system_output[:, 0].reshape(-1, 1) * classifier_output # add the weighted classifier prediction to the team prediction
    for idx in range(param["NUM_EXPERTS"]): # continue with human experts
        one_hot_expert_preds = torch.tensor(np.eye(param["NUM_CLASSES"])[expert_preds[idx].astype(int)]).to(classifier_output.device)
        team_probs = team_probs + allocation_system_output[:, idx + 1].reshape(-1, 1) * one_hot_expert_preds

    log_output = torch.log(team_probs + 1e-7)
    system_loss = nn.NLLLoss()(log_output, targets)

    return system_loss

def get_metrics(epoch, allocation_system_outputs, classifier_outputs, expert_preds, targets, loss_fn, param=None):
    metrics = {}

    # Metrics for system
    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    classifier_preds = np.argmax(classifier_outputs, 1)
    preds = np.vstack((classifier_preds, expert_preds)).T
    system_preds = preds[range(len(preds)), allocation_system_decisions.astype(int)]
    system_accuracy = get_accuracy(system_preds, targets)

    system_loss = loss_fn(epoch=epoch,
                          classifier_output=torch.tensor(classifier_outputs).float(),
                          allocation_system_output=torch.tensor(allocation_system_outputs).float(),
                          expert_preds=expert_preds,
                          targets=torch.tensor(targets).long(), param=param)

    metrics["System Accuracy"] = system_accuracy
    metrics["System Loss"] = system_loss

    # Metrics for classifier
    classifier_accuracy, classifier_task_subset_accuracy, classifier_coverage = get_classifier_metrics(classifier_preds, allocation_system_decisions, targets)
    metrics["Classifier Accuracy"] = classifier_accuracy
    metrics["Classifier Task Subset Accuracy"] = classifier_task_subset_accuracy
    metrics["Classifier Coverage"] = classifier_coverage

    # Metrics for experts 
    expert_accuracies, experts_task_subset_accuracies, experts_coverages = get_experts_metrics(expert_preds, allocation_system_decisions, targets, param=param)

    for expert_idx, (expert_accuracy, expert_task_subset_accuracy, expert_coverage) in enumerate(zip(expert_accuracies, experts_task_subset_accuracies, experts_coverages)):
        metrics[f'Expert {expert_idx+1} Accuracy'] = expert_accuracy
        metrics[f'Expert {expert_idx+1} Task Subset Accuracy'] = expert_task_subset_accuracy
        metrics[f'Expert {expert_idx+1} Coverage'] = expert_coverage

    return system_accuracy, system_loss, metrics

def get_accuracy(preds, targets):
    if len(targets) > 0:
        acc = accuracy_score(targets, preds)
    else:
        acc = 0

    return acc

def get_coverage(task_subset_targets, targets):
    num_images = len(targets)
    num_images_in_task_subset = len(task_subset_targets)
    coverage = num_images_in_task_subset / num_images

    return coverage

def get_classifier_metrics(classifier_preds, allocation_system_decisions, targets):
    # classifier performance on all tasks
    classifier_accuracy = get_accuracy(classifier_preds, targets)

    # filter for subset of tasks that are allocated to the classifier
    task_subset = (allocation_system_decisions == 0)

    # classifier performance on those tasks
    task_subset_classifier_preds = classifier_preds[task_subset]
    task_subset_targets = targets[task_subset]
    classifier_task_subset_accuracy = get_accuracy(task_subset_classifier_preds, task_subset_targets)

    # coverage
    classifier_coverage = get_coverage(task_subset_targets, targets)

    return classifier_accuracy, classifier_task_subset_accuracy, classifier_coverage

def get_experts_metrics(expert_preds, allocation_system_decisions, targets, param=None):
    expert_accuracies = []
    expert_task_subset_accuracies = []
    expert_coverages = []

    # calculate metrics for each expert
    for expert_idx in range(param["NUM_EXPERTS"]):

        # expert performance on all tasks
        preds = expert_preds[expert_idx]
        expert_accuracy = get_accuracy(preds, targets)

        # filter for subset of tasks that are allocated to the expert with number "idx"
        task_subset = (allocation_system_decisions == expert_idx+1)

        # expert performance on tasks assigned by allocation system
        task_subset_expert_preds = preds[task_subset]
        task_subset_targets = targets[task_subset]
        expert_task_subset_accuracy = get_accuracy(task_subset_expert_preds, task_subset_targets)

        # coverage
        expert_coverage = get_coverage(task_subset_targets, targets)

        expert_accuracies.append(expert_accuracy)
        expert_task_subset_accuracies.append(expert_task_subset_accuracy)
        expert_coverages.append(expert_coverage)

    return expert_accuracies, expert_task_subset_accuracies, expert_coverages


class NihAverageExpert:
    def __init__(self, expert_fns=[]):
        self.expert_fns = expert_fns
        self.num_experts = len(self.expert_fns)

    def predict(self, input, targets, filenames):
        all_experts_predictions = [expert_fn(input, targets, filenames) for expert_fn in self.expert_fns]
        predictions = [None] * len(filenames)

        for idx, expert_predictions in enumerate(all_experts_predictions):
            predictions[idx::self.num_experts] = expert_predictions[idx::self.num_experts]

        return predictions
    
class Resnet(torch.nn.Module):
    def __init__(self, type="18"):
        super().__init__()

        if type == "18":
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            #self.resnet = torchvision.models.resnet18(pretrained=True)
        elif type == "50":
            self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        del self.resnet.fc

        if "resnet_" + type + ".pth" in os.listdir():
            print('load Resnet-' + type + ' checkpoint resnet.pth')
            print(self.resnet.load_state_dict(
                torch.load("resnet" + type + ".pth"),
                strict=False))
        else:
            print('load Resnet-' + type +' pretrained on ImageNet')

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.training = False

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        features = torch.flatten(x, 1)
        return features
    
class Network(nn.Module):
    def __init__(self, output_size, softmax_sigmoid="softmax", param=None):
        super().__init__()
        self.softmax_sigmoid = softmax_sigmoid

        self.classifier = nn.Sequential(
            nn.Dropout(param["L2D"]["HEMMER"]["DROPOUT"]),
            nn.Linear(512, param["L2D"]["HEMMER"]["NUM_HIDDEN_UNITS"]),
            nn.ReLU(),
            nn.Linear(param["L2D"]["HEMMER"]["NUM_HIDDEN_UNITS"], output_size)
        )

    def forward(self, features):
        output = self.classifier(features)
        if self.softmax_sigmoid == "softmax":
            output = nn.Softmax(dim=1)(output)
        elif self.softmax_sigmoid == "sigmoid":
            output = nn.Sigmoid()(output)
        return output