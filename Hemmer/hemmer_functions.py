import NIH.Dataset as ds
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

def mixture_of_ai_experts_loss(allocation_system_output, classifiers_outputs, targets, param=None):
    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, param["NUM_CLASSES"])).to(allocation_system_output.device)
    classifiers_outputs = classifiers_outputs.to(allocation_system_output.device)

    for idx in range(param["NUM_EXPERTS"]+1):
        team_probs = team_probs + allocation_system_output[:, idx].reshape(-1, 1) * classifiers_outputs[idx]

    log_output = torch.log(team_probs + 1e-7)
    moae_loss = nn.NLLLoss()(log_output, targets)

    return moae_loss

def mixture_of_human_experts_loss(allocation_system_output, human_expert_preds, targets, param=None):
    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, param["NUM_CLASSES"])).to(allocation_system_output.device)

    # human experts
    for idx in range(param["NUM_EXPERTS"]):
        one_hot_expert_preds = torch.tensor(np.eye(param["NUM_CLASSES"])[human_expert_preds[idx].astype(int)]).to(allocation_system_output.device)
        team_probs = team_probs + allocation_system_output[:, idx].reshape(-1, 1) * one_hot_expert_preds

    log_output = torch.log(team_probs + 1e-7)
    mohe_loss = nn.NLLLoss()(log_output, targets)

    return mohe_loss

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
    """expert_accuracies, experts_task_subset_accuracies, experts_coverages = get_experts_metrics(expert_preds, allocation_system_decisions, targets)

    for expert_idx, (expert_accuracy, expert_task_subset_accuracy, expert_coverage) in enumerate(zip(expert_accuracies, experts_task_subset_accuracies, experts_coverages)):
        metrics[f'Expert {expert_idx+1} Accuracy'] = expert_accuracy
        metrics[f'Expert {expert_idx+1} Task Subset Accuracy'] = expert_task_subset_accuracy
        metrics[f'Expert {expert_idx+1} Coverage'] = expert_coverage"""

    return system_accuracy, system_loss, metrics

def train_one_epoch(epoch, feature_extractor, classifier, allocation_system, train_loader, optimizer, scheduler, expert_fns, loss_fn, param=None):
    feature_extractor.eval()
    classifier.train()
    allocation_system.train()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, (batch_input, batch_targets, batch_filenames) in enumerate(train_loader):
        batch_targets = batch_targets.to(device)

        expert_batch_preds = np.empty((param["NUM_EXPERTS"], len(batch_targets)))
        for idx, expert_fn in enumerate(expert_fns):
            expert_batch_preds[idx] = np.array(expert_fn(batch_input, batch_targets, batch_filenames))

        batch_features = feature_extractor(batch_input)
        batch_outputs_classifier = classifier(batch_features)
        batch_outputs_allocation_system = allocation_system(batch_features)

        batch_loss = loss_fn(epoch=epoch, classifier_output=batch_outputs_classifier, allocation_system_output=batch_outputs_allocation_system,
                                expert_preds=expert_batch_preds, targets=batch_targets, param=param)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if param["USE_LR_SCHEDULER"]:
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

    return system_accuracy, system_loss, system_preds, allocation_system_decisions, targets

def run_team_performance_optimization(method, seed, nih_dataloader, expert_fns, param=None):
    print(f'Team Performance Optimization with {method}')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if method == "Joint Sparse Framework":
        loss_fn = joint_sparse_framework_loss
        allocation_system_activation_function = "sigmoid"


    elif method == "Our Approach":
        loss_fn = our_loss
        allocation_system_activation_function = "softmax"

    feature_extractor = Resnet().to(device)

    overall_allocation_system_decisions = []
    overall_system_preds = []
    overall_targets = []

    for fold_idx in range(param["K"]):
        print(f'Running fold {fold_idx+1} out of {param["K"]}')

        classifier = Network(output_size=param["NUM_CLASSES"],
                            softmax_sigmoid="softmax", param=param).to(device)

        allocation_system = Network(output_size=param["NUM_EXPERTS"] + 1,
                                 softmax_sigmoid=allocation_system_activation_function, param=param).to(device)

        train_loader, val_loader, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)

        parameters = list(classifier.parameters()) + list(allocation_system.parameters())
        optimizer = torch.optim.Adam(parameters, lr=param["LR"], betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, param["EPOCHS"] * len(train_loader))

        best_val_system_accuracy = 0
        best_val_system_loss = 100
        best_metrics = None

        for epoch in tqdm(range(1, param["EPOCHS"] + 1)):
            train_one_epoch(epoch, feature_extractor, classifier, allocation_system, train_loader, optimizer, scheduler, expert_fns, loss_fn, param)

            val_system_accuracy, val_system_loss, _, _, _ = evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, val_loader, expert_fns, loss_fn, param=param)
            _, _, test_system_preds, test_allocation_system_decisions, test_targets = evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, test_loader, expert_fns, loss_fn, param=param)

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

        overall_system_preds.extend(list(best_epoch_system_preds))
        overall_allocation_system_decisions.extend(list(best_epoch_allocation_system_decisions))
        overall_targets.extend(list(best_epoch_targets))

    system_accuracy = get_accuracy(overall_system_preds, overall_targets)
    classifier_coverage = np.sum([1 for dec in overall_allocation_system_decisions if dec==0])
    
    return system_accuracy, classifier_coverage

def get_accuracy_of_best_expert(seed, nih_dataloader, expert_fns, PATH, maxLabels=800, param=None):

    inputs_list = []
    targets_list = []
    filenames = []
    
    for fold_idx in range(param["K"]):
        print(f'Running fold {fold_idx+1} out of {param["K"]}')
        _, _, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)
  
        with torch.no_grad():
            for i, (batch_input, batch_targets, batch_filenames) in enumerate(test_loader):
                inputs_list.extend(batch_input)
                targets_list.extend(list(batch_targets.numpy()))
                filenames.extend(batch_filenames)

    expert_preds = np.empty((param["NUM_EXPERTS"], len(targets_list)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(inputs_list, targets_list, filenames))

    expert_accuracies = []
    for idx in range(param["NUM_EXPERTS"]):
        preds = expert_preds[idx]
        acc = accuracy_score(targets_list, preds)
        expert_accuracies.append(acc)

    print(f'Best Expert Accuracy: {max(expert_accuracies)}\n')

    return max(expert_accuracies)

def get_accuracy_of_average_expert(seed, nih_dataloader, expert_fns, PATH, maxLabels=800, param=None):

    inputs_list = []
    targets_list = []
    filenames = []

    for fold_idx in range(param["K"]):
        print(f'Running fold {fold_idx+1} out of {param["K"]}')
        _, _, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)
  
        with torch.no_grad():
            for i, (batch_input, batch_targets, batch_filenames) in enumerate(test_loader):
                inputs_list.extend(batch_input)
                targets_list.extend(list(batch_targets.numpy()))
                filenames.extend(batch_filenames)


    avg_expert = NihAverageExpert(expert_fns)
    avg_expert_preds = avg_expert.predict(inputs_list, targets_list, filenames)
    avg_expert_acc = accuracy_score(targets_list, avg_expert_preds)
    print(f'Average Expert Accuracy: {avg_expert_acc}\n')

    return avg_expert_acc

def train_full_automation_one_epoch(epoch, feature_extractor, classifier, train_loader, optimizer, scheduler, param=None):
    # switch to train mode
    feature_extractor.eval()
    classifier.train()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, (batch_input, batch_targets, _) in enumerate(train_loader):
        batch_targets = batch_targets.to(device)

        batch_features = feature_extractor(batch_input)
        batch_outputs_classifier = classifier(batch_features)

        log_output = torch.log(batch_outputs_classifier + 1e-7)
        batch_loss = nn.NLLLoss()(log_output, batch_targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if param["USE_LR_SCHEDULER"]:
            scheduler.step()

def evaluate_full_automation_one_epoch(epoch, feature_extractor, classifier, data_loader):
    feature_extractor.eval()
    classifier.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)

    with torch.no_grad():
        for i, (batch_input, batch_targets, _) in enumerate(data_loader):
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_classifier_outputs = classifier(batch_features)

            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))
            targets = torch.cat((targets, batch_targets))

    log_output = torch.log(classifier_outputs + 1e-7)
    full_automation_loss = nn.NLLLoss()(log_output, targets.long())

    classifier_outputs = classifier_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    classifier_preds = np.argmax(classifier_outputs, 1)

    return full_automation_loss, classifier_preds, targets

def run_full_automation(seed, nih_dataloader, PATH, maxLabels=800, param=None):
    print(f'Training full automation baseline')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_extractor = Resnet().to(device)

    overall_classifier_preds = []
    overall_targets = []


    for fold_idx in range(param["K"]):
        print(f'Running fold {fold_idx+1} out of {param["K"]}')

        classifier = Network(output_size=param["NUM_CLASSES"],
                            softmax_sigmoid="softmax", param=param).to(device)

        train_loader, val_loader, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)

        parameters = list(classifier.parameters())
        optimizer = torch.optim.Adam(parameters, lr=param["LR"], betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, param["EPOCHS"] * len(train_loader))

        best_val_system_loss = 100

        for epoch in tqdm(range(1, param["EPOCHS"] + 1)):
            train_full_automation_one_epoch(epoch, feature_extractor, classifier, train_loader, optimizer, scheduler)

            val_system_loss, _, _ = evaluate_full_automation_one_epoch(epoch, feature_extractor, classifier, val_loader)
            _, test_classifier_preds, test_targets = evaluate_full_automation_one_epoch(epoch, feature_extractor, classifier, test_loader)

            if val_system_loss < best_val_system_loss:
                best_val_system_loss = val_system_loss
                best_epoch_classifier_preds = test_classifier_preds
                best_epoch_targets = test_targets

        overall_classifier_preds.extend(list(best_epoch_classifier_preds))
        overall_targets.extend(list(best_epoch_targets))

    classifier_accuracy = get_accuracy(overall_classifier_preds, overall_targets)
    
    return classifier_accuracy

def train_moae_one_epoch(feature_extractor, classifiers, allocation_system, train_loader, optimizer, scheduler, param=None):
    # switch to train mode
    feature_extractor.eval()
    allocation_system.train()
    for classifier in classifiers:
        classifier.train()
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, (batch_input, batch_targets, batch_filenames) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        batch_features = feature_extractor(batch_input)
        batch_outputs_allocation_system = allocation_system(batch_features)
        batch_outputs_classifiers = torch.empty((param["NUM_EXPERTS"]+1, len(batch_targets), param["NUM_CLASSES"]))
        for idx, classifier in enumerate(classifiers):
            batch_outputs_classifiers[idx] = classifier(batch_features)

        # compute and record loss
        batch_loss = mixture_of_ai_experts_loss(allocation_system_output=batch_outputs_allocation_system,
                                                   classifiers_outputs=batch_outputs_classifiers, targets=batch_targets, param=param)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if param["USE_LR_SCHEDULER"]:
            scheduler.step()

def evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system, data_loader, param=None):
    feature_extractor.eval()
    allocation_system.eval()
    for classifier in classifiers:
        classifier.eval()
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifiers_outputs = torch.tensor([]).to(device)
    allocation_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).long().to(device)
    filenames = []

    with torch.no_grad():
        for i, (batch_input, batch_targets, batch_filenames) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_allocation_system_outputs = allocation_system(batch_features)
            batch_outputs_classifiers = torch.empty((param["NUM_EXPERTS"]+1, len(batch_targets), param["NUM_CLASSES"])).to(device)
            for idx, classifier in enumerate(classifiers):
                batch_outputs_classifiers[idx] = classifier(batch_features)

            classifiers_outputs = torch.cat((classifiers_outputs, batch_outputs_classifiers), dim=1)
            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_targets.float()))
            filenames.extend(batch_filenames)

    moae_loss = mixture_of_ai_experts_loss(allocation_system_output=allocation_system_outputs,
                                                   classifiers_outputs=classifiers_outputs, targets=targets.long(), param=param)

    classifiers_outputs = classifiers_outputs.cpu().numpy()
    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    classifiers_preds = np.argmax(classifiers_outputs, 2).T
    team_preds = classifiers_preds[range(len(classifiers_preds)), allocation_system_decisions.astype(int)]

    return moae_loss, team_preds, targets

def run_moae(seed, nih_dataloader, PATH, maxLabels=800, param=None):
    print(f'Training Mixture of artificial experts baseline')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_extractor = Resnet().to(device)

    overall_system_preds = []
    overall_targets = []

    for fold_idx in range(param["K"]):
        print(f'Running fold {fold_idx+1} out of {param["K"]}')

        allocation_system = Network(output_size=param["NUM_EXPERTS"] + 1,
                                 softmax_sigmoid="softmax", param=param).to(device)

        classifiers = []
        for _ in range(param["NUM_EXPERTS"]+1):
            classifier = Network(output_size=param["NUM_CLASSES"],
                                softmax_sigmoid="softmax", param=param).to(device)
            classifiers.append(classifier)

        train_loader, val_loader, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)

        parameters = list(allocation_system.parameters())
        for classifier in classifiers:
            parameters += list(classifier.parameters())

        optimizer = torch.optim.Adam(parameters, lr=param["LR"], betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, param["EPOCHS"] * len(train_loader))

        best_val_system_loss = 100

        for epoch in tqdm(range(1, param["EPOCHS"] + 1)):
            train_moae_one_epoch(feature_extractor, classifiers, allocation_system, train_loader, optimizer, scheduler)

            val_system_loss, _, _ = evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system, val_loader, param)
            _, test_system_preds, test_targets = evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system, test_loader, param)

            if val_system_loss < best_val_system_loss:
                best_val_system_loss = val_system_loss
                best_epoch_system_preds = test_system_preds
                best_epoch_targets = test_targets

        overall_system_preds.extend(list(best_epoch_system_preds))
        overall_targets.extend(list(best_epoch_targets))

    system_accuracy = get_accuracy(overall_system_preds, overall_targets)

    print(f'Mixture of Artificial Experts Accuracy: {system_accuracy}\n')
    return system_accuracy

def train_mohe_one_epoch(feature_extractor, allocation_system, train_loader, optimizer, scheduler, expert_fns, param=None):
    # switch to train mode
    feature_extractor.eval()
    allocation_system.train()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, (batch_input, batch_targets, batch_filenames) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        expert_batch_preds = np.empty((param["NUM_EXPERTS"], len(batch_targets)))
        for idx, expert_fn in enumerate(expert_fns):
            expert_batch_preds[idx] = np.array(expert_fn(batch_input, batch_targets, batch_filenames))

        batch_features = feature_extractor(batch_input)
        batch_outputs_allocation_system = allocation_system(batch_features)

        # compute and record loss
        batch_loss = mixture_of_human_experts_loss(allocation_system_output=batch_outputs_allocation_system,
                                                   human_expert_preds=expert_batch_preds, targets=batch_targets, param=param)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if param["USE_LR_SCHEDULER"]:
            scheduler.step()

def evaluate_mohe_one_epoch(feature_extractor, allocation_system, data_loader, expert_fns, param=None):
    feature_extractor.eval()
    allocation_system.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    allocation_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).long().to(device)

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
            batch_allocation_system_outputs = allocation_system(batch_features)

            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_targets.float()))

            

    expert_preds = np.empty((param["NUM_EXPERTS"], len(targets)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(inputs_list, targets_list, filenames))

    mohe_loss = mixture_of_human_experts_loss(allocation_system_output=allocation_system_outputs,
                                                   human_expert_preds=expert_preds, targets=targets.long(), param=param)

    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    expert_preds = expert_preds.T
    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    team_preds = expert_preds[range(len(expert_preds)), allocation_system_decisions.astype(int)-1]
    
    return mohe_loss, team_preds, targets

def run_mohe(seed, nih_dataloader, expert_fns, PATH, maxLabels=800, param=None):
    print(f'Training Mixture of human experts baseline')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_extractor = Resnet().to(device)

    overall_system_preds = []
    overall_targets = []

    for fold_idx in range(param["K"]):
        print(f'Running fold {fold_idx+1} out of {param["K"]}')

        allocation_system = Network(output_size=param["NUM_EXPERTS"] + 1,
                                 softmax_sigmoid="softmax", param=param).to(device)

        train_loader, val_loader, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)

        optimizer = torch.optim.Adam(allocation_system.parameters(), lr=param["LR"], betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, param["EPOCHS"] * len(train_loader))

        best_val_system_loss = 100

        for epoch in tqdm(range(1, param["EPOCHS"] + 1)):
            train_mohe_one_epoch(feature_extractor, allocation_system, train_loader, optimizer, scheduler, expert_fns)

            val_system_loss, _, _ = evaluate_mohe_one_epoch(feature_extractor, allocation_system, val_loader, expert_fns, param)
            _, test_system_preds, test_targets = evaluate_mohe_one_epoch(feature_extractor, allocation_system, test_loader, expert_fns, param)

            if val_system_loss < best_val_system_loss:
                best_val_system_loss = val_system_loss
                best_epoch_system_preds = test_system_preds
                best_epoch_targets = test_targets

        overall_system_preds.extend(list(best_epoch_system_preds))
        overall_targets.extend(list(best_epoch_targets))

    system_accuracy = get_accuracy(overall_system_preds, overall_targets)

    print(f'Mixture of Human Experts Accuracy: {system_accuracy}\n')
    return system_accuracy

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
    def __init__(self):
        super().__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)
        del self.resnet.fc

        if "resnet.pth" in os.listdir():
            print('load Resnet-18 checkpoint resnet.pth')
            print(self.resnet.load_state_dict(
                torch.load("resnet.pth"),
                strict=False))
        else:
            print('load Resnet-18 pretrained on ImageNet')

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
            nn.Dropout(param["DROPOUT"]),
            nn.Linear(512, param["NUM_HIDDEN_UNITS"]),
            nn.ReLU(),
            nn.Linear(param["NUM_HIDDEN_UNITS"], output_size)
        )

    def forward(self, features):
        output = self.classifier(features)
        if self.softmax_sigmoid == "softmax":
            output = nn.Softmax(dim=1)(output)
        elif self.softmax_sigmoid == "sigmoid":
            output = nn.Sigmoid()(output)
        return output