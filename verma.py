#import Verma.experts as vexp
import Verma.losses as vlos
from Verma.utils import AverageMeter, accuracy
import Verma.resnet50 as vres

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

import sklearn
import copy

import gc
from torch.utils.data import DataLoader


import torchvision.transforms as transforms
from PIL import Image


def train(model, train_loader, valid_loader, test_loader, expert_fns, config, seed="", experts=None, fold=None, full_dataloader=None, param=None):

    print("Start L2D Training")

    NEPTUNE = config["NEPTUNE"]["NEPTUNE"]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    n_classes = config["n_classes"] + len(expert_fns)
    kwargs = {"num_workers": 0, "pin_memory": True}

    model = model.to(device)
    #cudnn.benchmark = True
    optimizer = torch.optim.Adam(
        model.parameters(), config["lr"], weight_decay=config["weight_decay"]
    )
    criterion = vlos.Criterion()
    loss_fn = getattr(criterion, config["loss_type"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader) * config["epochs"]
    )
    best_validation_loss = np.inf
    patience = 0
    iters = 0
    warmup_iters = config["warmup_epochs"] * len(train_loader)
    lrate = config["lr"]

    metrics_pretrain_all = {}
    metrics_train_all = {}
    metrics_val_all = {}
    metrics_test_all = {}
    metrics_full_all = {}

    for epoch in range(0, param["epochs_pretrain"]):
        iters, train_loss = train_epoch(
            iters,
            warmup_iters,
            lrate,
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            expert_fns,
            loss_fn,
            n_classes,
            config["alpha"],
            config,
            classifier_only=True
        )

        experts_fns_eval = []
        labelerIds = []
        for labelerId, expert in experts.items():
            experts_fns_eval.append(expert.predict)
            labelerIds.append(labelerId)
        #metrics = evaluate(model, expert_fns, loss_fn, n_classes, valid_loader, config)

        metrics_pretrain = evaluate(model, experts_fns_eval, loss_fn, n_classes, train_loader, config, print_m=True)
        if param["NEPTUNE"]["NEPTUNE"]:
            run = param["NEPTUNE"]["RUN"]
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/Classifier_only/system_accuracy"].append(metrics_pretrain["system_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/Classifier_only/expert_accuracy"].append(metrics_pretrain["expert_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/Classifier_only/classifier_accuracy"].append(metrics_pretrain["classifier_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/Classifier_only/alone_classifier"].append(metrics_pretrain["alone_classifier"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/Classifier_only/validation_loss"].append(metrics_pretrain["validation_loss"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/Classifier_only/cov_classifier"].append(metrics_pretrain["cov_classifier"])
            for index in range(len(metrics_pretrain["cov_experts"])):
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/Classifier_only/accuracy_expert_{labelerIds[index]}"].append(metrics_pretrain["acc_experts"][index])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/Classifier_only/cov_expert_{labelerIds[index]}"].append(metrics_pretrain["cov_experts"][index])
                
        metrics_pretrain_all[epoch] = metrics_pretrain
    epoch = 0

    for epoch in range(0, config["epochs"]):
        iters, train_loss = train_epoch(
            iters,
            warmup_iters,
            lrate,
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            expert_fns,
            loss_fn,
            n_classes,
            config["alpha"],
            config,
        )

        experts_fns_eval = []
        labelerIds = []
        for labelerId, expert in experts.items():
            experts_fns_eval.append(expert.predict)
            labelerIds.append(labelerId)
        #metrics = evaluate(model, expert_fns, loss_fn, n_classes, valid_loader, config)

        metrics_train = evaluate(model, experts_fns_eval, loss_fn, n_classes, train_loader, config, print_m=False)
        if param["NEPTUNE"]["NEPTUNE"]:
            run = param["NEPTUNE"]["RUN"]
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/system_accuracy"].append(metrics_train["system_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/expert_accuracy"].append(metrics_train["expert_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/classifier_accuracy"].append(metrics_train["classifier_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/alone_classifier"].append(metrics_train["alone_classifier"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/validation_loss"].append(metrics_train["validation_loss"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/cov_classifier"].append(metrics_train["cov_classifier"])
            for index in range(len(metrics_train["cov_experts"])):
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/accuracy_expert_{labelerIds[index]}"].append(metrics_train["acc_experts"][index])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Train/cov_expert_{labelerIds[index]}"].append(metrics_train["cov_experts"][index])
        
        metrics_val = evaluate(model, experts_fns_eval, loss_fn, n_classes, valid_loader, config)

        if NEPTUNE:
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Val/system_accuracy"].append(metrics_val["system_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Val/expert_accuracy"].append(metrics_val["expert_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Val/classifier_accuracy"].append(metrics_val["classifier_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Val/alone_classifier"].append(metrics_val["alone_classifier"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Val/validation_loss"].append(metrics_val["validation_loss"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Val/cov_classifier"].append(metrics_val["cov_classifier"])
            for index in range(len(metrics_train["cov_experts"])):
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Val/accuracy_expert_{labelerIds[index]}"].append(metrics_val["acc_experts"][index])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Val/cov_expert_{labelerIds[index]}"].append(metrics_val["cov_experts"][index])

        metrics_test = evaluate(model, experts_fns_eval, loss_fn, n_classes, test_loader, config, print_m=False)
        if param["NEPTUNE"]["NEPTUNE"]:
            run = param["NEPTUNE"]["RUN"]
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/system_accuracy"].append(metrics_test["system_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/expert_accuracy"].append(metrics_test["expert_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/classifier_accuracy"].append(metrics_test["classifier_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/alone_classifier"].append(metrics_test["alone_classifier"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/validation_loss"].append(metrics_test["validation_loss"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/cov_classifier"].append(metrics_test["cov_classifier"])
            for index in range(len(metrics_train["cov_experts"])):
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/accuracy_expert_{labelerIds[index]}"].append(metrics_test["acc_experts"][index])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/cov_expert_{labelerIds[index]}"].append(metrics_test["cov_experts"][index])

        metrics_full = None
        if full_dataloader is not None:
            metrics_full = evaluate(model, experts_fns_eval, loss_fn, n_classes, full_dataloader, config, print_m=False)
            if param["NEPTUNE"]["NEPTUNE"]:
                run = param["NEPTUNE"]["RUN"]
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/system_accuracy_all"].append(metrics_full["system_accuracy"])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/expert_accuracy_all"].append(metrics_full["expert_accuracy"])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/classifier_accuracy_all"].append(metrics_full["classifier_accuracy"])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/alone_classifier_all"].append(metrics_full["alone_classifier"])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/validation_loss_all"].append(metrics_full["validation_loss"])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/cov_classifier_all"].append(metrics_full["cov_classifier"])
                for index in range(len(metrics_train["cov_experts"])):
                    run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/accuracy_expert_{labelerIds[index]}"].append(metrics_full["acc_experts"][index])
                    run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/cov_expert_{labelerIds[index]}"].append(metrics_full["cov_experts"][index])

       
        metrics_train_all[epoch] = metrics_train
        metrics_val_all[epoch] = metrics_val
        metrics_test_all[epoch] = metrics_test
        metrics_full_all[epoch] = metrics_full

        validation_loss = metrics_val["validation_loss"]

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            """
            print(
                "Saving the model with classifier accuracy {}".format(
                    metrics["classifier_accuracy"]
                ),
                flush=True,
            )
            save_path = os.path.join(
                config["ckp_dir"],
                config["experiment_name"]
                + "_"
                + str(len(expert_fns))
                + "_experts"
                + "_seed_"
                + str(seed),
            )"""
            #torch.save(model.state_dict(), save_path + ".pt")
            # Additionally save the whole config dict
            #with open(save_path + ".json", "w") as f:
            #    json.dump(config, f)
            patience = 0
        else:
            print("Losses")
            print(best_validation_loss)
            print(validation_loss)
            patience += 1

        if patience >= config["patience"]:
            print("Early Exiting Training.", flush=True)
            break
            
    print("Evaluate on Test Data")
    metrics_test = evaluate(model, experts_fns_eval, loss_fn, n_classes, test_loader, config)
    if param["NEPTUNE"]["NEPTUNE"]:
        run = param["NEPTUNE"]["RUN"]
        run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/system_accuracy"].append(metrics_test["system_accuracy"])
        run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/expert_accuracy"].append(metrics_test["expert_accuracy"])
        run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/classifier_accuracy"].append(metrics_test["classifier_accuracy"])
        run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/alone_classifier"].append(metrics_test["alone_classifier"])
        run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/validation_loss"].append(metrics_test["validation_loss"])
        run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/cov_classifier"].append(metrics_test["cov_classifier"])
        for index in range(len(metrics_train["cov_experts"])):
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/accuracy_expert_{labelerIds[index]}"].append(metrics_train["acc_experts"][index])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Test/cov_expert_{labelerIds[index]}"].append(metrics_train["cov_experts"][index])

    metrics_full = None
    if full_dataloader is not None:
        print("Test on all Data")
        metrics_full = evaluate(model, experts_fns_eval, loss_fn, n_classes, full_dataloader, config)
        if param["NEPTUNE"]["NEPTUNE"]:
            run = param["NEPTUNE"]["RUN"]
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/system_accuracy_all"].append(metrics_full["system_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/expert_accuracy_all"].append(metrics_full["expert_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/classifier_accuracy_all"].append(metrics_full["classifier_accuracy"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/alone_classifier_all"].append(metrics_full["alone_classifier"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/validation_loss_all"].append(metrics_full["validation_loss"])
            run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/cov_classifier_all"].append(metrics_full["cov_classifier"])
            for index in range(len(metrics_train["cov_experts"])):
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/accuracy_expert_{labelerIds[index]}"].append(metrics_train["acc_experts"][index])
                run[f"Seed_{seed}/Fold_{fold}/L2D/Verma/Full/cov_expert_{labelerIds[index]}"].append(metrics_train["cov_experts"][index])
        
    return metrics_train_all, metrics_val_all, metrics_test_all, metrics_full_all, metrics_pretrain_all

def train_epoch(
    iters,
    warmup_iters,
    lrate,
    train_loader,
    model,
    optimizer,
    scheduler,
    epoch,
    expert_fns,
    loss_fn,
    n_classes,
    alpha,
    config,
    classifier_only=False
):
    """ Train for one epoch """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    epoch_train_loss = []

    for i, (input, target, hpred) in enumerate(train_loader):
        if iters < warmup_iters:
            lr = lrate * float(iters) / warmup_iters
            #print(iters, lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        target_cpu = target
        target = target.to(device)
        input = input.to(device)
        hpred = hpred

        # compute output
        output = model(input)

        if config["loss_type"] == "softmax":
            output = F.softmax(output, dim=1)

        # get expert  predictions and costs
        #batch_size = output.size()[0]  # batch_size
        batch_size = output.size(0)
        collection_Ms = []
        # We only support \alpha=1
        for _, fn in enumerate(expert_fns):
            # We assume each expert function has access to the extra metadata, even if they don't use it.
            m = fn(input, target, hpred)
            #m = fn(hpred)
            
            m2 = [0] * batch_size
            for j in range(0, batch_size):
                #if m[j] == target[j].item():
                if m[j] == target_cpu[j]:
                    m[j] = 1
                    m2[j] = alpha
                else:
                    m[j] = 0
                    m2[j] = 1
                if classifier_only: #Set expert always to false, if only the classifier should be trained
                    m[j] = 0
            m = torch.tensor(m, device=device)
            m2 = torch.tensor(m2, device=device)
            #m = m.to(device)
            #m2 = m2.to(device)
            #Optimization
            #m2 = torch.where(m == target, alpha, 1)
            #m = torch.where(m == target, 1, 0)
            
            collection_Ms.append((m, m2))

        # compute loss
        loss = loss_fn(output, target, collection_Ms, n_classes)
        epoch_train_loss.append(loss.item())

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        #losses.update(loss.data.item(), input.size(0))
        losses.update(loss.detach().item(), input.size(0))
        top1.update(prec1.detach().item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not iters < warmup_iters:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        iters += 1

        if i % 10 == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                ),
                flush=True,
            )

    return iters, np.average(epoch_train_loss)


def evaluate(model, expert_fns, loss_fn, n_classes, data_loader, config, print_m=True):
    """
    Computes metrics for deferal
    -----
    Arguments:
    net: model
    expert_fn: expert model
    n_classes: number of classes
    loader: data loader
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    #  === Individual Expert Accuracies === #
    expert_correct_dic = {k: 0 for k in range(len(expert_fns))}
    expert_total_dic = {k: 0 for k in range(len(expert_fns))}
    #  === Individual  Expert Accuracies === #
    alpha = config["alpha"]
    losses = []
    with torch.no_grad():
        for images, labels, hpred in data_loader:
            labels_cpu = labels
            images, labels = images.to(device), labels.to(device)
        #for data in data_loader:
        #    images, labels, hpred = data
        #    images, labels, hpred = images.to(device), labels.to(device), hpred
            outputs = model(images)
            if config["loss_type"] == "softmax":
                outputs = F.softmax(outputs, dim=1)
            elif config["loss_type"] == "ova":
                ouputs = F.sigmoid(outputs)

            _, predicted = torch.max(outputs, 1)
            #_, predicted = torch.max(outputs.data, 1)
            
            #batch_size = outputs.size()[0]  # batch_size
            batch_size = outputs.size(0)

            expert_predictions = []
            collection_Ms = []  # a collection of 3-tuple
            for i, fn in enumerate(expert_fns, 0):
                exp_prediction1 = fn(images, labels, hpred)
                #exp_prediction1 = fn(hpred)
                m = [0] * batch_size
                m2 = [0] * batch_size
                for j in range(0, batch_size):
                    #if exp_prediction1[j] == labels[j].item():
                    if exp_prediction1[j] == labels_cpu[j]:
                        m[j] = 1
                        m2[j] = alpha
                    else:
                        m[j] = 0
                        m2[j] = 1

                m = torch.tensor(m, device=device)
                m2 = torch.tensor(m2, device=device)

                #m = torch.tensor([1 if pred == label.item() else 0 for pred, label in zip(exp_prediction1, labels)])
                #m2 = torch.tensor([alpha if pred == label.item() else 1 for pred, label in zip(exp_prediction1, labels)])

                #collection_Ms.append((m.to(device), m2.to(device)))
                #expert_predictions.append(exp_prediction1)
                #End of optimization
                
                collection_Ms.append((m, m2))
                expert_predictions.append(exp_prediction1)

            loss = loss_fn(outputs, labels, collection_Ms, n_classes)
            losses.append(loss.detach().item())

            for i in range(batch_size):
                r = predicted[i].item() >= n_classes - len(expert_fns)
                prediction = predicted[i]
                if predicted[i] >= n_classes - len(expert_fns):
                    max_idx = 0
                    # get second max
                    for j in range(0, n_classes - len(expert_fns)):
                        if outputs.data[i][j] >= outputs.data[i][max_idx]:
                            max_idx = j
                    prediction = max_idx
                else:
                    prediction = predicted[i]
                alone_correct += (prediction == labels[i]).item()
                if r == 0:
                    total += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                elif r == 1:
                    deferred_exp = (predicted[i] - (n_classes - len(expert_fns))).item()
                    # cdeferred_exp = ((n_classes - 1) - predicted[i]).item()  # reverse order, as in loss function
                    exp_prediction = expert_predictions[deferred_exp][i]
                    #
                    # Deferral accuracy: No matter expert ===
                    exp += exp_prediction == labels[i].item()
                    exp_total += 1
                    # Individual Expert Accuracy ===
                    expert_correct_dic[deferred_exp] += (
                        exp_prediction == labels[i].item()
                    )
                    expert_total_dic[deferred_exp] += 1
                    #
                    correct_sys += exp_prediction == labels[i].item()
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)

    #  === Individual Expert Accuracies === #
    expert_accuracies = {
        "expert_{}".format(str(k)): 100
        * expert_correct_dic[k]
        / (expert_total_dic[k] + 0.0002)
        for k in range(len(expert_fns))
    }
    expert_coverage = {
        "expert_{}_coverage".format(str(k)): expert_total_dic[k] for k in range(len(expert_fns))
    }
    # Add expert accuracies dict
    to_print = {
        "coverage": cov,
        "system_accuracy": 100 * correct_sys / real_total,
        "expert_accuracy": 100 * exp / (exp_total + 0.0002),
        "classifier_accuracy": 100 * correct / (total + 0.0001),
        "alone_classifier": 100 * alone_correct / real_total,
        "validation_loss": np.average(losses),
        "n_experts": len(expert_fns),
        **expert_accuracies,
        **expert_coverage,
    }
    if print_m:
        print(to_print, flush=True)
    to_print["cov_classifier"] = total
    to_print["acc_experts"] = [100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002) for k in range(len(expert_fns))]
    to_print["cov_experts"] = [expert_total_dic[k] for k in range(len(expert_fns))]
    return to_print