import sys, os

from absl import flags
from absl import app

from SSL.feature_extractor.utils import save_to_logs, get_train_dir
from SSL.feature_extractor.emb_model_lib import EmbeddingModel

import Dataset.Dataset as ds
import torch

import random

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from SSL.LinearModel import LinearNN
import SSL.datasets.nih as nih
from SSL.utils import accuracy, setup_default_logging, AverageMeter, WarmupCosineLrScheduler, AverageMeterOptimized
from SSL.utils import load_from_checkpoint
#from SSL.Expert import CIFAR100Expert, NIHExpert
from SSL.feature_extractor.embedding_model import EmbeddingModel as EmbeddingModelL

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_embedded_model(dataloaders, param, neptune_param, fold, seed):
    args = param["EMBEDDED"]["ARGS"]
    path = param["PATH"]
    neptune_param = neptune_param

    #wkdir = os.getcwd() + "/SSL_Working"
    wkdir = param["Parent_PATH"] + "/SSL_Working/"# + param["DATASET"]
    
    sys.path.append(wkdir)

    SAVE = True

    # get training directory
    train_dir = get_train_dir(wkdir, args, 'emb_net', param, seed, fold)

    print("Train dir: " + train_dir)

    NEPTUNE = neptune_param["NEPTUNE"]

    writer = None

    if SAVE:
        # initialize summary writer for tensorboard
        writer = SummaryWriter(train_dir + 'logs/')

    # initialize base model
    emb_model = EmbeddingModel(args, wkdir, writer, dataloaders, param, neptune_param, seed, fold)
    # try to load previous training runs
    start_epoch = emb_model.load_from_checkpoint(mode='latest')

    
    
    # train model
    for epoch in range(start_epoch, param["EMBEDDED"]["EPOCHS"]):
        # train one epoch
        loss = emb_model.train_one_epoch(epoch)
        # get validation accuracy
        valid_acc = emb_model.get_test_accuracy(return_acc=True)
        print(f'loss: {loss}')

        if NEPTUNE:
            run = param["NEPTUNE"]["RUN"]
            run[f'Embedded/Seed_{seed}/Fold_{fold}/Val/loss'].append(loss)
            run[f'Embedded/Seed_{seed}/Fold_{fold}/Val/accuracy'].append(valid_acc)
        # save logs to json
        if SAVE:
            save_to_logs(train_dir, valid_acc, loss.item())
            # save model to checkpoint
            emb_model.save_to_checkpoint(epoch, loss, valid_acc)
    # get test accuracy
    acc = emb_model.get_test_accuracy()

    if NEPTUNE:
        run = param["NEPTUNE"]["RUN"]
        run[f"Embedded/Seed{seed}/Fold_{fold}/Test/accuracy"].append(acc)

    return emb_model


def set_model(args):
    """Initialize models

    Lineare Modelle, welche später die extrahierten Features übergeben bekommen

    :param args: training arguments
    :return: tuple
        - model: Initialized model
        - criteria_x: Supervised loss function
        - ema_model: Initialized ema model
    """
    if args["dataset"].lower() == 'cifar100':
        feature_dim = 1280
    elif args["dataset"].lower() == 'nih':
        if args["type"] == "18":
            feature_dim = 512
        elif args["type"] == "50":
            feature_dim = 2048
    else:
        print(f'Dataset {args["dataset"]} not defined')
        sys.exit()
    model = LinearNN(num_classes=args["n_classes"], feature_dim=feature_dim, proj=True)

    model.train()
    model.cuda() 

    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    if args["eval_ema"]:
        ema_model = LinearNN(num_classes=args["n_classes"], feature_dim=feature_dim, proj=True)
        for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        ema_model.cuda()  
        ema_model.eval()

        if torch.cuda.device_count() > 1:
            print("Use ", torch.cuda.device_count(), "GPUs!")
            ema_model = nn.DataParallel(ema_model)
    else:
        ema_model = None
        
    criteria_x = nn.CrossEntropyLoss().cuda()
    return model, criteria_x, ema_model


def train_one_epoch(epoch,
                    model,
                    ema_model,
                    emb_model,
                    prob_list,
                    criteria_x,
                    optim,
                    lr_schdlr,
                    dltrain_x,
                    dltrain_u,
                    args,
                    n_iters,
                    logger,
                    queue_feats,
                    queue_probs,
                    queue_ptr,
                    ):
    """Train one epoch on the train set

    :param epoch: Current epoch
    :param model: Model
    :param ema_model: EMA-Model
    :param emb_model: Embedding model
    :param prob_list: List of probabilities
    :param criteria_x: Supervised loss function
    :param optim: Optimizer
    :param lr_schdlr: Learning rate scheduler
    :param dltrain_x: Data loader for the labeled training instances
    :param dltrain_u: Data loader for the unlabeled training instances
    :param args: Training arguments
    :param n_iters: Number of iterations per epoch
    :param logger: Logger
    :param queue_feats: Memory bank feature vectors
    :param queue_probs: Memory bank probabilities
    :param queue_ptr: Memory bank ptr
    :return: tuple
        - Average supervised loss
        - Average unsupervised loss
        - Average contrastive loss
        - Average mask
        - Average number of edges in the pseudo label graph
        - Percentage of correct pseudo labels
        - Memory bank feature vectors
        - Memory bank probabilities
        - Memory bank ptr
        - List of probabilities
    """

    model.train()
    #Old Code
    """
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    loss_contrast_meter = AverageMeter()
    # the number of correct pseudo-labels
    n_correct_u_lbs_meter = AverageMeter()
    # the number of confident unlabeled data
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()
    # the number of edges in the pseudo-label graph
    pos_meter = AverageMeter()"""

    #Optimized
    loss_x_meter = AverageMeterOptimized()
    loss_u_meter = AverageMeterOptimized()
    loss_contrast_meter = AverageMeterOptimized()
    # the number of correct pseudo-labels
    n_correct_u_lbs_meter = AverageMeterOptimized()
    # the number of confident unlabeled data
    n_strong_aug_meter = AverageMeterOptimized()
    mask_meter = AverageMeterOptimized()
    # the number of edges in the pseudo-label graph
    pos_meter = AverageMeterOptimized()

    
    epoch_start = time.time()  # start time
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    for it in range(n_iters):
        ims_x_weak, lbs_x, im_id, gt_x = next(dl_x) #transformed image, expert label, filename, gt_labels
        (ims_u_weak, ims_u_strong0, ims_u_strong1), lbs_u_real, im_id, gt_u = next(dl_u) #transformed images, expert label, filename, gt_labels

        lbs_x = lbs_x.type(torch.LongTensor).cuda()
        gt_x = gt_x.type(torch.LongTensor).cuda()

        lbs_u_real = lbs_u_real.cuda()
        gt_u = gt_u.cuda()

        if args["expert_predict"] == "right":
            # Compare human expert labels with ground truth labels
            correct_predictions = torch.eq(lbs_x, gt_x).type(torch.LongTensor).cuda()
            lbs_x = correct_predictions

            correct_predictions = torch.eq(lbs_u_real, gt_u).type(torch.LongTensor).cuda()
            lbs_u_real = correct_predictions

        # --------------------------------------
        bt = ims_x_weak.size(0)
        btu = ims_u_weak.size(0)

        imgs = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong0, ims_u_strong1], dim=0).cuda()
        embedding = emb_model.get_embedding(batch=imgs)

        logits, features = model(embedding)

        """logits_x = logits[:bt]
        logits_u_w, logits_u_s0, logits_u_s1 = torch.split(logits[bt:], btu)
        
        feats_x = features[:bt]
        feats_u_w, feats_u_s0, feats_u_s1 = torch.split(features[bt:], btu)"""

        logits_x, logits_u_w, logits_u_s0, logits_u_s1 = torch.split(logits, [bt, btu, btu, btu])
        feats_x, feats_u_w, feats_u_s0, feats_u_s1 = torch.split(features, [bt, btu, btu, btu])

        
        loss_x = criteria_x(logits_x, lbs_x)

        with torch.no_grad():
            logits_u_w = logits_u_w.detach()
            feats_x = feats_x.detach()
            feats_u_w = feats_u_w.detach()
            
            #probs = torch.softmax(logits_u_w, dim=1)  
            probs = F.softmax(logits_u_w, dim=1) 
            # DA
            prob_list.append(probs.mean(0))
            if len(prob_list)>32:
                prob_list.pop(0)
            prob_avg = torch.stack(prob_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)   

            probs_orig = probs.clone()
            
            if epoch>0 or it>args["queue_batch"]: # memory-smoothing 
                A = torch.exp(torch.mm(feats_u_w, queue_feats.t())/args["temperature"])       
                A = A/A.sum(1,keepdim=True)                    
                probs = args["alpha"]*probs + (1-args["alpha"])*torch.mm(A, queue_probs)               
            
            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(args["thr"]).float() 
                   
            feats_w = torch.cat([feats_u_w,feats_x],dim=0)   
            #Old
            #onehot = torch.zeros(bt,args["n_classes"]).cuda().scatter(1,lbs_x.view(-1,1),1)
            #Optimized
            onehot = torch.zeros(bt, args["n_classes"], device="cuda")
            onehot.scatter_(1, lbs_x.view(-1, 1), 1)


            
            probs_w = torch.cat([probs_orig,onehot],dim=0)
            
            # update memory bank
            n = bt+btu   
            queue_feats[queue_ptr:queue_ptr + n,:] = feats_w
            queue_probs[queue_ptr:queue_ptr + n,:] = probs_w      
            queue_ptr = (queue_ptr+n)%args["queue_size"]

            
        # embedding similarity
        sim = torch.exp(torch.mm(feats_u_s0, feats_u_s1.t())/args["temperature"]) 
        sim_probs = sim / sim.sum(1, keepdim=True)
        
        # pseudo-label graph with self-loop
        Q = torch.mm(probs, probs.t())       
        Q.fill_diagonal_(1)    
        pos_mask = (Q>=args["contrast_th"]).float()
            
        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)
        
        # contrastive loss
        loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
        loss_contrast = loss_contrast.mean()  
        
        # unsupervised classification loss
        loss_u = - torch.sum((F.log_softmax(logits_u_s0,dim=1) * probs),dim=1) * mask                
        loss_u = loss_u.mean()
        
        loss = loss_x + args["lam_u"] * loss_u + args["lam_c"] * loss_contrast
        
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        lr_schdlr.step()

        if args["eval_ema"]:
            with torch.no_grad():
                ema_model_update(model, ema_model, args["ema_m"])

        #Old Code
        
        """loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())
        loss_contrast_meter.update(loss_contrast.item())
        mask_meter.update(mask.mean().item())       
        pos_meter.update(pos_mask.sum(1).float().mean().item())
        
        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())"""

        loss_x_meter.addTensor(loss_x)
        loss_u_meter.addTensor(loss_u)
        loss_contrast_meter.addTensor(loss_contrast)
        mask_meter.addTensor(mask.mean())       
        pos_meter.addTensor(pos_mask.sum(1).float().mean())
        
        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.addTensor(corr_u_lb.sum())
        n_strong_aug_meter.addTensor(mask.sum())

        #if (it + 1) % 128 == 0:
        if (it + 1) == n_iters:

            #Needed for optimized Meters
            loss_x_meter.getAverage()
            loss_u_meter.getAverage()
            loss_contrast_meter.getAverage()
            mask_meter.getAverage()      
            pos_meter.getAverage()
    
            n_correct_u_lbs_meter.getAverage()
            n_strong_aug_meter.getAverage()
            
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)

            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}. loss_u: {:.3f}. loss_x: {:.3f}. loss_c: {:.3f}. "
                        "n_correct_u: {:.2f}/{:.2f}. Mask:{:.3f}. num_pos: {:.1f}. LR: {:.3f}. Time: {:.2f}".format(
                args["dataset"], args["n_labeled"], args["seed"], args["exp_dir"], epoch, it + 1, loss_u_meter.avg, loss_x_meter.avg, loss_contrast_meter.avg, n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, mask_meter.avg, pos_meter.avg, lr_log, t))
            epoch_start = time.time()

    #Needed for optimized Meters
    loss_x_meter.getAverage()
    loss_u_meter.getAverage()
    loss_contrast_meter.getAverage()
    mask_meter.getAverage()      
    pos_meter.getAverage()
    
    n_correct_u_lbs_meter.getAverage()
    n_strong_aug_meter.getAverage()
            
    return loss_x_meter.avg, loss_u_meter.avg, loss_contrast_meter.avg, mask_meter.avg, pos_meter.avg, n_correct_u_lbs_meter.avg/max(n_strong_aug_meter.avg, 0.000001), queue_feats, queue_probs, queue_ptr, prob_list


def evaluate(model, ema_model, emb_model, dataloader, param):
    """Evaluate model on train or validation set

    :param model: Model
    :param ema_model: EMA-Model
    :param emb_model: Embedding model
    :param dataloader: Data loader for the evaluation set
    :return: tuple
        - Accuracy of the model
        - Accuracy of the ema_model
    """
    
    model.eval()
    preds = []
    targets = []
    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()

    with torch.no_grad():
        for ims, lbs, im_id, gt in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            gt = gt.cuda()

            if param["EXPERT_PREDICT"] == "right":
                correct_predictions = torch.eq(lbs, gt).type(torch.LongTensor).cuda()
                lbs = correct_predictions

            embedding = emb_model.get_embedding(batch=ims)
            logits, _ = model(embedding)
            scores = F.softmax(logits, dim=1)
            preds += torch.argmax(scores, dim=1).detach().cpu().tolist()
            targets += lbs.detach().cpu().tolist()
            top1 = accuracy(scores, lbs, (1, ))
            top1_meter.update(top1.item())
            
            if ema_model is not None:
                embedding = emb_model.get_embedding(batch=ims)
                logits, _ = ema_model(embedding)
                scores = F.softmax(logits, dim=1)
                top1 = accuracy(scores, lbs, (1, ))
                ema_top1_meter.update(top1.item())
    return top1_meter.avg, ema_top1_meter.avg


@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """Momentum update of evaluation model (exponential moving average)

    :param model: Model
    :param ema_model: EMA-Model
    :param ema_m: Ema parameter
    :return:
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1-ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)

class exper:
    def __init__(self, id):
        self.labeler_id = id


def getExpertModelSSL(labelerId, sslDataset, seed, fold_idx, n_labeled, embedded_model=None, param=None, neptune_param=None, added_epochs=0):
    args = {
        "dataset": "NIH", #
        "wresnet_k": 2, #width factor of wide resnet
        "wresnet_n": 28, #depth of wide resnet
        "n_classes": 2, #number of classes in dataset
        "mu": 7, #factor of train batch size of unlabeled samples
        #"n_imgs_per_epoch": 32768, #number of training images for each epoch
        #"n_imgs_per_epoch": 4381,
        "eval_ema": True, #whether to use ema model for evaluation
        "ema_m": 0.999, #
        "lam_u": 1., #coefficient of unlabeled loss
        "lr": 0.03, #learning rate for training
        "weight_decay": 5e-4, #weight decay
        "momentum": 0.9, #momentum for optimizer
        "temperature": 0.2, #softmax temperature
        "low_dim": 64, #
        "lam_c": 1, #coefficient of contrastive loss
        "contrast_th": 0.8, #pseudo label graph threshold
        "thr": 0.95, #pseudo label threshold
        "alpha": 0.9, #
        "queue_batch": 5, #number of batches stored in memory bank
        "exp_dir": "EmbeddingCM_bin", #experiment id
        #"ex_strength": 4323195249, #Strength of the expert 
        #"ex_strength": 4295232296
    }

    args["labelerId"] = labelerId
    args["ex_strength"] = labelerId
    args["n_labeled"] = n_labeled
    args["seed"] = seed
    args["n_epoches"] = param["SSL"]["N_EPOCHS"]
    if added_epochs != 0: #Maybe start "new" when ssl for active learning beacuse additional training doesn't work (nan values)
        args["n_epoches"] = added_epochs
        print(f"Epochs added: {added_epochs}")
        
    args["batchsize"] = param["SSL"]["BATCHSIZE"]
    args["n_imgs_per_epoch"] = param["SSL"]["N_IMGS_PER_EPOCH"]
    if param["EMBEDDED"]["ARGS"]["model"] == "resnet18":
        args["type"] = "18"
    elif param["EMBEDDED"]["ARGS"]["model"] == "resnet50":
        args["type"] = "50"
    path = param["PATH"]
    args["n_classes"] = param["n_classes"]

    args["expert_predict"] = param["EXPERT_PREDICT"]

    #Setzt Logger fest
    out_path = f"{param['Parent_PATH']}/SSL_Working/{param['DATASET']}/SSL/"
        
    logger, output_dir = setup_default_logging(out_path, args)
    logger.info(dict(args))
    
    tb_logger = SummaryWriter(output_dir)

    set_seed(seed)

    #Calculates number of iterations
    n_iters_per_epoch = args["n_imgs_per_epoch"] // args["batchsize"]  # 1024
    n_iters_all = n_iters_per_epoch * args["n_epoches"]  # 1024 * 200

    emb_model = EmbeddingModelL(out_path[:-5], args["dataset"], type=args["type"], param=param, seed=seed, fold=fold_idx)
    

    if args["expert_predict"] == "right":
        args["n_classes"] = param["NUM_CLASSES"]

    #Erstellt das Modell
    model, criteria_x, ema_model = set_model(args)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))
    #Lädt das trainierte eingebettete Modell
    #emb_model = EmbeddingModelL(os.getcwd() + "/SSL_Working", args["dataset"], type=args["type"])

    if 'nih' in param["DATASET"].lower(): #Erstellt den Experten mit seiner ID
        exp = exper(int(args["labelerId"]))
    else:
        exp = exper(args["labelerId"])
        
    dltrain_x, dltrain_u = sslDataset.get_train_loader_interface( 
            exp, args["batchsize"], args["mu"], n_iters_per_epoch, L=args["n_labeled"], method='comatch', pin_memory=False)
    dlval = sslDataset.get_val_loader_interface(exp, batch_size=64, num_workers=param["num_worker"], fold_idx=fold_idx)
    dtest = sslDataset.get_test_loader_interface(exp, batch_size=64, num_workers=param["num_worker"], fold_idx=fold_idx)

    

    wd_params, non_wd_params = [], []
    for name, params in model.named_parameters():
        if 'bn' in name:
            non_wd_params.append(params)  
        else:
            wd_params.append(params)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=args["lr"], weight_decay=args["weight_decay"],
                            momentum=args["momentum"], nesterov=True)

    lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)
    
    model, ema_model, optim, lr_schdlr, start_epoch, metrics, prob_list, queue = \
        load_from_checkpoint(output_dir, model, ema_model, optim, lr_schdlr)

    if added_epochs > 0: #Reset for added epochs
        optim = torch.optim.SGD(param_list, lr=args["lr"], weight_decay=args["weight_decay"],
                            momentum=args["momentum"], nesterov=True)

        lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)

        queue = None

    # memory bank
    args["queue_size"] = args["queue_batch"]*(args["mu"]+1)*args["batchsize"]
    if queue is not None:
        queue_feats = queue['queue_feats']
        queue_probs = queue['queue_probs']
        queue_ptr = queue['queue_ptr']
    else:
        queue_feats = torch.zeros(args["queue_size"], args["low_dim"]).cuda()
        queue_probs = torch.zeros(args["queue_size"], args["n_classes"]).cuda()
        queue_ptr = 0

    train_args = dict(
        model=model,
        ema_model=ema_model,
        emb_model=emb_model,
        prob_list=prob_list,
        criteria_x=criteria_x,
        optim=optim,
        lr_schdlr=lr_schdlr,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        args=args,
        n_iters=n_iters_per_epoch,
        logger=logger
    )
    
    best_acc = -1
    best_epoch = 0

    if metrics is not None:
        best_acc = metrics['best_acc']
        best_epoch = metrics['best_epoch']
    logger.info('-----------start training--------------')

    if added_epochs > 0:
        start_epoch = 0
    
    for epoch in range(start_epoch, args["n_epoches"]):
        
        loss_x, loss_u, loss_c, mask_mean, num_pos, guess_label_acc, queue_feats, queue_probs, queue_ptr, prob_list = \
        train_one_epoch(epoch, **train_args, queue_feats=queue_feats,queue_probs=queue_probs,queue_ptr=queue_ptr)

        top1, ema_top1 = evaluate(model, ema_model, emb_model, dlval, param)


        tb_logger.add_scalar('loss_x', loss_x, epoch)
        tb_logger.add_scalar('loss_u', loss_u, epoch)
        tb_logger.add_scalar('loss_c', loss_c, epoch)
        tb_logger.add_scalar('guess_label_acc', guess_label_acc, epoch)
        tb_logger.add_scalar('test_acc', top1, epoch)
        tb_logger.add_scalar('test_ema_acc', ema_top1, epoch)
        tb_logger.add_scalar('mask', mask_mean, epoch)
        tb_logger.add_scalar('num_pos', num_pos, epoch)

        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch

        logger.info("Epoch {}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{}".
                    format(epoch, top1, ema_top1, best_acc, best_epoch))

        if param["NEPTUNE"]["NEPTUNE"]:
            run = param["NEPTUNE"]["RUN"]
            run[f"SSL/Seed_{seed}/Fold_{fold_idx}/Expert_{labelerId}/Train/" + "Accuracy"].append(top1)
            run[f"SSL/Seed_{seed}/Fold_{fold_idx}/Expert_{labelerId}/Train/" + "Ema_Accuracy"].append(ema_top1)
        
        save_obj = {
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'optimizer': optim.state_dict(),
            'lr_scheduler': lr_schdlr.state_dict(),
            'prob_list': prob_list,
            'queue': {'queue_feats':queue_feats, 'queue_probs':queue_probs, 'queue_ptr':queue_ptr},
            'metrics': {'best_acc': best_acc, 'best_epoch': best_epoch},
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(output_dir, 'ckp.latest'))
    _, _ = evaluate(model, ema_model, emb_model, dlval, param)
    _, _ = evaluate(model, ema_model, emb_model, dtest, param)

    return emb_model, model