import json
import os
import torch
import timm
import numpy as np
import math
import torch.nn as nn

from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.models.resnet import resnet50, resnet18
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights

import SSL.feature_extractor.data_loading as prep

from SSL.feature_extractor.wideresnet import WideResNet
from SSL.feature_extractor.utils import get_train_dir, printProgressBar
from SSL.feature_extractor.metrics import get_confusion_matrix


class EmbeddingModel:
    """Class representing the embedding model

    :param args: Training arguments for the embedding model
    :param wkdir: Working directory
    :param writer: Tensorboard writer

    :ivar global_step: Global setp
    :ivar args: Training arguments
    :ivar writer: Tensorboard writer
    :ivar device: Active device
    :ivar train_dir: Training directory of the embedding model
    :ivar model: Embedding model
    :ivar optimizer: Optimizer
    :ivar scheduler: Learning rate scheduler
    :ivar loss_function: Loss function
    :ivar train_data: Train dataset
    :ivar test_data: Test dataset
    :ivar val_data: Validation dataset
    :ivar train_loader: Train dataloader
    :ivar test_loader: Test dataloader
    :ivar val_loader: Validation dataloader
    """
    def __init__(self, args, wkdir, writer, dataloaders, param, neptune_param=None):
        self.global_step = 0
        self.args = args
        self.writer = writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dir = get_train_dir(wkdir, args, 'emb_net')
        self.model = self.get_model()
        if torch.cuda.device_count() > 1:
            print("Use ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args['lr'], weight_decay=5e-4, momentum=0.9,
                                         nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [60, 120, 160], gamma=0.2)
        self.loss_function = nn.CrossEntropyLoss()

        self.train_loader = dataloaders[0]
        self.test_loader = dataloaders[1]
        self.val_loader = dataloaders[2]

        self.train_data = self.train_loader.dataset

        
        self.save_model_args()

        self.param=param
        self.neptune_param = neptune_param

        print(str(math.ceil((len(self.train_data.targets) / self.args['batch']))))
        
        if neptune_param is not None:
            NEPTUNE = neptune_param["NEPTUNE"]

    def get_model(self):
        """Initialize model

        :return: model
        """
        # load model
        if self.args['model'] == 'wideresnet':
            model = WideResNet(28, 10, 0.2, self.args['num_classes'])
        elif self.args['model'] == 'resnet18':
            model = Resnet(self.args['num_classes'])
        elif self.args["model"] == "resnet50":
            model = Resnet(self.args['num_classes'], type="50")
        else:
            model = timm.create_model(self.args['model'], pretrained=True, num_classes=self.args['num_classes'])
        print('Loaded Model', self.args['model'])
        # load model to device
        #model = prep.to_device(model, self.device)
        model.to(self.device)

        return model

    def train_one_epoch(self, epoch):
        """Train one epoch

        :param epoch: Epoch
        :return: loss
        """
        self.model.train()
        for ii, (data, target, filename) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            target = target.long()
            pred = self.model(data)

            loss = self.loss_function(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += len(filename)
            printProgressBar(ii + 1, math.ceil((len(self.train_data.targets) / self.args['batch'])),
                             prefix='Train Epoch ' + str(epoch + 1) + ':',
                             suffix='Complete', length=40)
            if self.writer is not None:
                self.writer.add_scalar('Loss/total', loss, self.global_step)
                self.writer.add_scalar('LR/lr', self.optimizer.param_groups[0]["lr"], self.global_step)
        print("ii " + str(ii))
        return loss

    
    def get_validation_accuracy(self, epoch, return_acc=False, print_acc=True):
        """Get validation accuracy

        :param epoch: Epoch
        :param return_acc: Boolean flag for returning the accuracy
        :param print_acc: Boolean flag for printing the accuracy
        :return: (Accuracy) optional
        """
        predict = []
        targets = []
        self.model.eval()
        for i, (data, target, indices) in enumerate(self.val_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            # get model artificial_expert_labels
            with torch.no_grad():
                output = self.model(data)

            # get predicted classes from model output
            predicted_class = torch.argmax(output, dim=1).detach().cpu().numpy()

            for p in predicted_class:
                predict.append(p)
            for t in target:
                targets.append(t.item())

        # calculate accuracy score
        acc = accuracy_score(targets, predict)
        if print_acc:
            if epoch is None:
                print('Val-Accuracy:', acc)
            else:
                print('Epoch:', epoch + 1, '- Val-Accuracy:', acc)
        if self.writer is not None:
            self.writer.add_scalar('Acc/valid', acc, self.global_step)
        if return_acc: return acc

    
    def get_test_accuracy(self, return_acc=False, print_acc=True):
        """Get test accuracy

        :param return_acc: Boolean flag for returning the accuracy
        :param print_acc: Boolean flag for printing the accuracy
        :return: (Accuracy) optional
        """
        predict = []
        targets = []
        self.model.eval()
        for i, (data, target, indices) in enumerate(self.test_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            # get model artificial_expert_labels
            with torch.no_grad():
                output = self.model(data)

            # get predicted classes from model output
            m = nn.Softmax(dim=1)
            predicted_class = torch.argmax(output, dim=1).detach().cpu().numpy()
            for p in predicted_class:
                predict.append(p)
            for t in target:
                targets.append(t.item())

        # calculate accuracy score
        acc = accuracy_score(targets, predict)

        cm_true = get_confusion_matrix(targets, predict)
        cat_acc = cm_true.diagonal() / cm_true.sum(axis=1)
        if print_acc: print('Test-Accuracy:', acc, '\nTest-Acc-Class', cat_acc)
        if return_acc: return acc

    
    def predict_test_data(self):
        """Predict test data

        :return: artificial_expert_labels
        """
        predict = []
        self.model.eval()
        for i, (data, target, indices) in enumerate(self.test_loader):
            data.to(self.device)
            target.to(self.device)
            with torch.no_grad():
                output = self.model(data)

            m = nn.Softmax(dim=1)
            predicted_class = torch.argmax(m(output), dim=1).detach().cpu().numpy()
            for p in predicted_class:
                predict.append(int(p))

        return predict

    def load_from_checkpoint(self, mode='best'):
        """Load from checkpoint

        :param mode: Checkpoint to load (best or latest)
        :return: epoch
        """
        cp_dir = self.train_dir + '/checkpoints/checkpoint.' + mode
        try:
            checkpoint = torch.load(cp_dir)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_step = checkpoint['global_step']
            epoch = checkpoint['epoch']
            print('Found latest checkpoint at', cp_dir)
            print('Continuing in epoch', epoch + 1)
        except:
            epoch = 0
            print('No Checkpoint found')
            print('Starting new from epoch', epoch + 1)

        return epoch

    def save_to_checkpoint(self, epoch, loss, acc):
        with open(self.train_dir + '/logs/exp_log.json', 'r') as f:
            log = json.load(f)

        if acc >= np.max(log['valid_acc']):
            torch.save({'epoch': epoch,
                        'global_step': self.global_step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'valid_acc': acc}, self.train_dir + '/checkpoints/checkpoint.best')

        torch.save({'epoch': epoch,
                    'global_step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'valid_acc': acc}, self.train_dir + '/checkpoints/checkpoint.latest')

    def save_model_args(self):
        with open(self.train_dir + 'args/model_args.json', 'w') as f:
            json.dump(self.args, f)


class Resnet(torch.nn.Module):
    def __init__(self, num_classes, type="18"):
        super().__init__()
        self.num_classes = num_classes
        if type == "18":
            #self.resnet = resnet18(pretrained=True)
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            # del self.resnet.fc
        elif type == "50":
            self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        try:
            print('load Resnet-' + type + ' checkpoint')
            self.load_my_state_dict(
                torch.load(
                    os.getcwd()[:-len('Embedding-Semi-Supervised')] + "/nih_images/checkpoint.pretrain"),
                strict=False)
        except FileNotFoundError:
            print('load Resnet-' + type + ' pretrained on ImageNet')

        # for param in self.resnet.parameters():
        #    param.requires_grad = False

        # self.training = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # print(self.resnet)

    def load_my_state_dict(self, state_dict, strict=True):
        pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        self.resnet.load_state_dict(pretrained_dict, strict=strict)

    def forward(self, x, return_features=False):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        features = torch.flatten(x, 1)
        if return_features:
            return features
        else:
            out = self.resnet.fc(features)
            out = nn.Softmax(dim=1)(out)
            return out
