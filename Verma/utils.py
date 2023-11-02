import torch
import numpy as numpy
import torch.nn as nn


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

#Optimized AverageMeter which stores tensors unit final need
class AverageMeterOptimized(object):
    """
    Computes and stores the average and current value

    :ivar val: Value
    :ivar avg: Average
    :ivar sum: Sum
    :ivar count: Count
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.tensors = []

    """def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / (self.count + 1e-20)
        self.avg = self.sum / self.count"""

    def addTensor(self, tensor):
        self.tensors.append(tensor)
        self.count += 1

    def getAverage(self):
        if len(self.tensors) >= 1:
            for tensor in self.tensors:
                self.sum += tensor.item()
            self.val = self.tensors[-1].item()
            self.avg = self.sum / self.count
            self.tensors.clear()