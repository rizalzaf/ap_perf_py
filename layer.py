from metric import Metric
import numpy as np
import torch 
import torch.nn as nn
from torch.autograd import Function

class MetricFunction(Function):
    @staticmethod
    def forward(ctx, input, target, metric):
        # the input is an n length vector
        if len(input.size()) != 1:
            raise ValueError("Input must be a vector")

        use_cuda = input.is_cuda

        # convert to numpy
        input_np = input.cpu().numpy()
        target_np = target.cpu().numpy()

        # obj and q
        obj_np, q_np = metric.objective(input_np, target_np)
        
        # obj - ground truth
        obj_np += np.dot(input_np, target_np)

        # back to torch
        q = torch.tensor(q_np).float()
        obj = torch.tensor([obj_np]).float()

        # save for computing gradient
        if use_cuda: 
            q = q.cuda()
            obj = obj.cuda()

        # save for computing gradient
        ctx.save_for_backward(q, target)

        # negative of obj, since the original formulation is max_\theta
        # whereas the optimizer only support minimization
        return -obj

    @staticmethod
    def backward(ctx, grad_output):
        # load saved tensor
        q, target = ctx.saved_tensors

        # grad = y - q 
        grad = target - q 

        return -grad, None, None


class MetricLayer(nn.Module):
    def __init__(self, metric):
        super(MetricLayer, self).__init__()
        self.metric = metric
        
    def forward(self, input, target):
        # Apply function
        return MetricFunction.apply(input, target, self.metric)

