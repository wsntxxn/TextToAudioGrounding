
import math
import ipdb
import torch.nn as nn
import torch.nn.functional as F
import torch as th

class NetVF(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super().__init__()
        self.feature_size =feature_size
        self.cluster_size = cluster_size

        init_sc = (1 / math.sqrt(feature_size))
        
        self.cluster_weights = nn.Parameter(init_sc * th.randn(feature_size, cluster_size))
        self.cluster_weights2 =nn.Parameter(init_sc *th.randn(1, feature_size, cluster_size))
        covar_weights = nn.Parameter(init_sc * th.randn(feature_size, cluster_size).normal_(mean=1))
        self.covar_weights = th.square(covar_weights)
        self.batch_norm = nn.BatchNorm1d(cluster_size) if add_batch_norm else None
        self.out_dim = self.cluster_size * feature_size
        self.hidden1_weights=nn.Parameter(init_sc *(
                    th.randn(2 * self.cluster_size * self.feature_size, 
                             self.out_dim)))


    def forward(self, x, mask=None):
        self.sanity_checks(x)
        max_sample = x.size()[1]
        x =x.view(-1, self.feature_size)
        if x.device != self.cluster_weights.device:
            msg = f"x.device {x.device} != cluster.device {self.cluster_weights.device}"
            raise ValueError(msg)

        assignment = th.matmul(x, self.cluster_weights)

        if self.batch_norm:
            assignment = self.batch_norm(assignment)

        assignment = F.softmax(assignment, dim=-1)
        assignment = assignment.reshape(-1, max_sample,self.cluster_size)
        a_sum = th.sum(assignment, dim=1, keepdim=True)
        a = a_sum * self.cluster_weights2

        assignment = assignment.transpose(1,2)

        x = x.view(-1, max_sample, self.feature_size)
        fv1 = th.matmul(assignment, x)
        fv1 = fv1.transpose(1,2)

        #computing second order fv
        a2  = a_sum * th.square(self.cluster_weights2)
        b2 = fv1 * self.cluster_weights2
        fv2 = th.matmul(assignment, th.square(x))

        fv2 = fv2.transpose(1,2)
        fv2 = a2 + fv2 + th.mul(b2,-2)
        
        device = x.device
        eps = th.tensor([1e-6]).to(device)
        self.covar_weights = self.covar_weights.to(device) + eps 
        fv2 = th.div(fv2, th.square(self.covar_weights))
        dv2 = fv2 - a_sum
        
        fv2 = F.normalize(fv2)
        fv2 = fv2.reshape(-1, self.cluster_size * self.feature_size)

        fv2 = F.normalize(fv2)
        
        fv1 = fv1 - a
        fv1 = th.div(fv1, self.covar_weights)
        
        fv1 = F.normalize(fv1)
        fv1 = fv1.reshape(-1, self.cluster_size * self.feature_size)

        fv1 = F.normalize(fv1)

        fv = th.cat((fv1, fv2),dim=1)

        fv = th.matmul(fv, self.hidden1_weights)
        
        return fv

    def sanity_checks(self, x):
        """Catch any nans in the inputs/clusters"""
        if th.isnan(th.sum(x)):
            print("nan inputs")
            ipdb.set_trace()
        if th.isnan(self.cluster_weights[0][0]):
            print("nan cluster_weights")
            ipdb.set_trace()

