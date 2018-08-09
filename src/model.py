
import torch
from torch import nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Add(nn.Module):
    def forward(self, input1, input2):
        return torch.add(input1, input2)

class PRN(nn.Module):
    def __init__(self,node_count,coeff):
        super(PRN, self).__init__()
        self.flatten   = Flatten()
        self.height    = coeff*28
        self.width     = coeff*18
        self.dens1     = nn.Linear(self.height*self.width*17, node_count)
        self.bneck     = nn.Linear(node_count, node_count)
        self.dens2     = nn.Linear(node_count, self.height*self.width*17)
        self.drop      = nn.Dropout()
        self.add       = Add()
        self.softmax   = nn.Softmax(dim=1)

    def forward(self, x):
        res = self.flatten(x)
        out = self.drop(F.relu(self.dens1(res)))
        out = self.drop(F.relu(self.bneck(out)))
        out = F.relu(self.dens2(out))
        out = self.add(out,res)
        out = self.softmax(out)
        out = out.view(out.size()[0],self.height, self.width, 17)

        return out

