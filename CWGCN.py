import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv

def mydense_to_sparse(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)
    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]
    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])
    return torch.stack(index, dim=0), edge_attr

class DGN(torch.nn.Module):
    def __init__(self,tt0=None):
        super().__init__()
        self.atten=nn.Parameter(torch.ones(62,1))
        self.gcn1=DenseGCNConv(1,1)
    #    self.lined=nn.Linear(1,1)
    #    self.pool1=SAGPooling(1,ratio=0.5)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(62, 10),
            nn.Linear(10,3)
      )
    def forward(self, x,adj, batch0,t=0): 
        #x=(m,5*62) adj=(m,5*62*62) batch0=m
        m=adj.size(0)
        adj=adj.reshape(m,5,62,62)
#        adj=adj*0.5+0.5
        adj=adj[:,4,:,:].reshape(m,62,62)
        x=x.reshape(m,5,62)
        x=x[:,4,:].reshape(m,1,62)
        #x=torch.ones((m,1,62)).to(x)
        x=x.permute(0,2,1)
        x=self.gcn1(x,adj)
        x=x*self.atten
    #    xx=self.lined(x)
        if t==1:
           aa=torch.squeeze(self.atten)
           aa=torch.abs(aa)
           #print(aa)
           s1, t1 = aa.topk(k=62, dim=0, largest=True)
           s1 = s1.tolist()
           t1 = t1.tolist()
           s1 = [round(num, 4) for num in s1]
           return s1,t1
     
        xx =self.mlp(x)
        pred = F.softmax(xx,1)
        return  xx, pred