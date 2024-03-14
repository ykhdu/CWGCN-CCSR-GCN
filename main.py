import time
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score,precision_recall_fscore_support
from CCSR_GCN import TAF1_M
from datapipe import get_dataset

##########################################################
"""
Settings for training 
"""
subjects = 15
epochs = 200
classes = 3  # Num. of classes
device = torch.device('cuda', 1)
# device=torch.device('cpu')
version = 10
print('***' * 20)

##########################################################

def train(model, train_loader, crit, optimizer):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # Multiple Classes classification Loss functionp
        label = torch.argmax(data.y.view(-1, classes), axis=1)
        label = label.to(device)  # , dtype=torch.long) #, dtype=torch.int64)
        output, _ = model(data.x, data.adj, data.batch)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def evaluate(model, loader, save_result=False):
    model.eval()

    predictions = []
    labels = []
    adjs = []

    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1, classes)
            data = data.to(device)

            _, pred = model(data.x, data.adj, data.batch)
            pred = pred.detach().cpu().numpy()
            pred = np.squeeze(pred)
            predictions.append(pred)
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    f1 = f1_score(np.argmax(labels, axis=-1), np.argmax(predictions, axis=-1), average='macro')
    acc = accuracy_score(np.argmax(labels, axis=-1), np.argmax(predictions, axis=-1))
    precision, recall, _, support = precision_recall_fscore_support(np.argmax(labels, axis=-1), np.argmax(predictions, axis=-1), zero_division=0)
    print(support)
    return precision, recall, acc, f1


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic=True
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# Network00=SymSimGCNNet
# Network00=DGCNN
result_data = []
all_last_acc = []
all_last_AUC = []
flag = 0

sum1 = 0.0
sum2 = 0.0
sum3 = 0.0
a1 = list(range(0, 2010))
a2 = list(range(3394, 5404))
a3 = list(range(6788, 8798))
b1 = list(range(2010, 3394))
b2 = list(range(5404, 6788))
b3 = list(range(8798, 10182))
aaa = [[a1, b1], [a2, b2], [a3, b3]]
myseed = [1708276400
,1708337741
,1708230390]
long = 1
Network00 = TAF1_M
list_all = [] #
for cv_n in range(subjects):
    sumlist = []
    seed_list = []
    sub_list_p = []
    sub_list_r = []
    for i in range(long):
        _, test_dataset0 = get_dataset(15, cv_n)
        #   0:2010  2010:3394      3394:5404 5404:6788          6788:8798 8798:10182
        for iii in range(3):
            setup_seed(myseed[cv_n * 3 + iii])
            sub_list = []
            train_dataset = test_dataset0[aaa[iii][0]]
            test_dataset = test_dataset0[aaa[iii][1]]
            #   train_list=random.sample(range(0,47516),5000)
            #   train_dataset=train_dataset0[train_list]
            train_loader = DataLoader(train_dataset, batch_size=100, drop_last=False, shuffle=True,generator=torch.Generator().manual_seed(1))
            test_loader = DataLoader(test_dataset, batch_size=1384, drop_last=False, shuffle=False)
            modelB = Network00(cv_n, iii).to(device)
            #   ,weight_decay=0.001
            optimizer = torch.optim.Adam(modelB.parameters(), lr=0.007, weight_decay=0.001)
            crit = torch.nn.CrossEntropyLoss()  #
            themax = 0.0
            the_max_p = 0.0
            the_max_r = 0.0
            for epoch in range(epochs):
                t0 = time.time()
                loss = train(modelB, train_loader, crit, optimizer)
                _,_, train_acc, train_f1 = evaluate(modelB, train_loader)
                val_p,val_r, val_acc, val_f1 = evaluate(modelB, test_loader)
                t1 = time.time()
                # print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f}, VAUC:{:.2f}, Vacc:{:.4f}, Time: {:.2f}'. format(cv_n, epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, (t1-t0)))
                if val_acc > themax:
                    themax = val_acc
                    the_max_p = val_p[0]
                    the_max_r = val_r[0]
                    #   print('Results::::::::::::')
            print(myseed[cv_n * 3 + iii])
            print(cv_n + 1, '---', iii, '---the max acc is', themax)
            # loader2 = DataLoader(train_dataset, batch_size=2010, drop_last=False, shuffle=False)
            # evaluate2(modelB, loader2)
            #  val_AUC, val_acc, val_f1 = evaluate(modelB, test_loader)
            #   print('VAUC:{:.2f}, Vacc:{:.4f},Val_f1:{:.2f}'.format( val_AUC, val_acc,val_f1))
            sub_list.append(float(format(themax * 100, '.2f')))
            list_all.append(float(format(themax * 100, '.2f'))) #
            sub_list_p.append(float(format(the_max_p * 100, '.2f')))
            sub_list_r.append(float(format(the_max_r * 100, '.2f')))
            sumlist.append(sub_list)
            seed_list.append(myseed)
    print(sumlist)
    print(sub_list_p)
    print(sub_list_r)
print(list_all) #
