import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import StandardScaler

from metric import Metric
from layer import MetricLayer

def load_data(ds, standardize=True):
    df = pd.read_csv("data-cv/" + ds + ".csv", sep=',', header=None)
    X_all = df.values[:,:-1].astype(np.float32)
    y_all = df.values[:,-1]
    y_all = (y_all >= 7).astype(np.float32)  # to binary

    n_all = len(y_all)
    np.random.seed(1)
    id_perm = np.random.permutation(n_all)
    n_tr = n_all // 2
    
    # split
    X_tr = X_all[id_perm[:n_tr], :]
    X_ts = X_all[id_perm[n_tr:], :]
    y_tr = y_all[id_perm[:n_tr]]
    y_ts = y_all[id_perm[n_tr:]]

    # normalize
    if standardize:
        scaler = StandardScaler()
        scaler.fit(X_tr)

        X_tr = scaler.transform(X_tr)
        X_ts = scaler.transform(X_ts)

    return X_tr, X_ts, y_tr, y_ts


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.X[idx, :], self.y[idx]]


class Net(nn.Module):
    def __init__(self, nvar):
        super().__init__()
        self.fc1 = nn.Linear(nvar, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()


# performance metric
# F-Beta
fbeta_str = """ 
@metric FBeta beta
function define(::Type{FBeta}, C::ConfusionMatrix, beta)
    return ((1 + beta^2) * C.tp) / (beta^2 * C.ap + C.pp)  
end   
"""

f2 = Metric(fbeta_str)
f2.initialize(2.0)
f2.special_case_positive()


## prec given rec
precrec_str = """
@metric PrecRec th
function define(::Type{PrecRec}, C::ConfusionMatrix, th)
    return C.tp / C.pp
end   
function constraint(::Type{PrecRec}, C::ConfusionMatrix, th)
    return C.tp / C.ap >= th
end 
"""

precrec80 = Metric(precrec_str)
precrec80.initialize(0.8)
precrec80.special_case_positive()
precrec80.cs_special_case_positive(True)


# perf 
pm = f2
# pm = precrec80


ds = "whitewine"
X_tr, X_ts, y_tr, y_ts = load_data(ds, standardize=True)

trainset = TabularDataset(X_tr, y_tr)
testset = TabularDataset(X_ts, y_ts)

trainloader = DataLoader(trainset, batch_size=25, shuffle=True)

method = "ap-perf"
# method = "bce-loss"

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

nvar = X_tr.shape[1]
model = Net(nvar).to(device)

if method == "ap-perf":
    criterion = MetricLayer(f2).to(device)
else:
    criterion = nn.BCEWithLogitsLoss().to(device)


optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(100):

    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()


    # evaluate after each epoch
    model.eval()

    # train
    train_data = torch.tensor(X_tr).to(device)
    tr_output = model(train_data)
    tr_pred = (tr_output >= 0.0).float()
    tr_pred_np = tr_pred.cpu().numpy()

    train_acc = np.sum(y_tr == tr_pred_np) / len(y_tr)
    train_metric = pm.compute_metric(tr_pred_np, y_tr)
    
    # test
    test_data = torch.tensor(X_ts).to(device)
    ts_output = model(test_data)
    ts_pred = (ts_output >= 0.0).float()
    ts_pred_np = ts_pred.cpu().numpy()

    test_acc = np.sum(y_ts == ts_pred_np) / len(y_ts)
    test_metric = pm.compute_metric(ts_pred_np, y_ts)

    model.train()

    print('#{} | Acc tr: {:.5f} | Acc ts: {:.5f} | Metric tr: {:.5f} | Metric ts: {:.5f}'.format(
        epoch, train_acc, test_acc, train_metric, test_metric))






