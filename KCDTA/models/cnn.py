import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class cnn(nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, output_dim=128, dropout=0.2):
        super(cnn, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.n_output = n_output

        # # combine
        self.conv7 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 2))
        self.conv8 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(2, 2))
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2))
        self.ppool2d = nn.MaxPool2d(kernel_size=(2, 2))

        self.ppfc1 = nn.Linear(2592, 1024)
        self.ppfc2 = nn.Linear(1024, 256)
        self.ppfc3 = nn.Linear(256, output_dim)

        # protein
        self.conv4 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(2, 2, 2),padding=(1,1,1))
        self.conv5 = nn.Conv3d(in_channels=16, out_channels=64, kernel_size=(2, 2, 2),padding=(1,1,1))
        self.conv6 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(2, 2, 2),padding=(1,1,1))
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.pfc1 = nn.Linear(864, 2048)
        self.pfc2 = nn.Linear(2048, 1024)
        self.pfc3 = nn.Linear(1024, output_dim)

        # # drug
        # self.conv1 = GCNConv(num_features_xd, num_features_xd)
        # self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        # self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        # self.fc_g1 = torch.nn.Linear(num_features_xd * 4, 1024)
        # self.fc_g2 = torch.nn.Linear(1024, output_dim)

        # drud
        dim = 32
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv10 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv11 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # dpFNN
        self.ddfc1 = nn.Linear(256, 1024)
        self.ddfc2 = nn.Linear(1024, 512)
        self.ddfc3 = nn.Linear(512, 128)

        # FNN
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input , "target" represents the protein processed by k-mers, and "dcpro" represents the protein processed by Cartesian product operation
        target = data.target
        target = target.unsqueeze(1)
        dcpro = data.dcpro
        dcpro = dcpro.unsqueeze(1)

        xd = self.conv7(dcpro)
        xd = self.relu(xd)

        xd = self.conv8(xd)
        xd = self.relu(xd)

        xd = self.conv9(xd)
        xd = self.relu(xd)
        xd = self.ppool2d(xd)

        xd = xd.view(-1, 2592)

        xd = self.ppfc1(xd)
        xd = self.relu(xd)
        xd = self.dropout(xd)
        xd = self.ppfc2(xd)
        xd = self.relu(xd)
        xd = self.dropout(xd)
        xd = self.ppfc3(xd)
        xd = self.relu(xd)
        xd = self.dropout(xd)


        # target
        xt = self.conv4(target)
        xt = self.relu(xt)
        xt = self.pool3d(xt)

        xt = self.conv5(xt)
        xt = self.relu(xt)
        xt = self.pool3d(xt)

        xt = self.conv6(xt)
        xt = self.relu(xt)
        xt = self.pool3d(xt)


        xt = xt.view(-1, 864)

        xt = self.pfc1(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        xt = self.pfc2(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        xt = self.pfc3(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)


        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv10(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv11(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        xq = torch.cat((xd,xt) , 1)

        xq = self.ddfc1(xq)
        xq = self.relu(xq)
        xq = self.dropout(xq)
        xq = self.ddfc2(xq)
        xq = self.relu(xq)
        xq = self.dropout(xq)
        xq = self.ddfc3(xq)
        xq = self.relu(xq)
        xq = self.dropout(xq)



        xc = torch.cat((x ,xq), 1)

        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out






















