import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from networks import LATENT_CODE_SIZE, POINT_DIM

class STN3d(nn.Module):
    """
    input transform net
    """
    def __init__(self):
        super(STN3d, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv1d(POINT_DIM, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.layers2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, POINT_DIM**2),
        )


    def forward(self, x):
        batchsize = x.size()[0]
        x = self.layers1(x)
        x = torch.max(x, 2)[0]

        x = self.layers2(x)

        iden = Variable(torch.from_numpy(np.eye(POINT_DIM).flatten().astype(np.float32))).view(1,POINT_DIM**2).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, POINT_DIM, POINT_DIM)
        return x


class STNkd(nn.Module):
    """
    feature transform net
    """
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, use_global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.layer1 = nn.Sequential(
            nn.Conv1d(POINT_DIM, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
        )

        self.use_global_feat = use_global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

        self.latent_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, LATENT_CODE_SIZE),
        )

    def forward(self, x):
        n_pts = x.size()[2]
        input_trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, input_trans)
        x = x.transpose(2, 1)
        x = self.layer1(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.layer2(x)
        x = torch.max(x, 2)[0]

        if self.use_global_feat:
            global_feat = x
        else:
            global_feat = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            global_feat = torch.cat([global_feat, pointfeat], 1)

        # now map down the global feature to a latent code
        latent_code = self.latent_layers(global_feat)
        return latent_code, input_trans # [N, LATENT_CODE_SIZE]
