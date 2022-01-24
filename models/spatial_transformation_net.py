import torch.nn as nn
import torch
def _init_identity(module, dim):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 0.0)
        with torch.no_grad():
            module.bias.data = torch.eye(dim).view(-1)
class PointSpatialTransformer(nn.Module):
    def __init__(self, dim):
        super(PointSpatialTransformer, self).__init__()
        self.dim = dim
        self.features = nn.Sequential(nn.Conv1d(dim, 64, kernel_size=1),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(True),
                                      nn.Conv1d(64, 128, kernel_size=1),
                                      nn.BatchNorm1d(128),
                                      nn.ReLU(True),
                                      nn.Conv1d(128, 1024, kernel_size=1),
                                      nn.BatchNorm1d(1024),
                                      nn.ReLU(True))

        self.regressor = nn.Sequential(nn.Linear(1024, 512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(True),
                                       nn.Linear(512, 256))

        self.transform = nn.Linear(256, dim * dim)

        # Initialize initial transformation to be identity
        _init_identity(self.transform, dim)
    def get_transpose(self, x):
        x = self.features(x)
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        x = self.transform(x)
        # resize x into 2D square matrix
        x = x.view(x.size(0), self.dim, self.dim)
        return x
    def forward(self, x):
        x = x.transpose(2, 1)
        trans_inp = self.get_transpose(x)
        x = torch.bmm(trans_inp, x).transpose(2, 1)
        return x#, trans_inp

if __name__ == '__main__':
    augment_net = PointSpatialTransformer(dim=3).cuda()
    x = torch.randn([2, 5, 3])
    x = x.cuda()
    y, trans_inp = augment_net(x)
    import numpy as np
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    print('x', x.shape)
    print('y', y.shape)
    eq = np.equal(x, y).all()
    print(eq)
    print(trans_inp.shape)