import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction
import torch
from models.chamfer_python import distChamfer

class PointDecoder(nn.Module):
    def __init__(self, zdim, num_points=2048):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        # self.fc1 = nn.Linear(100, 128)
        # self.fc2 = nn.Linear(128, 256)
        # self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(zdim, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()
    def forward(self, x):
        batchsize = x.size()[0]
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.th(self.fc5(x))
        x = x.view(batchsize, self.num_points, 3)
        return x


class PointEncoder(nn.Module):
    def __init__(self, feat_dim):
        super(PointEncoder, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 0, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, feat_dim)
    def forward(self, xyz):
        xyz = xyz.transpose(2, 1)
        B, _, _ = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024) # [bs, 512]
        feat = self.fc1(x)
        return feat

class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, zdim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        ms = F.relu(self.fc_bn1(self.fc1(x)))
        ms = F.relu(self.fc_bn2(self.fc2(ms)))
        ms = self.fc3(ms)

        return ms

class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        if args.encoder == 'pn':
            self.encoder = Encoder(zdim=args.feat_dim)
        else:
            self.encoder = PointEncoder(feat_dim=args.feat_dim)
        self.decoder = PointDecoder(args.feat_dim, args.shape_points)

    def forward(self, xyz):
        feat = self.encoder(xyz)
        recon = self.decoder(feat)
        return recon

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.recon = args.recon
        self.w1 = args.w1
        self.w2 = args.w2
        self.gradient_clip = args.gradient_clip
        from tools.custom_loss import OnlineTripletLoss
        if args.hard_negative_mining:
            from dataset.TripletSampler import HardestNegativeTripletSelector
            self.crt_tl = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(margin=args.margin, sketch_anchor=args.sketch_anchor))
        else:
            if args.recon:
                from dataset.TripletSampler import AllNegativeTripletSelector_network
                self.crt_tl = OnlineTripletLoss(args.margin, AllNegativeTripletSelector_network())
                # from dataset.TripletSampler import AllNegativeTripletSelector
                # self.crt_tl = OnlineTripletLoss(args.margin, AllNegativeTripletSelector())

            else:
                if args.loss == 'tl':
                    from dataset.TripletSampler import AllNegativeTripletSelector
                    self.crt_tl = OnlineTripletLoss(args.margin, AllNegativeTripletSelector(symmetric=args.symmetric))
                elif args.loss == 'cl_1': #contrastive loss
                    if args.use_aug_loss:
                        from tools.custom_loss import ContrastiveLoss_1_aug
                        self.crt_tl = ContrastiveLoss_1_aug(temperature=args.tao)
                    else:
                        from tools.custom_loss import ContrastiveLoss_1
                        self.crt_tl = ContrastiveLoss_1(temperature=args.tao)
                elif args.loss == 'cl_2':  # contrastive loss
                    from tools.custom_loss import ContrastiveLoss_2
                    self.crt_tl = ContrastiveLoss_2(temperature=args.tao)
                elif args.loss == 'cl_3':  # contrastive loss_3
                    from tools.custom_loss import ContrastiveLoss_3
                    self.crt_tl = ContrastiveLoss_3(temperature=args.tao)

                else:
                    from tools.custom_loss import RegressionLoss
                    self.crt_tl = RegressionLoss(args)
        self.stn = None
        if args.stn:
            from models.spatial_transformation_net import PointSpatialTransformer
            self.stn = PointSpatialTransformer(dim=3)

        self.encoder2 = None
        if args.encoder == 'pn2':
            self.encoder = PointEncoder(feat_dim=args.feat_dim)
            if args.type == 'hetero':
                self.encoder2 = PointEncoder(feat_dim=args.feat_dim)
        elif args.encoder == 'pn':
            self.encoder = Encoder(zdim=args.feat_dim)
        elif args.encoder == 'dgcnn':
            from models.dgcnn import DGCNN
            self.encoder = DGCNN(args)
        elif args.encoder == 'foldnet':
            from models.foldnet import FoldNet_Encoder
            self.encoder = FoldNet_Encoder(args)
        else:
            NotImplementedError

        self.decoder_name = args.decoder
        if self.recon:
            if args.decoder == 'mlp':
                self.decoder = PointDecoder(args.feat_dim, args.num_points)
            elif args.decoder == 'foldnet':
                from models.foldnet import FoldNet_Decoder
                self.decoder = FoldNet_Decoder(args)
            from chamferdist import ChamferDistance

            self.loss = ChamferDistance()

    def ae_recon(self, xyz):
        feat = self.encoder(xyz)
        recon = self.decoder(feat)
        return recon

    def ae_forward(self, inputs, target, opt):
        opt.zero_grad()
        feat = self.encoder(inputs)
        recon = self.decoder(feat)
        loss = self.loss(recon, target, bidirectional=True)
        loss.backward()
        opt.step()
        return loss


    def rec_forward(self, xyz, opt):
        opt.zero_grad()

        B, _, _ = xyz.shape
        mini_batch = int(B/3)
        network = xyz[2*mini_batch:, :, :]
        feat = self.encoder(xyz)

        sketch_feat = feat[:mini_batch, :]
        recon = self.decoder(sketch_feat)
        dist1, dist2, _, _ = distChamfer(network, recon)
        rec_loss = (torch.mean(dist1)) + (torch.mean(dist2))

        tl_loss = self.crt_tl(feat, xyz)

        loss = self.w1 * tl_loss + self.w2 * rec_loss

        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip)
        opt.step()

        return {
            'tl_loss': tl_loss,
            'recon_loss': rec_loss,
            'loss': loss
        }


    def forward(self, xyz, shape_ids, opt, share=True):
        opt.zero_grad()
        if self.stn is not None:
            B = xyz.shape[0]
            mini_batch = int(B / 2)
            sketchs, shapes = torch.split(xyz, [mini_batch, mini_batch], dim=0)
            x_recon = self.stn(sketchs)
            xyz = torch.cat([x_recon, shapes], axis=0)
        if not share:
            B = xyz.shape[0]
            feat1 = self.encoder(xyz[:int(B/2), :, :])
            feat2 = self.encoder2(xyz[int(B/2):, :, :])
            feat = torch.cat([feat1, feat2])
        else:
            feat = self.encoder(xyz)

        tl_loss = self.crt_tl(feat, shape_ids, xyz)

        loss = tl_loss

        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip)
        opt.step()

        return {
            'tl_loss': tl_loss,
        }

    def extract_feature(self, xyz, shape=False):
        # if self.stn is not None and not shape:
        #     xyz = self.stn(xyz)
        if self.encoder2 is not None and shape:
            feat = self.encoder2(xyz)
        else:
            feat = self.encoder(xyz)
        return feat



