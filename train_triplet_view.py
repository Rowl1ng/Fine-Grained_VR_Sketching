import numpy as np
import os, json
import torch
import logging
from pathlib import Path
from args import get_args

import tools.provider as provider
from tensorboardX import SummaryWriter
import time
from tools.evaluation import compute_distance#, compute_acc_at_k
from dataset.Dataset_Loader import get_2d_loader

from utils.utils import save_view, resume_view
from tools.misc import get_latest_ckpt, clip_gradient
from torch.backends import cudnn
from torch.autograd import Variable

np.random.seed(0)
torch.manual_seed(0)


def log_string(str, logger):
    logger.info(str)
    print(str)

def main(args):

    # '''HYPER PARAMETER'''
    cudnn.benchmark = True

    '''MODEL LOADING'''
    import models.ngram_sbr_net as ngvnn
    if args.use_pn2:
        from models.pointnet2_cls_msg import PointEncoder
        net_p = PointEncoder(feat_dim=args.feat_dim).cuda()
    else:
        net_p = ngvnn.Net_Prev(pretraining=False, num_views=12).cuda()

    if args.backbone == 'vgg11':
        net_whole = ngvnn.Net_Whole().cuda()
    elif args.backbone == 'resnet50':
        net_whole = ngvnn.Net_Whole_resnet().cuda()

    shape_optim = torch.optim.SGD(net_p.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    base_param_ids = set(map(id, net_whole.features.parameters()))
    new_params = [p for p in net_whole.parameters() if id(p) not in base_param_ids]
    param_groups = [
        {'params': net_whole.features.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]
    sketch_optim = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    shape_scheduler = torch.optim.lr_scheduler.StepLR(shape_optim, step_size=args.step_size, gamma=args.gamma)
    sketch_scheduler = torch.optim.lr_scheduler.StepLR(sketch_optim, step_size=args.step_size, gamma=args.gamma)

    if args.resume_checkpoint:
        if args.train:
            # finetune
            resume_checkpoint = args.resume_checkpoint
        else:
            # eval
            resume_checkpoint = Path(os.path.join(args.save_dir, "runs", args.save_name, 'checkpoints',  'checkpoint-{}.pt'.format(str(args.resume_checkpoint).zfill(5))))

        if not os.path.exists(resume_checkpoint):
            print('Best checkpoint Not Found!')
            return None
        net_p, net_whole, shape_optim, sketch_optim, shape_sche, sketch_sche, start_epoch, valid_acc_best = resume_view(
            resume_checkpoint, net_p, net_whole, shape_optim, sketch_optim, shape_scheduler, sketch_scheduler)
        print('Resumed from: ' + str(resume_checkpoint))
        if args.train:
            start_epoch = 1
        experiment_dir = Path(os.path.join(args.save_dir, "runs", args.save_name))
        experiment_dir.mkdir(exist_ok=True)
        checkpoints_dir = experiment_dir.joinpath('checkpoints/')
        checkpoints_dir.mkdir(exist_ok=True)
    else:
        if args.save_name is None:
            experiment_dir = Path(os.path.join(args.save_dir, "runs", str(time.strftime('%Y-%m-%d_%H_%M_%S'))))
        else:
            experiment_dir = Path(os.path.join(args.save_dir, "runs", args.save_name))
        # resume_checkpoint = experiment_dir.joinpath('checkpoints/checkpoint-latest.pt')
        experiment_dir.mkdir(exist_ok=True)
        checkpoints_dir = experiment_dir.joinpath('checkpoints/')
        checkpoints_dir.mkdir(exist_ok=True)
        resume_checkpoint = get_latest_ckpt(checkpoints_dir)
        if resume_checkpoint is not None:
            net_p, net_whole, shape_optim, sketch_optim, shape_sche, sketch_sche, start_epoch, valid_acc_best = resume_view(
                resume_checkpoint, net_p, net_whole, shape_optim, sketch_optim, shape_scheduler, sketch_scheduler)
            print('Resumed from: ' + str(resume_checkpoint))
        else:
            start_epoch = 1
            valid_acc_best = 0

    '''CREATE DIR'''

    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir)

    if args.train:
        '''LOG'''
        config_f = open(os.path.join(log_dir, 'config.json'), 'w')
        json.dump(vars(args), config_f)
        config_f.close()

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''DATA LOADING'''
    log_string('Load dataset ...', logger)
    train_shape_loader, train_sketch_loader, val_shape_loader, val_sketch_loader, test_shape_loader, test_sketch_loader = get_2d_loader(args)

    if args.train:
        from tools.custom_loss import OnlineTripletLoss
        from dataset.TripletSampler import AllNegativeTripletSelector
        crt_tl = OnlineTripletLoss(args.margin, AllNegativeTripletSelector())

        best_epoch = -1
        '''TRANING'''
        logger.info('Start training...')
        for epoch in range(start_epoch, args.epochs+1):
            log_string('Epoch (%d/%s):' % (epoch, args.epochs), logger)

            # plot learning rate
            if writer is not None:
                writer.add_scalar('lr/optimizer', shape_scheduler.get_lr()[0], epoch)
            #
            train(crt_tl, train_sketch_loader, train_shape_loader, net_p, net_whole, shape_optim, sketch_optim, writer, epoch, args, logger)
            shape_scheduler.step()
            sketch_scheduler.step()
            if epoch % args.save_freq == 0:
                log_string("Test:", logger)
                cur_metric = validate(args, logger, val_sketch_loader, val_shape_loader, net_p, net_whole)
                top1 = cur_metric[0]
                is_best = top1 > valid_acc_best
                if is_best:
                    best_epoch = epoch
                valid_acc_best = max(top1, valid_acc_best)

                writer.add_scalar('val/top-1', top1, epoch)
                writer.add_scalar('val/top-5', cur_metric[1], epoch)
                writer.add_scalar('val/top-10', cur_metric[2], epoch)

                log_string('\n * Finished epoch {:3d}  top1: {:.4f}  best: {:.4f} @epoch {}\n'.
                           format(epoch, top1, valid_acc_best, best_epoch), logger)
                checkpoint_path = os.path.join(str(checkpoints_dir), 'checkpoint-{}.pt'.format(str(epoch).zfill(5)))
                save_view(epoch, net_p, net_whole, shape_optim, sketch_optim, shape_scheduler, sketch_scheduler, valid_acc_best, log_dir, checkpoint_path)
                logger.info('Save epoch {} checkpoint to: {}'.format(epoch, checkpoint_path))

        logger.info('End of training...')
        writer.export_scalars_to_json(log_dir.joinpath("all_scalars.json"))
        writer.close()

        log_string('Best metric {}'.format(valid_acc_best), logger)
    else:
        logger.info('Start testing...')

        # log_string("Test on val set:", logger)
        # cur_metric = validate(logger, val_sketch_loader, val_shape_loader, model, save=False, save_dir=log_dir, batch_size=args.batch_size)

        log_string("Test on test set:", logger)
        cur_metric = validate(args, logger, test_sketch_loader, test_shape_loader, net_p, net_whole, save=True, save_dir=log_dir)

    return experiment_dir

def train(crt_tl, sketch_dataloader, shape_dataloader, net_p, net_whole, shape_optim, sketch_optim, writer, epoch, args, logger):
    net_p = net_p.train()
    net_whole = net_whole.train()
    start_time = time.time()

    for bidx, (sketches_data, shapes_data) in enumerate(zip(sketch_dataloader, shape_dataloader)):
        step = bidx + len(sketch_dataloader) * (epoch - 1)
        shapes = shapes_data[0]
        if args.use_pn2:
            points = shapes.data.numpy()
            points = provider.random_point_dropout(points)
            points = provider.random_scale_point_cloud(points)
            points = provider.shift_point_cloud(points)
            shapes = torch.Tensor(points)
        else:
            shapes = shapes.view(shapes.size(0) * shapes.size(1), shapes.size(2), shapes.size(3), shapes.size(4))
            # expanding: (bz * 12) x 3 x 224 x 224
            shapes = shapes.expand(shapes.size(0), 3, shapes.size(2), shapes.size(3))
        sketches = sketches_data[0]
        sketches = sketches.view(sketches.size(0), sketches.size(2), sketches.size(3), sketches.size(4))

        shapes_v = Variable(shapes.cuda())
        sketches_v = Variable(sketches.cuda())

        shape_feat = net_p(shapes_v)
        sketch_feat = net_whole(sketches_v)
        feat = torch.cat([sketch_feat, shape_feat])

        tl_loss = crt_tl(feat, None)
        sketch_optim.zero_grad()
        shape_optim.zero_grad()

        tl_loss.backward()
        clip_gradient(sketch_optim, args.gradient_clip)
        clip_gradient(shape_optim, args.gradient_clip)
        sketch_optim.step()
        shape_optim.step()

        if step % args.log_freq == 0:
            duration = time.time() - start_time
            start_time = time.time()
            if writer is not None:
                writer.add_scalar('train/avg_time', duration, step)
                writer.add_scalar('train/tl', tl_loss, step)

            log_string(
                "Epoch %d Batch [%2d/%2d] Time [%3.2fs]  Triplet Loss %2.5f"
                % (epoch, bidx, len(sketch_dataloader), duration, tl_loss), logger)

def compute_acc_at_k(d_feat):
    count_1 = 0
    count_5 = 0
    count_10 = 0
    pair_sort = np.argsort(d_feat)
    query_num = pair_sort.shape[0]
    for idx1 in range(query_num):
        if idx1 in pair_sort[idx1, 0:1]:
            count_1 = count_1 + 1
        if idx1 in pair_sort[idx1, 0:5]:
            count_5 = count_5 + 1
        if idx1 in pair_sort[idx1, 0:10]:
            count_10 = count_10 + 1

    acc_1 = count_1 / float(query_num)
    acc_5 = count_5 / float(query_num)
    acc_10 = count_10 / float(query_num)
    return [acc_1, acc_5, acc_10]

def validate(args, logger, sketch_dataloader, shape_dataloader, net_p, net_whole, save=False, save_dir=''):
    sketch_features = []
    shape_features = []
    net_p = net_p.eval()
    net_whole =net_whole.eval()
    start_time = time.time()

    with torch.no_grad():
        for i, data in enumerate(sketch_dataloader):
            sketches = data[0]
            sketches = sketches.view(sketches.size(0) * sketches.size(1), sketches.size(2), sketches.size(3), sketches.size(4))
            # expanding: (bz * 12) x 3 x 224 x 224
            sketches = sketches.expand(sketches.size(0), 3, sketches.size(2), sketches.size(3))
            sketches_v = Variable(sketches.cuda())
            sketch_feat = net_whole(sketches_v)

            sketch_features.append(sketch_feat.data.cpu())

        for i, data in enumerate(shape_dataloader):
            shapes = data[0]
            if not args.use_pn2:
                shapes = shapes.view(shapes.size(0) * shapes.size(1), shapes.size(2), shapes.size(3), shapes.size(4))
                # expanding: (bz * 12) x 3 x 224 x 224
                shapes = shapes.expand(shapes.size(0), 3, shapes.size(2), shapes.size(3))
            shapes_v = Variable(shapes.cuda())
            shape_feat = net_p(shapes_v)
            shape_features.append(shape_feat.data.cpu())

    inference_duration = time.time() - start_time
    start_time = time.time()

    shape_features = torch.cat(shape_features, 0).numpy()
    sketch_features = torch.cat(sketch_features, 0).numpy()
    d_feat_z = compute_distance(sketch_features.copy(), shape_features.copy(), l2=True)
    acc_at_k_feat_z = compute_acc_at_k(d_feat_z)
    eval_duration = time.time() - start_time

    if save:
        # np.save(os.path.join(save_dir, 'shape_feat_{}.npy'.format(batch_size)), shape_features)
        # np.save(os.path.join(save_dir, 'sketch_feat_{}.npy'.format(batch_size)), sketch_features)
        np.save(os.path.join(save_dir, 'd_feat.npy'), d_feat_z)

    log_string(
        "Inference Time [%3.2fs]  Eval Time [%3.2fs]"
        % (inference_duration, eval_duration), logger)

    for acc_z_i, k in zip(acc_at_k_feat_z, [1, 5, 10]):
        log_string(' * Acc@{:d} z acc {:.4f}'.format(k, acc_z_i), logger)

    return acc_at_k_feat_z



if __name__ == '__main__':
    args = get_args()
    if args.windows:
        from args import get_parser
        parser = get_parser()
        args = parser.parse_args(args=[
            '--debug',
            '--train',
            # '--loss', 'rl',
            '--backbone', 'resnet50',
            '--lr', '1e-2',
            '--use_pn2',
            '--list_file', 'hs\{}_view.txt',
            '--save_name', '2d_sketch_pn2_resnet_1',
            '--data_dir', r'C:\Users\ll00931\Documents\chair_1005\all_networks',
            # '--resume_checkpoint', '3',
            '--epoch', '10', \
            '--batch_size', '4', \
            '--save_freq', '1',\
            '--log_freq', '1',\
            '--save_dir', r'C:\Users\ll00931\PycharmProjects\FineGrained_3DSketch'
            ])

    experiment_dir = main(args)
    # os.system('sh run_eval_all.sh 5 %s' % experiment_dir)
