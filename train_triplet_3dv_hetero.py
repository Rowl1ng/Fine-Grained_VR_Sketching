import numpy as np
import os, json
import torch
import logging
from pathlib import Path
from args import get_args

import tools.provider as provider
from tensorboardX import SummaryWriter
import time
from tools.evaluation import compute_distance, compute_acc_at_k
from dataset.Dataset_Loader import get_dataloader

from utils.utils import apply_random_rotation, save, resume
from tools.misc import get_latest_ckpt
from torch.backends import cudnn
from train_triplet_3dv import validate
np.random.seed(0)
torch.manual_seed(0)


def log_string(str, logger):
    logger.info(str)
    print(str)

def main(args):

    # '''HYPER PARAMETER'''
    cudnn.benchmark = True

    '''MODEL LOADING'''
    from models.pointnet2_cls_msg import get_model
    model = get_model(args)
    model = model.cuda()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

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
        model, optimizer, scheduler, start_epoch, valid_acc_best, log_dir = resume(
            resume_checkpoint, model, optimizer, scheduler)
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
            model, optimizer, scheduler, start_epoch, valid_acc_best, log_dir = resume(
                str(resume_checkpoint), model, optimizer, scheduler)
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
    # if args.recon:
    #     train_shape_loader, train_sketch_loader, train_network_loader, val_shape_loader, val_sketch_loader, test_shape_loader, test_sketch_loader = get_dataloader(args)
    # else:
    train_shape_loader, train_sketch_loader, val_shape_loader, val_sketch_loader, test_shape_loader, test_sketch_loader = get_dataloader(args)

    if args.train:
        best_epoch = -1
        '''TRANING'''
        logger.info('Start training...')
        for epoch in range(start_epoch, args.epochs+1):
            log_string('Epoch (%d/%s):' % (epoch, args.epochs), logger)

            # plot learning rate
            if writer is not None:
                writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], epoch)
            #
            # if args.recon:
            #     train_rec(train_sketch_loader, train_shape_loader, train_network_loader, model, optimizer, writer, epoch, args, logger)
            # else:
            train(train_sketch_loader, train_shape_loader, model, optimizer, writer, epoch, args, logger)
            if epoch == args.siamese_epoch:
                import copy
                to_load_state = copy.deepcopy(model.encoder.state_dict())
                model.encoder2.load_state_dict(to_load_state)
                log_string("Copy encoder 1 to encoder 2.", logger)
            scheduler.step()

            if epoch % args.save_freq == 0:
                log_string("Test:", logger)
                cur_metric = validate(logger, val_sketch_loader, val_shape_loader, model)
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
                save(model, optimizer, epoch + 1, scheduler, valid_acc_best, log_dir, checkpoint_path)
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
        cur_metric = validate(logger, test_sketch_loader, test_shape_loader, model, save=True, save_dir=log_dir, data_dir=args.data_dir, resume_checkpoint=args.resume_checkpoint)

    return experiment_dir

def train(sketch_dataloader, shape_dataloader, model, optimizer, writer, epoch, args, logger):
    model = model.train()
    start_time = time.time()

    for bidx, (sketches, shapes) in enumerate(zip(sketch_dataloader, shape_dataloader)):
        step = bidx + len(sketch_dataloader) * (epoch - 1)
        points = torch.cat([sketches[0], shapes[0]]).data.numpy()

        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])

        points = torch.Tensor(points)
        inputs = points.cuda()
        if args.random_rotate:
            inputs, _, _ = apply_random_rotation(inputs, rot_axis=1)
        if epoch < args.siamese_epoch:
            out = model(inputs, optimizer)
        else:
            out = model(inputs, optimizer, share=False)

        if step % args.log_freq == 0:
            duration = time.time() - start_time
            start_time = time.time()
            tl_loss = out['tl_loss']
            if writer is not None:
                writer.add_scalar('train/avg_time', duration, step)
                writer.add_scalar('train/tl', tl_loss, step)

            log_string(
                "Epoch %d Batch [%2d/%2d] Time [%3.2fs]  Triplet Loss %2.5f"
                % (epoch, bidx, len(sketch_dataloader), duration, tl_loss), logger)

def train_rec(sketch_dataloader, shape_dataloader, network_loader, model, optimizer, writer, epoch, args, logger):
    model = model.train()
    start_time = time.time()

    for bidx, (sketches, shapes, networks) in enumerate(zip(sketch_dataloader, shape_dataloader, network_loader)):
        step = bidx + len(sketch_dataloader) * (epoch - 1)
        points = torch.cat([sketches[0], shapes[0], networks[0]]).data.numpy()

        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        if args.random_rotate:
            points, _, _ = apply_random_rotation(points, rot_axis=1)

        points = torch.Tensor(points)
        inputs = points.cuda()
        out = model.rec_forward(inputs, optimizer)

        if step % args.log_freq == 0:
            duration = time.time() - start_time
            start_time = time.time()
            recon, tl_loss, loss = out['recon_loss'], out['tl_loss'], out['loss']
            if writer is not None:
                writer.add_scalar('train/avg_time', duration, step)
                writer.add_scalar('train/recon', recon, step)
                writer.add_scalar('train/tl', tl_loss, step)
                writer.add_scalar('train/loss', loss, step)

            log_string(
                "Epoch %d Batch [%2d/%2d] Time [%3.2fs]  Recon %2.5f Triplet %2.5f loss %2.5f"
                % (epoch, bidx, len(sketch_dataloader), duration, recon, tl_loss, loss), logger)

# def validate(logger, sketch_dataloader, shape_dataloader, model, save=False, save_dir=''):
#     sketch_features = []
#     shape_features = []
#     model = model.eval()
#     start_time = time.time()
#
#     with torch.no_grad():
#         for i, data in enumerate(sketch_dataloader):
#             sketch_points = data[0].cuda()
#             sketch_z = model.extract_feature(sketch_points)
#             sketch_features.append(sketch_z.data.cpu())
#
#         for i, data in enumerate(shape_dataloader):
#             shape_points = data[0].cuda()
#             shape_z = model.extract_feature(shape_points, shape=True)
#             shape_features.append(shape_z.data.cpu())
#
#     inference_duration = time.time() - start_time
#     start_time = time.time()
#
#     shape_features = torch.cat(shape_features, 0).numpy()
#     sketch_features = torch.cat(sketch_features, 0).numpy()
#     d_feat_z = compute_distance(sketch_features.copy(), shape_features.copy(), l2=True)
#     acc_at_k_feat_z = compute_acc_at_k(d_feat_z)
#     eval_duration = time.time() - start_time
#
#     if save:
#         # np.save(os.path.join(save_dir, 'shape_feat_{}.npy'.format(batch_size)), shape_features)
#         # np.save(os.path.join(save_dir, 'sketch_feat_{}.npy'.format(batch_size)), sketch_features)
#         np.save(os.path.join(save_dir, 'd_feat.npy'), d_feat_z)
#
#     log_string(
#         "Inference Time [%3.2fs]  Eval Time [%3.2fs]"
#         % (inference_duration, eval_duration), logger)
#
#     for acc_z_i, k in zip(acc_at_k_feat_z, [1, 5, 10]):
#         log_string(' * Acc@{:d} z acc {:.4f}'.format(k, acc_z_i), logger)
#
#     return acc_at_k_feat_z



if __name__ == '__main__':

    args = get_args()
    if args.windows:
        from args import get_parser
        parser = get_parser()
        args = parser.parse_args(args=[
            '--debug',
                                       '--train',
            '--type', 'hetero',
            '--lr', '1e-2',
            '--siamese_epoch', '1',
            # '--encoder', 'dgcnn',
            # '--use_aug',
            # '--aug_dir', 'synthetic_sketch_1.0',
            # '--aug_list_file', "aug/modelnet.txt",
            #                            '--encoder_data', 'hs',
                                       '--save_name', 'naive_3dv_hs_hetero3',
            '--data_dir', r'C:\Users\ll00931\Documents\chair_1005\all_networks',
                                       # '--resume_checkpoint', '3',
                                       '--epoch', '10', \
                                       '--batch_size', '12', \
                                        '--save_freq', '1',\
                                       '--log_freq', '1',\
                                       '--save_dir', r'C:\Users\ll00931\PycharmProjects\FineGrained_3DSketch'
                                       ])

    experiment_dir = main(args)
    # os.system('sh run_eval_all.sh 5 %s' % experiment_dir)
