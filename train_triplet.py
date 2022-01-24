import argparse
import numpy as np
import os, shutil, json
import torch
import datetime
import logging
from pathlib import Path
from args import get_args

import sys
import tools.provider as provider
import socket
import tools.misc as misc
from tensorboardX import SummaryWriter
import time
from tools.evaluation import compute_distance, compute_acc_at_k
import tools.chamfer_python as chamfer_python
from dataset.Dataset_Loader import get_dataloader

from softflow.networks import SoftPointFlow
from utils import apply_random_rotation, save, resume, visualize_point_clouds

from torch.backends import cudnn
import torch.nn as nn


def log_string(str, logger):
    logger.info(str)
    print(str)

def main(args):

    # '''HYPER PARAMETER'''
    cudnn.benchmark = True
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # resume training!!!
    #################################
    # if args.resume_checkpoint is None and os.path.exists(os.path.join(args.save_dir, 'checkpoint-latest.pt')):
    #     args.resume_checkpoint = os.path.join(args.save_dir, 'checkpoint-latest.pt')  # use the latest checkpoint
    #     print('Checkpoint is set to the latest one.')
    #################################

    '''MODEL LOADING'''
    model = SoftPointFlow(args)

    def _transform_(m):
        return nn.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    valid_acc_best = 0
    optimizer = model.make_optimizer(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    train_shape_loader, train_sketch_loader, test_shape_loader, test_sketch_loader = get_dataloader(args)

    init_data = get_init_data(args, train_shape_loader, train_sketch_loader)

    if args.resume_checkpoint is not None:
        model, optimizer, scheduler, start_epoch, valid_acc_best, log_dir = resume(
            args.resume_checkpoint, model, optimizer, scheduler)
        model.set_initialized(True)
        print('Resumed from: ' + args.resume_checkpoint)

    else:
        if args.save_name is None:
            experiment_dir = Path(os.path.join(args.save_dir, "runs", str(time.strftime('%Y-%m-%d_%H_%M_%S'))))
        else:
            experiment_dir = Path(os.path.join(args.save_dir, "runs", args.save_name))
        resume_checkpoint = experiment_dir.joinpath('checkpoints/checkpoint-latest.pt')
        if os.path.exists(resume_checkpoint):
            model, optimizer, scheduler, start_epoch, valid_acc_best, log_dir = resume(
                str(resume_checkpoint), model, optimizer, scheduler)
            model.set_initialized(True)
            print('Resumed from: ' + str(resume_checkpoint))
        else:
            start_epoch = 1
            experiment_dir.mkdir(exist_ok=True)
            with torch.no_grad():
                inputs, inputs_noisy, std_in = init_data
                inputs = inputs.to(args.gpu, non_blocking=True)
                inputs_noisy = inputs_noisy.to(args.gpu, non_blocking=True)
                std_in = std_in.to(args.gpu, non_blocking=True)
                _ = model(inputs, inputs_noisy, std_in, optimizer, init=True)
            del inputs, inputs_noisy, std_in
            print('Actnorm is initialized')

    '''CREATE DIR'''

    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir)

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


    best_epoch = -1
    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epochs+1):
        log_string('Epoch (%d/%s):' % (epoch, args.epochs), logger)

        # plot learning rate
        if writer is not None:
            writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], epoch)

        train(logger, train_sketch_loader, train_shape_loader, model, optimizer, writer, epoch, args)

        if epoch < args.stop_scheduler:
            scheduler.step()


        if epoch % args.valid_freq == 0:
            log_string("Test:", logger)
            cur_metric = validate(logger, test_sketch_loader, test_shape_loader, model)
            feat_z = np.array(cur_metric[0]).mean()  # mAP_feat_norm_z
            feat_w = np.array(cur_metric[1]).mean()  # mAP_feat_norm_z
            std = cur_metric[2]
            if args.use_z:
                top1 = feat_z
            else: # use w
                top1 = feat_w
            is_best = top1 > valid_acc_best
            if is_best:
                best_epoch = epoch
            valid_acc_best = max(top1, valid_acc_best)

            writer.add_scalar('val/val_feature_z', feat_z, epoch) # mAP_feat_norm
            writer.add_scalar('val/val_feature_w', feat_w, epoch)
            writer.add_histogram('val/noise', std, epoch)

            if is_best:
                logger.info('Save model...')
                savepath = os.path.join(str(checkpoints_dir), 'best_model.pt')
                log_string('Saving at %s' % savepath, logger)
                save(model, optimizer, epoch + 1, scheduler, valid_acc_best, log_dir, savepath)

            log_string('\n * Finished epoch {:3d}  top1: {:.4f}  best: {:.4f} @epoch {}\n'.
                       format(epoch, top1, valid_acc_best, best_epoch), logger)

        if epoch % args.save_freq == 0:
            # save(model, optimizer, epoch + 1, scheduler, best_top1, log_dir,
            #     os.path.join(str(checkpoints_dir), 'checkpoint-%d.pt' % epoch))
            save(model, optimizer, epoch + 1, scheduler, valid_acc_best, log_dir,
                os.path.join(str(checkpoints_dir), 'checkpoint-latest.pt'))


        # if epoch % args.viz_freq == 0:
        #     viz()


    logger.info('End of training...')
    writer.export_scalars_to_json(log_dir.joinpath("all_scalars.json"))
    writer.close()

    log_string('Best metric {}'.format(valid_acc_best), logger)

    return experiment_dir

def viz(model, unseen_inputs):
    with torch.no_grad():
        # reconstructions
        model.eval()
        samples = model.reconstruct(unseen_inputs)
        results = []
        for idx in range(min(16, unseen_inputs.size(0))):
            res = visualize_point_clouds(samples[idx], unseen_inputs[idx], idx,
                                         pert_order=train_loader.dataset.display_axis_order)

            results.append(res)
        res = np.concatenate(results, axis=1)
        imageio.imwrite(os.path.join(save_dir, 'images', 'SPF_epoch%d-gpu%s_recon_unseen.png' % (epoch, args.gpu)),
                        res.transpose(1, 2, 0))
        if writer is not None:
            writer.add_image('tr_vis/conditioned', torch.as_tensor(res), epoch)

        samples = model.reconstruct(seen_inputs)
        results = []
        for idx in range(min(16, seen_inputs.size(0))):
            res = visualize_point_clouds(samples[idx], seen_inputs[idx], idx,
                                         pert_order=train_loader.dataset.display_axis_order)

            results.append(res)
        res = np.concatenate(results, axis=1)
        imageio.imwrite(os.path.join(save_dir, 'images', 'SPF_epoch%d-gpu%s_recon_seen.png' % (epoch, args.gpu)),
                        res.transpose(1, 2, 0))
        if writer is not None:
            writer.add_image('tr_vis/conditioned', torch.as_tensor(res), epoch)

        num_samples = min(16, unseen_inputs.size(0))
        num_points = unseen_inputs.size(1)
        _, samples = model.sample(num_samples, num_points)
        results = []
        for idx in range(num_samples):
            res = visualize_point_clouds(samples[idx], unseen_inputs[idx], idx,
                                         pert_order=train_loader.dataset.display_axis_order)
            results.append(res)
        res = np.concatenate(results, axis=1)
        imageio.imwrite(os.path.join(save_dir, 'images', 'SPF_epoch%d-gpu%s_sample.png' % (epoch, args.gpu)),
                        res.transpose((1, 2, 0)))
        if writer is not None:
            writer.add_image('tr_vis/sampled', torch.as_tensor(res), epoch)

        print('image saved!')


def train(logger, sketch_dataloader, shape_dataloader, model, optimizer, writer, epoch, args):
    model = model.train()
    start_time = time.time()

    for bidx, (sketches, shapes) in enumerate(zip(sketch_dataloader, shape_dataloader)):
        step = bidx + len(sketch_dataloader) * (epoch - 1)
        points = torch.cat([sketches[0], shapes[0]]).data.numpy()

        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        if args.random_rotate:
            points, _, _ = apply_random_rotation(points, rot_axis=1)

        points = torch.Tensor(points)
        inputs = points.cuda(args.gpu, non_blocking=True)
        B, N, D = inputs.shape
        std = (args.std_max - args.std_min) * torch.rand_like(inputs[:, :, 0]).view(B, N, 1) + args.std_min
        eps = torch.randn_like(inputs) * std
        std_in = std / args.std_max * args.std_scale
        inputs_noisy = inputs + eps

        out = model(inputs, inputs_noisy, std_in, optimizer)
        entropy, prior, prior_nats, recon, recon_nats, tl_loss, loss, std = out['entropy'], out['prior'], out['prior_nats'],  out['recon'], out['recon_nats'], out['triplet'], out['loss'], out['std']

        if step % args.log_freq == 0:
            duration = time.time() - start_time
            start_time = time.time()
            if writer is not None:
                writer.add_scalar('train/avg_time', duration, step)
                writer.add_scalar('train/entropy', entropy, step)
                writer.add_scalar('train/prior', prior, step)
                writer.add_scalar('train/prior(nats)', prior_nats, step)
                writer.add_scalar('train/recon', recon, step)
                writer.add_scalar('train/recon(nats)', recon_nats, step)
                writer.add_scalar('train/tl', tl_loss, step)
                writer.add_scalar('train/loss', loss, step)

            log_string(
                "[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] Entropy %2.5f LatentNats %2.5f PointNats %2.5f Triplet %2.5f loss %2.5f"
                % (args.rank, epoch, bidx, len(sketch_dataloader), duration, entropy,
                   prior_nats, recon_nats, tl_loss, loss), logger)
        del inputs, inputs_noisy, std_in, out, eps

def validate(logger, sketch_dataloader, shape_dataloader, model):
    sketch_features = {item:[] for item in ['z_mu', 'w']}
    shape_features = {item:[] for item in ['z_mu', 'w']}
    std_list = []
    batch_time = misc.AverageMeter()
    end = time.time()

    model = model.eval()

    for i, data in enumerate(sketch_dataloader):
        points = data[0].cuda()
        sketch_z, sketch_w, _ = model.extract_feature(points)
        sketch_features['z_mu'].append(sketch_z.data.cpu())
        sketch_features['w'].append(sketch_w.data.cpu())

        batch_time.update(time.time() - end)

    for i, data in enumerate(shape_dataloader):
        points = data[0].cuda()
        shape_z, shape_w, std = model.extract_feature(points)
        shape_features['z_mu'].append(shape_z.data.cpu())
        shape_features['w'].append(shape_w.data.cpu())
        std_list.append(std.data.cpu())
        batch_time.update(time.time() - end)
        end = time.time()

    shape_features = {item: torch.cat(shape_features[item], 0).numpy() for item in ['z_mu', 'w']}
    sketch_features = {item: torch.cat(sketch_features[item], 0).numpy() for item in ['z_mu', 'w']}
    std = torch.cat(std_list, 0).numpy()
    d_feat_w = compute_distance(sketch_features['w'].copy(), shape_features['w'].copy(), l2=True)
    d_feat_z = compute_distance(sketch_features['z_mu'].copy(), shape_features['z_mu'].copy(), l2=True)
    acc_at_k_feat_w = compute_acc_at_k(d_feat_w)
    acc_at_k_feat_z = compute_acc_at_k(d_feat_z)

    for acc_z_i, acc_w_i, k in zip(acc_at_k_feat_z, acc_at_k_feat_w, [1, 5, 10]):
        log_string(' * Acc@{:d} z acc {:.4f}\tw acc {:.4f} '.format(k, acc_z_i, acc_w_i), logger)

    return acc_at_k_feat_z, acc_at_k_feat_w, std


def get_init_data(args, train_shape_loader, train_sketch_loader):

    shapes, _ = next(iter(train_shape_loader))
    sketches, _ = next(iter(train_sketch_loader))

    inputs = torch.cat([sketches, shapes])
    B, N, D = inputs.shape
    std = (args.std_max - args.std_min) * torch.rand_like(inputs[:, :, 0]).view(B, N, 1) + args.std_min
    eps = torch.randn_like(inputs) * std
    std_in = std / args.std_max * args.std_scale
    inputs_noisy = inputs + eps

    return (inputs, inputs_noisy, std_in)



if __name__ == '__main__':

    args = get_args()
    if args.debug:
        from args import get_parser
        parser = get_parser()
        args = parser.parse_args(args=['--debug',
                                       '--use_triplet',
                                       '--use_z',
                                       # '--zdim', '512',
                                       '--use_deterministic_encoder',
                                       '--use_pointnet2_encoder',
                                       # '--use_dependant_noise',
                                       '--save_name', 'train11',
                                       # '--reparameterize',
                                       # '--clip_gradient',
                                       # '--use_latent_flow',
                                       '--lr', '2e-3',
                                       '--list_file', r'C:\Users\ll00931\Documents\chair_1005\list\unique\{}_45participants.txt', \
                                       '--sketch_dir', r'C:\Users\ll00931\Documents\chair_1005\pointcloud\final_set', \
                                       '--shape_dir', r"C:\Users\ll00931\Documents\chair_1005\pointcloud\shape", \
                                       # '--resume_checkpoint', r'runs/2020-10-18_21_58_40/checkpoints/best_model.pt',
                                       '--epoch','10', \
                                       '--batch_size', '4', \
                                        '--n_flow', '12', \
                                        '--multi_freq', '4',\
                                        '--n_flow_AF', '9',\
                                        '--h_dims_AF', '256-256-256', \
                                        '--save_freq', '1',\
                                       '--valid_freq', '1',\
                                       '--log_freq', '10',\
                                       '--save_dir', r'C:\Users\ll00931\PycharmProjects\FineGrained_3DSketch'
                                       ])

    experiment_dir = main(args)
    # os.system('sh run_eval_all.sh 5 %s' % experiment_dir)
