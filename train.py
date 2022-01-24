import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.backends import cudnn
from torch import optim
from tensorboardX import SummaryWriter
import sys
import os, json
import logging
from pathlib import Path
from tools.misc import get_latest_ckpt
import faulthandler
import time
import gc
from softflow.networks import SoftPointFlow
from args import get_args
from utils.utils import AverageValueMeter, set_random_seed, apply_random_rotation, save, resume, visualize_point_clouds
# from datasets import get_trainset, get_testset, init_np_seed

faulthandler.enable()

def main_worker(args):
    # basic setup
    cudnn.benchmark = True

    # multi-GPU setup
    model = SoftPointFlow(args)
    def _transform_(m):
        return nn.DataParallel(m)
    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    from dataset.Dataset_Loader import get_dataloader
    train_shape_loader, train_sketch_loader, val_shape_loader, val_sketch_loader, test_shape_loader, test_sketch_loader = get_dataloader(args)

    start_epoch = 1
    valid_loss_best = 987654321
    optimizer = model.make_optimizer(args)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.resume_checkpoint is not None:
        model, optimizer, scheduler, start_epoch, valid_loss_best, log_dir = resume(
            args.resume_checkpoint, model, optimizer, scheduler)
        model.set_initialized(True)
        print('Resumed from: ' + args.resume_checkpoint)
    else:
        experiment_dir = Path(os.path.join(args.save_dir, "runs", args.save_name))
        init_data = get_init_data(args, train_shape_loader, train_sketch_loader)
        experiment_dir.mkdir(exist_ok=True)
        checkpoints_dir = experiment_dir.joinpath('checkpoints/')
        checkpoints_dir.mkdir(exist_ok=True)
        resume_checkpoint = get_latest_ckpt(checkpoints_dir)
        if resume_checkpoint is not None:
            model, optimizer, scheduler, start_epoch, valid_loss_best, log_dir = resume(
                resume_checkpoint, model, optimizer, scheduler)
            model.set_initialized(True)
            print('Resumed from: ' + args.resume_checkpoint)
        else:
            start_epoch = 1

        with torch.no_grad():
            inputs, inputs_noisy, std_in = init_data
            inputs = inputs.to(args.gpu, non_blocking=True)
            inputs_noisy = inputs_noisy.to(args.gpu, non_blocking=True)
            std_in = std_in.to(args.gpu, non_blocking=True)
            _ = model(inputs, inputs_noisy, std_in, optimizer, init=True)
        del inputs, inputs_noisy, std_in
        print('Actnorm is initialized')

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

    logger.info('Start training...')
    logger.info("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs+1):
        start_time = time.time()
        if writer is not None:
            writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], epoch)

        model.train()
        # train for one epoch

        for bidx, (sketches, shapes) in enumerate(zip(train_sketch_loader, train_shape_loader)):
            step = bidx + len(train_sketch_loader) * (epoch - 1)
            shapes = shapes[0]
            sketches = sketches[0]
            if args.random_rotate:
                tr_batch, _, _ = apply_random_rotation(
                    tr_batch, rot_axis=train_loader.dataset.gravity_axis)
            if args.ae_input == 'sketch':
                origin = sketches.cuda(args.gpu, non_blocking=True)
            else:
                origin = shapes.cuda(args.gpu, non_blocking=True)
            inputs = sketches.cuda(args.gpu, non_blocking=True)
            B, N, D = inputs.shape
            std = (args.std_max - args.std_min) * torch.rand_like(inputs[:,:,0]).view(B,N,1) + args.std_min

            eps = torch.randn_like(inputs) * std
            std_in = std / args.std_max * args.std_scale
            inputs_noisy = inputs + eps
            out = model(origin, inputs_noisy, std_in, optimizer, step, writer)
            entropy, prior_nats, recon_nats, loss = out['entropy'], out['prior_nats'], out['recon_nats'], out['loss']
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()
                if writer is not None:
                    writer.add_scalar('train/avg_time', duration, step)
                logger.info("Epoch %d Batch [%2d/%2d] Time [%3.2fs] Entropy %2.5f LatentNats %2.5f PointNats %2.5f loss %2.5f"
                      % ( epoch, bidx, len(train_sketch_loader), duration, entropy,
                         prior_nats, recon_nats, loss))
            del inputs, inputs_noisy, std_in, out, eps
            gc.collect()

        if epoch < args.stop_scheduler:
            scheduler.step()

        if epoch % args.valid_freq == 0:
            with torch.no_grad():
                model.eval()
                valid_loss = 0.0
                valid_entropy = 0.0
                valid_prior = 0.0
                valid_prior_nats = 0.0
                valid_recon = 0.0
                valid_recon_nats = 0.0
                for bidx, (sketches, shapes) in enumerate(zip(val_sketch_loader, val_shape_loader)):
                    step = bidx + len(val_sketch_loader) * epoch
                    shapes = shapes[0]
                    sketches = sketches[0]
                    if args.random_rotate:
                        tr_batch, _, _ = apply_random_rotation(
                            tr_batch, rot_axis=train_loader.dataset.gravity_axis)


                    if args.ae_input == 'sketch':
                        origin = sketches.cuda(args.gpu, non_blocking=True)
                    else:
                        origin = shapes.cuda(args.gpu, non_blocking=True)
                    inputs = sketches.cuda(args.gpu, non_blocking=True)
                    B, N, D = inputs.shape
                    std = (args.std_max - args.std_min) * torch.rand_like(inputs[:,:,0]).view(B,N,1) + args.std_min

                    eps = torch.randn_like(inputs) * std
                    std_in = std / args.std_max * args.std_scale
                    inputs_noisy = inputs + eps
                    out = model(origin, inputs_noisy, std_in, optimizer, valid=True)
                    valid_loss += out['loss'] / len(val_sketch_loader)
                    valid_entropy += out['entropy'] / len(val_sketch_loader)
                    valid_prior += out['prior'] / len(val_sketch_loader)
                    valid_prior_nats += out['prior_nats'] / len(val_sketch_loader)
                    valid_recon += out['recon'] / len(val_sketch_loader)
                    valid_recon_nats += out['recon_nats'] / len(val_sketch_loader)
                    del inputs, inputs_noisy, std_in, out, eps
                    gc.collect()

                if writer is not None:
                    writer.add_scalar('valid/entropy', valid_entropy, epoch)
                    writer.add_scalar('valid/prior', valid_prior, epoch)
                    writer.add_scalar('valid/prior(nats)', valid_prior_nats, epoch)
                    writer.add_scalar('valid/recon', valid_recon, epoch)
                    writer.add_scalar('valid/recon(nats)', valid_recon_nats, epoch)
                    writer.add_scalar('valid/loss', valid_loss, epoch)
                
                duration = time.time() - start_time
                logger.info("[Valid] Epoch %d Time [%3.2fs] Entropy %2.5f LatentNats %2.5f PointNats %2.5f loss %2.5f loss_best %2.5f"
                    % (epoch, duration, valid_entropy, valid_prior_nats, valid_recon_nats, valid_loss, valid_loss_best))
                if valid_loss < valid_loss_best:
                    valid_loss_best = valid_loss
                    # if not args.distributed or (args.rank % ngpus_per_node == 0):
                    #     save(model, optimizer, epoch + 1, scheduler, valid_loss_best, log_dir,
                    #         os.path.join(save_dir, 'checkpoint-best.pt'))
                    #     print('best model saved!')

        if epoch % args.save_freq == 0 :
            checkpoint_path = os.path.join(str(checkpoints_dir), 'checkpoint-{}.pt'.format(str(epoch).zfill(5)))
            save(model, optimizer, epoch + 1, scheduler, valid_loss_best, log_dir, checkpoint_path)
            # save(model, optimizer, epoch + 1, scheduler, valid_loss_best, log_dir,
            #     os.path.join(save_dir, 'checkpoint-latest.pt'))
            logger.info('Save epoch {} checkpoint to: {}'.format(epoch, checkpoint_path))

            # save visualizations
        # if epoch % args.viz_freq == 0:
        #     with torch.no_grad():
        #         # reconstructions
        #         model.eval()
        #         samples = model.reconstruct(unseen_inputs)
        #         results = []
        #         for idx in range(min(16, unseen_inputs.size(0))):
        #             res = visualize_point_clouds(samples[idx], unseen_inputs[idx], idx,
        #                                         pert_order=train_loader.dataset.display_axis_order)
        #
        #             results.append(res)
        #         res = np.concatenate(results, axis=1)
        #         imageio.imwrite(os.path.join(save_dir, 'images', 'SPF_epoch%d-gpu%s_recon_unseen.png' % (epoch, args.gpu)),
        #                         res.transpose(1, 2, 0))
        #         if writer is not None:
        #             writer.add_image('tr_vis/conditioned', torch.as_tensor(res), epoch)
        #
        #         samples = model.reconstruct(seen_inputs)
        #         results = []
        #         for idx in range(min(16, seen_inputs.size(0))):
        #             res = visualize_point_clouds(samples[idx], seen_inputs[idx], idx,
        #                                         pert_order=train_loader.dataset.display_axis_order)
        #
        #             results.append(res)
        #         res = np.concatenate(results, axis=1)
        #         imageio.imwrite(os.path.join(save_dir, 'images', 'SPF_epoch%d-gpu%s_recon_seen.png' % (epoch, args.gpu)),
        #                         res.transpose(1, 2, 0))
        #         if writer is not None:
        #             writer.add_image('tr_vis/conditioned', torch.as_tensor(res), epoch)
        #
        #         num_samples = min(16, unseen_inputs.size(0))
        #         num_points = unseen_inputs.size(1)
        #         _, samples = model.sample(num_samples, num_points)
        #         results = []
        #         for idx in range(num_samples):
        #             res = visualize_point_clouds(samples[idx], unseen_inputs[idx], idx,
        #                                         pert_order=train_loader.dataset.display_axis_order)
        #             results.append(res)
        #         res = np.concatenate(results, axis=1)
        #         imageio.imwrite(os.path.join(save_dir, 'images', 'SPF_epoch%d-gpu%s_sample.png' % (epoch, args.gpu)),
        #                         res.transpose((1, 2, 0)))
        #         if writer is not None:
        #             writer.add_image('tr_vis/sampled', torch.as_tensor(res), epoch)
        #
        #         print('image saved!')

def get_init_data(args, train_shape_loader, train_sketch_loader):

    shapes, _ = next(iter(train_shape_loader))
    sketches, _ = next(iter(train_sketch_loader))
    B, N, D = shapes.shape
    std = (args.std_max - args.std_min) * torch.rand_like(sketches[:,:,0]).view(B,N,1) + args.std_min
    eps = torch.randn_like(sketches) * std
    std_in = std / args.std_max * args.std_scale
    inputs_noisy = sketches + eps

    return (shapes, inputs_noisy, std_in)

def main(args):
    # args = get_args()
    set_random_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, 'images'))

    with open(os.path.join(args.save_dir, 'command.sh'), 'w') as f:
        f.write('python -X faulthandler ' + ' '.join(sys.argv))
        f.write('\n')

    main_worker(args)


if __name__ == '__main__':
    args = get_args()
    if args.windows:
        from args import get_parser
        parser = get_parser()
        args = parser.parse_args(args=[
            # '--debug',
                                        '--recon',
                                       '--train',
                                    '--ae_data', 'hs',
                                    '--ae_input', 'sketch',
                                       '--save_name', 'ae_pn_nf',
                                        '--use_deterministic_encoder',
            '--list_file', r'C:\Users\ll00931\Documents\chair_1005\list\unique\{}.txt', \
            '--sketch_dir', r'C:\Users\ll00931\Documents\chair_1005\pointcloud\final_set', \
            '--shape_dir', r"C:\Users\ll00931\Documents\chair_1005\pointcloud\shape",  \
                                       # '--resume_checkpoint', r'runs/ae_pn2',
                                       '--epoch', '100', \
                                       '--batch_size', '64', \
                                        '--save_freq', '10',\
                                       '--log_freq', '1',\
                                       '--save_dir', r'C:\Users\ll00931\PycharmProjects\FineGrained_3DSketch'
                                       ])

    main(args)
