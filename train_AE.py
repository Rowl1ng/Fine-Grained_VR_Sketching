import numpy as np
import os, json
import torch
import logging
from pathlib import Path
from args import get_args

import tools.provider as provider
import tools.misc as misc
from tensorboardX import SummaryWriter
import time
from utils.utils import apply_random_rotation, save, resume
from tools.misc import get_latest_ckpt

from torch.backends import cudnn
from models.pointnet2_cls_msg import AE
from models.chamfer_python import distChamfer

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
    # model = AE(args)
    model = model.cuda()
    valid_recon_best = 100000
    # if args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(
    #         model.parameters(),
    #         lr=args.learning_rate,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=args.weight_decay
    #     )
    # else:
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001 * 32 / args.batch_size, betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    if not args.train and args.resume_checkpoint:
        resume_checkpoint = Path(os.path.join(args.save_dir, "runs", args.save_name, 'checkpoints', 'checkpoint-{}.pt'.format(str(args.resume_checkpoint).zfill(5))))
        if not os.path.exists(resume_checkpoint):
            print('Best checkpoint Not Found!')
            return None
        model, optimizer, scheduler, start_epoch, valid_recon_best, log_dir = resume(
            resume_checkpoint, model, optimizer, scheduler)
        print('Resumed from: ' + str(resume_checkpoint))
        experiment_dir = Path(os.path.join(args.save_dir, "runs", args.save_name))
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
            model, optimizer, scheduler, start_epoch, valid_recon_best, log_dir = resume(
                str(resume_checkpoint), model, optimizer, scheduler)
            print('Resumed from: ' + str(resume_checkpoint))
        else:
            start_epoch = 1
            experiment_dir.mkdir(exist_ok=True)

    '''CREATE DIR'''

    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
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
    # if args.ae_data == 'hs':
    #     from dataset.Dataset_Loader import get_dataloader
    #     train_shape_loader, train_sketch_loader, val_shape_loader, val_sketch_loader, _, _ = get_dataloader(
    #         args)
    # elif args.ae_data == 'network':
    from dataset.Dataset_Loader import get_dataloader_aug
    train_shape_loader, train_sketch_loader, val_shape_loader, val_sketch_loader,  test_shape_loader, test_sketch_loader = get_dataloader_aug(args)

    if args.train:
        best_epoch = -1
        '''TRANING'''
        logger.info('Start training...')
        for epoch in range(start_epoch, args.epochs+1):
            log_string('Epoch (%d/%s):' % (epoch, args.epochs), logger)

            # plot learning rate
            if writer is not None:
                writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], epoch)

            train(train_sketch_loader, train_shape_loader, model, optimizer, writer, epoch, args, logger)
            scheduler.step()

            if epoch % args.save_freq == 0:
                log_string("Test:", logger)
                cur_metric = validate(val_sketch_loader, val_shape_loader, model)
                is_best = cur_metric < valid_recon_best
                if is_best:
                    best_epoch = epoch
                valid_recon_best = min(cur_metric, valid_recon_best)
                writer.add_scalar('val/recon', cur_metric, epoch) # mAP_feat_norm
                log_string('\n * Finished epoch {:3d}  top1: {:.4f}  best: {:.4f} @epoch {}\n'.
                           format(epoch, cur_metric, valid_recon_best, best_epoch), logger)
                if not args.debug:
                    checkpoint_path = os.path.join(str(checkpoints_dir), 'checkpoint-{}.pt'.format(str(epoch).zfill(5)))
                    save(model, optimizer, epoch + 1, scheduler, valid_recon_best, log_dir, checkpoint_path)
                    logger.info('Save epoch {} checkpoint to: {}'.format(epoch, checkpoint_path))

        logger.info('End of training...')
        writer.export_scalars_to_json(log_dir.joinpath("all_scalars.json"))
        writer.close()

        log_string('Best metric {}'.format(valid_recon_best), logger)
    else:
        log_string("Evaluate on test set:", logger)
        cur_metric = validate(val_sketch_loader, val_shape_loader, model, save=True, save_dir=log_dir)
        log_string('metric on test set: {}'.format(cur_metric), logger)


    return experiment_dir

def train(sketch_dataloader, shape_dataloader, model, optimizer, writer, epoch, args, logger):
    model = model.train()
    start_time = time.time()

    for bidx, (sketches, shapes) in enumerate(zip(sketch_dataloader, shape_dataloader)):
        step = bidx + len(sketch_dataloader) * (epoch - 1)
        points = torch.cat([sketches[0], shapes[0]], axis = 1).data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        if args.random_rotate:
            points, _, _ = apply_random_rotation(points, rot_axis=1)
        n_points = sketches[0].shape[1]
        points = torch.Tensor(points).cuda()
        sketches = points[:, :n_points, :]
        shapes = points[:, n_points:, :]
        if args.ae_type == 'ae':
            rec_loss = model.ae_forward(sketches, sketches, optimizer)
        else:
            rec_loss = model.ae_forward(shapes, sketches, optimizer)

        if step % args.log_freq == 0:
            duration = time.time() - start_time
            start_time = time.time()
            if writer is not None:
                writer.add_scalar('train/recon_loss', rec_loss, step)
            log_string(
                "Epoch %d Batch [%2d/%2d] Time [%3.2fs]  loss %2.5f"
                % (epoch, bidx, len(sketch_dataloader), duration, rec_loss), logger)

def validate(sketch_dataloader, shape_dataloader, model, save=False, save_dir=''):
    model = model.eval()
    rec_loss = misc.AverageMeter()
    sketches_list = []
    recons_list = []
    with torch.no_grad():
        for bidx, (sketches, shapes) in enumerate(zip(sketch_dataloader, shape_dataloader)):
            sketches = torch.Tensor(sketches[0]).cuda()
            shapes = torch.Tensor(shapes[0]).cuda()
            if args.ae_type == 'ae':
                recon = model.ae_recon(sketches)
            else:
                recon = model.ae_recon(shapes)
            loss = model.loss(sketches, recon, bidirectional=True)
            rec_loss.update(loss)
            sketches_list.append(sketches)
            recons_list.append(recon)


    if save:
        sketches_list = torch.cat(sketches_list, 0).data.cpu().numpy()
        recons_list = torch.cat(recons_list, 0).data.cpu().numpy()

        print(str(save_dir))
        np.save(save_dir.joinpath('sketches.npy'), sketches_list)
        # np.save(save_dir.joinpath('shapes.npy'), shapes.data.cpu().numpy())
        np.save(save_dir.joinpath('recon.npy'),  recons_list.data)
    return rec_loss.avg.cpu()

if __name__ == '__main__':

    args = get_args()
    if args.windows:
        from args import get_parser
        parser = get_parser()
        args = parser.parse_args(args=[
            # '--debug',
                                        '--recon',
                                       '--train',
                                       '--save_name', 'ae_pn2_foldnet_sketch_shape',
                                        '--encoder', 'pn2',
            '--decoder', 'foldnet',
            '--ae_type', 'ed',
            '--ae_input', 'sketch',
            '--ae_output', 'shape',
            '--aug_list_file', 'aug/modelnet_702.txt',
            '--aug_dir', 'synthetic_sketch_1.0',
            '--lr', '1e-4',
            '--data_dir', r'C:\Users\ll00931\Documents\chair_1005\all_networks',
                                       # '--resume_checkpoint', '100',
                                       '--epoch', '100', \
                                       '--batch_size', '8', \
                                        '--save_freq', '10',\
                                       '--log_freq', '1',\
                                       '--save_dir', r'C:\Users\ll00931\PycharmProjects\FineGrained_3DSketch'
                                       ])

    experiment_dir = main(args)
    # os.system('sh run_eval_all.sh 5 %s' % experiment_dir)
