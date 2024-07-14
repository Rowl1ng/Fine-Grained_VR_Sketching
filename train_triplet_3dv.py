import numpy as np
import os
import json
import torch
import logging
from pathlib import Path
from args import get_args

import tools.provider as provider
from tensorboardX import SummaryWriter
import time
from tools.evaluation import compute_distance, compute_acc_at_k
from dataset.Dataset_Loader import get_dataloader

from utils.utils import apply_random_rotation, apply_random_scale_xyz, save, resume
from tools.misc import get_latest_ckpt
from torch.backends import cudnn

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    if args.resume_checkpoint:
        if args.train:
            # finetune
            resume_checkpoint = args.resume_checkpoint
        else:
            # eval
            resume_checkpoint = os.path.join(args.save_dir, "runs", args.save_name, 'checkpoints',  'checkpoint-{}.pt'.format(str(args.resume_checkpoint).zfill(5)))
            if not os.path.exists(resume_checkpoint):
                resume_checkpoint = os.path.join(args.save_dir, "runs", args.save_name, 'checkpoints',  'checkpoint_best.pt')
            resume_checkpoint = Path(resume_checkpoint)
        if not os.path.exists(resume_checkpoint):
            print('Resume checkpoint Not Found!')
            return None
        model, optimizer, scheduler, start_epoch, valid_acc_best, log_dir = resume(
            resume_checkpoint, model, optimizer, scheduler)
        print('Resumed from: ' + str(resume_checkpoint) + 'with start epoch:'+ str(start_epoch))
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
        resume_checkpoint = os.path.join(str(checkpoints_dir), 'checkpoint_latest.pt') #get_latest_ckpt(checkpoints_dir)
        if os.path.exists(resume_checkpoint):
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
    if args.recon:
        train_shape_loader, train_sketch_loader, train_network_loader, val_shape_loader, val_sketch_loader, test_shape_loader, test_sketch_loader = get_dataloader(args)
    else:
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
            if args.recon:
                train_rec(train_sketch_loader, train_shape_loader, train_network_loader, model, optimizer, writer, epoch, args, logger)
            else:
                train(train_sketch_loader, train_shape_loader, model, optimizer, writer, epoch, args, logger)
            scheduler.step()

            if epoch % args.save_freq == 0:
                log_string("Test:", logger)
                cur_metric = validate(args, logger, val_sketch_loader, val_shape_loader, model)
                top1 = cur_metric[0]
                is_best = top1 > valid_acc_best
                if is_best:
                    best_epoch = epoch
                    checkpoint_path = os.path.join(str(checkpoints_dir), 'checkpoint_best.pt')
                    save(model, optimizer, epoch + 1, scheduler, valid_acc_best, log_dir, checkpoint_path)

                valid_acc_best = max(top1, valid_acc_best)

                writer.add_scalar('val/top-1', top1, epoch)
                writer.add_scalar('val/top-5', cur_metric[1], epoch)
                writer.add_scalar('val/top-10', cur_metric[2], epoch)

                log_string('\n * Finished epoch {:3d}  top1: {:.4f}  best: {:.4f} @epoch {}\n'.
                           format(epoch, top1, valid_acc_best, best_epoch), logger)

                checkpoint_path = os.path.join(str(checkpoints_dir), 'checkpoint_latest.pt')
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
        cur_metric = validate(args, logger, test_sketch_loader, test_shape_loader, model, save=True, save_dir=log_dir, data_dir=args.data_dir, resume_checkpoint=args.resume_checkpoint)
        # get_recon(test_sketch_loader, model, save_dir=log_dir)

    return experiment_dir

def train(sketch_dataloader, shape_dataloader, model, optimizer, writer, epoch, args, logger):
    model = model.train()
    start_time = time.time()

    for bidx, (sketches, shapes) in enumerate(zip(sketch_dataloader, shape_dataloader)):
        step = bidx + len(sketch_dataloader) * (epoch - 1)
        if args.use_aug_loss:
            points = torch.cat([sketches[0], sketches[0], shapes[0]]).data.numpy()
        else:
            points = torch.cat([sketches[0], shapes[0]]).data.numpy()
        shape_ids = sketches[1]
        points = provider.random_point_dropout(points)
        points = provider.random_scale_point_cloud(points)
        points = provider.shift_point_cloud(points)

        points = torch.Tensor(points)
        inputs = points.cuda()
        minibatch = shapes[0].shape[0]
        if args.transform_anchor:
            transform_index = -minibatch
        else:
            transform_index = inputs.shape[0]
        if args.random_rotate:
            inputs[:transform_index, :, :], _, _ = apply_random_rotation(inputs[:transform_index, :, :], rotate_mode=args.rotate_mode)
        if args.random_scale:
            inputs[:transform_index, :, :], _ = apply_random_scale_xyz(inputs[:transform_index, :, :])

        out = model(inputs, shape_ids, optimizer)

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
        points = provider.random_scale_point_cloud(points)
        points = provider.shift_point_cloud(points)
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

def get_recon(sketch_dataloader, model, save_dir):
    model = model.eval()
    sketches_list = []
    recons_list = []

    with torch.no_grad():
        for i, data in enumerate(sketch_dataloader):
            sketches = data[0].cuda()
            recon = model.ae_recon(sketches)
            sketches_list.append(sketches)
            recons_list.append(recon)

    sketches_list = torch.cat(sketches_list, 0).data.cpu().numpy()
    recons_list = torch.cat(recons_list, 0).data.cpu().numpy()
    print(str(save_dir))
    np.save(save_dir.joinpath('sketches.npy'), sketches_list)
    # np.save(save_dir.joinpath('shapes.npy'), shapes.data.cpu().numpy())
    np.save(save_dir.joinpath('recon.npy'), recons_list.data)


def validate(args, logger, sketch_dataloader, shape_dataloader, model, save=False, save_dir='', data_dir='', resume_checkpoint=''):
    sketch_features = []
    shape_features = []
    model = model.eval()
    start_time = time.time()

    with torch.no_grad():
        for i, data in enumerate(sketch_dataloader):
            sketch_points = data[0].cuda()
            sketch_z = model.extract_feature(sketch_points, shape=False)
            sketch_features.append(sketch_z.data.cpu())

        for i, data in enumerate(shape_dataloader):
            shape_points = data[0].cuda()
            shape_z = model.extract_feature(shape_points, shape=True)
            shape_features.append(shape_z.data.cpu())

    inference_duration = time.time() - start_time
    start_time = time.time()

    shape_features = torch.cat(shape_features, 0).numpy()
    sketch_features = torch.cat(sketch_features, 0).numpy()
    dist = compute_distance(sketch_features.copy(), shape_features.copy(), l2=True)
    pair_sort = np.argsort(dist)
    sketch_num = pair_sort.shape[0]
    all_list = [i for i in range(0, sketch_num)]
    acc_at_k = compute_acc_at_k(pair_sort, all_list)
    eval_duration = time.time() - start_time

    log_string(
        "Inference Time [%3.2fs]  Eval Time [%3.2fs]"
        % (inference_duration, eval_duration), logger)

    for acc_z_i, k in zip(acc_at_k, [1, 5, 10]):
        log_string(' * Acc@{:d} z acc {:.4f}'.format(k, acc_z_i), logger)

    if save:
        # np.save(os.path.join(save_dir, 'shape_feat_{}.npy'.format(batch_size)), shape_features)
        # np.save(os.path.join(save_dir, 'sketch_feat_{}.npy'.format(batch_size)), sketch_features)
        log_string(
            "Resumed from {} epoch.".format(resume_checkpoint), logger)

        np.save(os.path.join(save_dir, 'd_feat_{}.npy'.format(args.test_sketch_dir)), dist)

        group_A = []
        group_B = []
        test_list = os.path.join(data_dir, 'list/hs/test_45participants.txt')
        name_list = [line.rstrip() for line in open(test_list)]
        for idx, item in enumerate(name_list):
            participant_id = item.split('_')[3]
            if participant_id in ['5', '36', '20', '38']:
                group_B.append(idx)
            else:
                group_A.append(idx)
        participants5_list = [i for i in range(sketch_num - 50, sketch_num)]
        list_dict = {
            'par5': participants5_list,
            'group_A': group_A,
            'group_B': group_B
        }
        # all_list = [i for i in range(0, sketch_num)]
        for item in list_dict.keys():
            log_string(item, logger)
            acc_at_k = compute_acc_at_k(pair_sort, list_dict[item])
            for acc_z_i, k in zip(acc_at_k, [1, 5, 10]):
                log_string(' * Acc@{:d} z acc {:.4f}'.format(k, acc_z_i), logger)

    return acc_at_k



if __name__ == '__main__':
    args = get_args()
    if args.windows:
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

        from args import get_parser
        parser = get_parser()
        args = parser.parse_args(args=[
            # '--debug',
            '--train',
            # '--type', 'hetero',
            # '--loss', 'rl',
            '--lr', '1e-2',
            # '--percentage', '0.6',
            # '--symmetric',
            # '--stn',
            # '--w2', '1',
            # '--recon',
            # '--tao', '1.0',
            # '--encoder', 'dgcnn',
            # '--random_rotate',
            # '--rotate_mode', '4_rotations',
            # '--transform_anchor',
            # '--encoder_data', 'ss',
            # '--sketch_dir', 'aligned_sketch',
            # '--aug_list_file', 'hs/train.txt',
            # '--aug_dir', 'ss_1.0',
            # '--random_scale',
            # '--use_aug_loss',
            # '--use_aug',
            # '--aug_dir', 'ss_1.0',
            '--train_list_file', "hs/train_size/4_20_0.txt",
            # '--encoder_data', 'hs',
            '--save_name', 'default',
            '--data_dir', r'C:\Users\ll00931\OneDrive - University of Surrey\Documents\chair_1005\all_networks',
            # '--resume_checkpoint', '3',
            '--epoch', '10', \
            '--batch_size', '12', \
            '--save_freq', '1',\
            '--log_freq', '1',\
            '--save_dir', r'C:\Users\ll00931\PycharmProjects\FineGrained_3DSketch'
           ])
    # print(args.recon)
    # quit()
    experiment_dir = main(args)
    # os.system('sh run_eval_all.sh 5 %s' % experiment_dir)
