import torch.utils.data
import torch
from dataset.TripletSampler import BatchSampler
from dataset.PointCloudLoader import PointCloudDataLoader, PointCloudDataLoader_ss, PointCloudDataLoader_aug

def get_2d_loader_multiple(args):
    minibatch = int(args.batch_size//2)
    test_batch_size = args.batch_size

    from dataset.MultiViewLoader import MultiviewImgDataset
    train_shape_dataset = MultiviewImgDataset(set='train', list_file=args.list_file, data_dir=args.data_dir, test_mode=False, data_type='shape')
    val_shape_dataset = MultiviewImgDataset(set='val', list_file=args.list_file, data_dir=args.data_dir,
                                            data_type='shape')
    test_shape_dataset = MultiviewImgDataset(set='test', list_file=args.list_file, data_dir=args.data_dir,
                                             data_type='shape')

    train_sketch_dataset = MultiviewImgDataset(set='train', list_file=args.list_file, data_dir=args.data_dir, test_mode=False, \
                                               data_type=args.sketch_data_type, num_views=3)

    seed = 0
    shape_train_batch_sampler = BatchSampler(train_shape_dataset, minibatch, seed)
    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_sampler=shape_train_batch_sampler)
    sketch_train_batch_sampler = BatchSampler(train_sketch_dataset, minibatch, seed)
    train_sketch_loader = torch.utils.data.DataLoader(train_sketch_dataset, batch_sampler=sketch_train_batch_sampler)

    val_sketch_dataset = MultiviewImgDataset(set='val', list_file=args.list_file, data_dir=args.data_dir, data_type=args.sketch_data_type, num_views=3)
    val_shape_loader = torch.utils.data.DataLoader(val_shape_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    print('test_batch_size: ',test_batch_size)
    val_sketch_loader = torch.utils.data.DataLoader(val_sketch_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    test_sketch_dataset = MultiviewImgDataset(set='test', list_file=args.list_file, data_dir=args.data_dir, data_type=args.sketch_data_type, num_views=3)

    test_shape_loader = torch.utils.data.DataLoader(test_shape_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    test_sketch_loader = torch.utils.data.DataLoader(test_sketch_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    return train_shape_loader, train_sketch_loader, val_shape_loader, val_sketch_loader,  test_shape_loader, test_sketch_loader


def get_2d_loader_view(args):
    minibatch = int(args.batch_size//2)
    test_batch_size = args.batch_size

    from dataset.MultiViewLoader import MultiviewImgDataset
    train_shape_dataset = MultiviewImgDataset(set='train', list_file=args.list_file, data_dir=args.data_dir, test_mode=False, \
                                              data_type=args.shape_view_type, num_views=1, view=args.shape_view)
    val_shape_dataset = MultiviewImgDataset(set='val', list_file=args.list_file, data_dir=args.data_dir,
                                            data_type=args.shape_view_type, num_views=1, view=args.shape_view)
    test_shape_dataset = MultiviewImgDataset(set='test', list_file=args.list_file, data_dir=args.data_dir,
                                             data_type=args.shape_view_type, num_views=1, view=args.shape_view)

    train_sketch_dataset = MultiviewImgDataset(set='train', list_file=args.list_file, data_dir=args.data_dir, test_mode=False, data_type='sketch', view=args.view)

    seed = 0
    shape_train_batch_sampler = BatchSampler(train_shape_dataset, minibatch, seed)
    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_sampler=shape_train_batch_sampler)
    sketch_train_batch_sampler = BatchSampler(train_sketch_dataset, minibatch, seed)
    train_sketch_loader = torch.utils.data.DataLoader(train_sketch_dataset, batch_sampler=sketch_train_batch_sampler)

    val_sketch_dataset = MultiviewImgDataset(set='val', list_file=args.list_file, data_dir=args.data_dir, data_type='sketch', view=args.view)
    val_shape_loader = torch.utils.data.DataLoader(val_shape_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    val_sketch_loader = torch.utils.data.DataLoader(val_sketch_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    test_sketch_dataset = MultiviewImgDataset(set='test', list_file=args.list_file, data_dir=args.data_dir, data_type='sketch', view=args.view)

    test_shape_loader = torch.utils.data.DataLoader(test_shape_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    test_sketch_loader = torch.utils.data.DataLoader(test_sketch_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    return train_shape_loader, train_sketch_loader, val_shape_loader, val_sketch_loader,  test_shape_loader, test_sketch_loader

def get_2d_loader(args):
    minibatch = int(args.batch_size//2)
    test_batch_size = args.batch_size

    from dataset.MultiViewLoader import MultiviewImgDataset
    if args.use_pn2:
        train_list_file = args.list_file
        train_shape_dataset = PointCloudDataLoader(npoints=args.shape_points,list_file=train_list_file,
                                                   uniform=True,
                                                   data_dir=args.data_dir,
                                                   split='train', data_type='shape',
                                                   debug=args.debug)
        val_shape_dataset = PointCloudDataLoader(npoints=args.shape_points, list_file=args.list_file,
                                                 uniform=True,
                                                 data_dir=args.data_dir,
                                                 split='val', data_type='shape',
                                                 debug=args.debug)
        test_shape_dataset = PointCloudDataLoader(npoints=args.shape_points, list_file=args.list_file,
                                                  uniform=True,
                                                  data_dir=args.data_dir,
                                                  split='test', data_type='shape',
                                                  debug=args.debug)
    else:
        train_shape_dataset = MultiviewImgDataset(set='train', list_file=args.list_file, data_dir=args.data_dir, test_mode=False, data_type='shape')
        val_shape_dataset = MultiviewImgDataset(set='val', list_file=args.list_file, data_dir=args.data_dir,
                                                data_type='shape')
        test_shape_dataset = MultiviewImgDataset(set='test', list_file=args.list_file, data_dir=args.data_dir,
                                                 data_type='shape')

    train_sketch_dataset = MultiviewImgDataset(set='train', list_file=args.list_file, data_dir=args.data_dir, test_mode=False, data_type='sketch', view=args.view)

    seed = 0
    shape_train_batch_sampler = BatchSampler(train_shape_dataset, minibatch, seed)
    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_sampler=shape_train_batch_sampler)
    sketch_train_batch_sampler = BatchSampler(train_sketch_dataset, minibatch, seed)
    train_sketch_loader = torch.utils.data.DataLoader(train_sketch_dataset, batch_sampler=sketch_train_batch_sampler)

    val_sketch_dataset = MultiviewImgDataset(set='val', list_file=args.list_file, data_dir=args.data_dir, data_type='sketch', view=args.view)
    val_shape_loader = torch.utils.data.DataLoader(val_shape_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    val_sketch_loader = torch.utils.data.DataLoader(val_sketch_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    test_sketch_dataset = MultiviewImgDataset(set='test', list_file=args.list_file, data_dir=args.data_dir, data_type='sketch', view=args.view)

    test_shape_loader = torch.utils.data.DataLoader(test_shape_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    test_sketch_loader = torch.utils.data.DataLoader(test_sketch_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    return train_shape_loader, train_sketch_loader, val_shape_loader, val_sketch_loader,  test_shape_loader, test_sketch_loader

def get_dataloader(args):
    test_batch_size = 16 #args.batch_size
    minibatch = int(args.batch_size//2)
    if args.use_aug:
        train_shape_dataset = PointCloudDataLoader_aug(npoints=args.shape_points,list_file=[args.list_file, args.aug_list_file],
                                                   uniform=True,
                                                   data_dir=args.data_dir,
                                                   aug_dir=args.aug_dir,
                                                   data_type='shape', percentage=args.percentage,
                                                   debug=args.debug, seed=args.seed)

        train_sketch_dataset = PointCloudDataLoader_aug(npoints=args.sketch_points,list_file=[args.list_file, args.aug_list_file],
                                                    uniform=True,
                                                    data_dir=args.data_dir,
                                                    aug_dir=args.aug_dir,
                                                    data_type=args.sketch_dir, percentage=args.percentage,
                                                    debug=args.debug, seed=args.seed)
        if args.recon:
            train_network_dataset = PointCloudDataLoader_aug(npoints=args.sketch_points,
                                                            list_file=[args.list_file, args.aug_list_file],
                                                            uniform=True,
                                                            data_dir=args.data_dir,
                                                            aug_dir='network',
                                                            data_type='network',
                                                            debug=args.debug)
    else:
        if args.encoder_data == 'hs':
            train_list_file = args.list_file
            train_sketch_dir = args.sketch_dir
            if args.train_list_file != '':
                train_list_file = args.train_list_file
        else:
            train_list_file = args.aug_list_file
            train_sketch_dir = args.aug_dir

        train_shape_dataset = PointCloudDataLoader(npoints=args.shape_points,list_file=train_list_file,
                                                   uniform=True,
                                                   data_dir=args.data_dir,
                                                   split='train', data_type='shape', percentage=args.percentage,
                                                   debug=args.debug, seed=args.seed)

        train_sketch_dataset = PointCloudDataLoader(npoints=args.sketch_points,list_file=train_list_file,
                                                    uniform=True,
                                                   data_dir=args.data_dir,
                                                   split='train', data_type=train_sketch_dir, percentage=args.percentage,
                                                    debug=args.debug, seed=args.seed)

    seed = 0
    shape_train_batch_sampler = BatchSampler(train_shape_dataset, minibatch, seed)
    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_sampler=shape_train_batch_sampler)

    sketch_train_batch_sampler = BatchSampler(train_sketch_dataset, minibatch, seed)
    train_sketch_loader = torch.utils.data.DataLoader(train_sketch_dataset, batch_sampler=sketch_train_batch_sampler)

    if args.recon:
        network_train_batch_sampler = BatchSampler(train_network_dataset, minibatch, seed)
        train_network_loader = torch.utils.data.DataLoader(train_network_dataset, batch_sampler=network_train_batch_sampler)

    val_shape_dataset = PointCloudDataLoader(npoints=args.shape_points, list_file=args.list_file,
                                              uniform=True,
                                               data_dir=args.data_dir,
                                               split='val', data_type='shape',
                                             debug=args.debug)

    val_sketch_dataset = PointCloudDataLoader(npoints=args.sketch_points, list_file=args.list_file,
                                               uniform=True,
                                               data_dir=args.data_dir,
                                               split='val', data_type=args.sketch_dir,
                                             debug=args.debug)


    val_shape_loader = torch.utils.data.DataLoader(val_shape_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    val_sketch_loader = torch.utils.data.DataLoader(val_sketch_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    test_shape_dataset = PointCloudDataLoader(npoints=args.shape_points, list_file=args.list_file,
                                              uniform=True,
                                               data_dir=args.data_dir,
                                               split='test', data_type='shape',
                                             debug=args.debug)

    test_sketch_dataset = PointCloudDataLoader(npoints=args.sketch_points, list_file=args.test_list_file,
                                               uniform=True,
                                               data_dir=args.data_dir,
                                               split='test', data_type=args.test_sketch_dir,
                                             debug=args.debug)


    test_shape_loader = torch.utils.data.DataLoader(test_shape_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    test_sketch_loader = torch.utils.data.DataLoader(test_sketch_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    if args.recon:
        return train_shape_loader, train_sketch_loader, train_network_loader, val_shape_loader, val_sketch_loader, test_shape_loader, test_sketch_loader

    return train_shape_loader, train_sketch_loader, val_shape_loader, val_sketch_loader,  test_shape_loader, test_sketch_loader

def get_dataloader_bi_level(args, seed):
    list_file = args.ss_list_file
    data_dir = args.ss_dir
    train_shape_dataset = PointCloudDataLoader_ss(npoints=args.shape_points, list_file=list_file,
                                                  uniform=True,
                                                  data_dir=data_dir,
                                                  split='train', data_type='shape',
                                                  debug=args.debug)

    train_sketch_dataset = PointCloudDataLoader_ss(npoints=args.sketch_points, list_file=list_file,
                                                   uniform=True,
                                                   data_dir=data_dir,
                                                   split='train', data_type='sketch',
                                                   debug=args.debug)
    shape_train_batch_sampler = BatchSampler(train_shape_dataset, int(args.batch_size//2), seed)
    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_sampler=shape_train_batch_sampler)

    sketch_train_batch_sampler = BatchSampler(train_sketch_dataset, int(args.batch_size//2), seed)
    train_sketch_loader = torch.utils.data.DataLoader(train_sketch_dataset, batch_sampler=sketch_train_batch_sampler)

    return train_shape_loader, train_sketch_loader

def get_dataloader_shape(args):
    test_batch_size = args.batch_size

    train_shape_dataset = PointCloudDataLoader(npoints=args.shape_points, list_file=args.list_file,
                                               uniform=True,
                                               data_dir=args.data_dir,
                                               data_type='shape',
                                               debug=args.debug)

    train_sketch_dataset = PointCloudDataLoader(npoints=args.sketch_points, list_file=args.list_file,
                                                uniform=False,
                                               data_dir=args.data_dir,
                                                data_type='shape',
                                                debug=args.debug)

    seed = 0
    shape_train_batch_sampler = BatchSampler(train_shape_dataset, int(args.batch_size//2), seed)
    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_sampler=shape_train_batch_sampler)

    sketch_train_batch_sampler = BatchSampler(train_sketch_dataset, int(args.batch_size//2), seed)
    train_sketch_loader = torch.utils.data.DataLoader(train_sketch_dataset, batch_sampler=sketch_train_batch_sampler)

    val_shape_dataset = PointCloudDataLoader(npoints=args.shape_points,list_file=args.list_file,
                                              uniform=True,
                                               data_dir=args.data_dir,
                                               split='test', data_type='shape',
                                             debug=args.debug, seed=0)

    val_sketch_dataset = PointCloudDataLoader(npoints=args.sketch_points,list_file=args.list_file,
                                               uniform=False,
                                               data_dir=args.data_dir,
                                               split='test', data_type='shape',
                                             debug=args.debug, seed=1)


    val_shape_loader = torch.utils.data.DataLoader(val_shape_dataset, batch_size= test_batch_size, shuffle=False, num_workers=4)
    val_sketch_loader = torch.utils.data.DataLoader(val_sketch_dataset, batch_size= test_batch_size, shuffle=False, num_workers=4)

    return train_shape_loader, train_sketch_loader, val_shape_loader, val_sketch_loader

def get_dataloader_chair(args):
    train_shape_dataset = PointCloudDataLoader(npoints=args.shape_points, list_file=args.aug_list_file,
                                               uniform=True,
                                               data_dir=args.data_dir,
                                               split='train', data_type='shape',
                                               debug=args.debug)
    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_size=int(args.batch_size/2), shuffle=True, num_workers=4, drop_last=True)
    return train_shape_loader

def get_dataloader_AE_shape(args):
    list_file = args.ae_list_file
    shape_dir = args.ae_shape_dir

    train_shape_dataset = PointCloudDataLoader_AE(npoints=args.shape_points,list_file=list_file,
                                               uniform=True,
                                               data_dir=shape_dir,
                                               split='train', data_type='shape',
                                               debug=args.debug)

    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_size=int(args.batch_size/2), shuffle=True, num_workers=4, drop_last=True)
    return train_shape_loader

def get_dataloader_network(args):
    test_batch_size = args.batch_size
    list_file = args.network_list_file
    data_dir = args.data_dir
    train_shape_dataset = PointCloudDataLoader(npoints=args.shape_points,list_file=list_file,
                                               uniform=True,
                                               data_dir=data_dir,
                                               split='train', data_type=args.ae_input,
                                               debug=args.debug)

    train_sketch_dataset = PointCloudDataLoader(npoints=args.sketch_points,list_file=list_file,
                                                uniform=True,
                                               data_dir=data_dir,
                                               split='train', data_type=args.ae_output,
                                                debug=args.debug)

    seed = 0
    shape_train_batch_sampler = BatchSampler(train_shape_dataset, int(args.batch_size//2), seed)
    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_sampler=shape_train_batch_sampler)

    sketch_train_batch_sampler = BatchSampler(train_sketch_dataset, int(args.batch_size//2), seed)
    train_sketch_loader = torch.utils.data.DataLoader(train_sketch_dataset, batch_sampler=sketch_train_batch_sampler)

    val_shape_dataset = PointCloudDataLoader(npoints=args.shape_points,list_file=list_file,
                                              uniform=True,
                                               data_dir=data_dir,
                                               split='test', data_type=args.ae_input,
                                             debug=args.debug)

    val_sketch_dataset = PointCloudDataLoader(npoints=args.sketch_points,list_file=list_file,
                                               uniform=True,
                                               data_dir=data_dir,
                                               split='test', data_type=args.ae_output,
                                             debug=args.debug)


    val_shape_loader = torch.utils.data.DataLoader(val_shape_dataset, batch_size= test_batch_size, shuffle=False, num_workers=4)
    val_sketch_loader = torch.utils.data.DataLoader(val_sketch_dataset, batch_size= test_batch_size, shuffle=False, num_workers=4)


    return train_shape_loader, train_sketch_loader, val_shape_loader, val_sketch_loader
