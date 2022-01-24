import argparse
import os

def add_args(parser):
    parser.add_argument('--debug', action='store_true',
                        help='Whether debug locally.')
    parser.add_argument('--windows', action='store_true',
                        help='Whether debug locally.')
    parser.add_argument('--train', action='store_true',
                        help='Whether debug locally.')

    parser.add_argument('-gradient_clip', type=float, default=0.05)  # previous i set it to be 0.01

    # 3DV baseline
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--feat_dim', type=int, default=512)

    parser.add_argument('--recon', action='store_true',
                        help='Whether to use recon branch.')
    parser.add_argument('--stn', action='store_true',
                        help='Whether to use recon branch.')

    parser.add_argument('--w1', type=float, default=1,
                        help='Learning rate for the Adam optimizer.')
    parser.add_argument('--w2', type=float, default=1,
                        help='Learning rate for the Adam optimizer.')
    # parser.add_argument('--use_pointnet_encoder', action='store_true',
    #                     help='Whether to use PointNet++ as encoder.')
    parser.add_argument('--encoder', type=str, default='pn2', choices={'pn2', 'pn', 'dgcnn', 'foldnet'})
    parser.add_argument('--decoder', type=str, default='mlp', choices={'mlp', 'foldnet'})
    parser.add_argument('--use_stn', action='store_true',
                        help='Whether to use a deterministic encoder.')
    parser.add_argument('--type', type=str, default='siam', choices={'siam', 'hetero'})
    parser.add_argument('--backbone', type=str, default='vgg11', choices={'vgg11', 'resnet50'})


    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    # Toy Model
    parser.add_argument('--sketch_points', type=int, default=1024)
    parser.add_argument('--shape_points', type=int, default=1024)
    parser.add_argument('--uniform', action='store_true', default=True,
                        help='Whether to use a deterministic encoder.')

    # Flow
    parser.add_argument('--use_deterministic_encoder', action='store_true',
                        help='Whether to use a deterministic encoder.')
    parser.add_argument('--use_latent_flow', action='store_true',
                        help='Whether to use a deterministic encoder.')
    parser.add_argument('--use_z', action='store_true',
                        help='Whether to use a deterministic encoder.')
    parser.add_argument('--h_dims_AF', type=str, default='128-128-128')
    parser.add_argument('--n_flow_AF', type=int, default=12)
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam.')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam.')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--step_size', type=int, default=20,
                        help='Step size for scheduler.')
    parser.add_argument('--gamma', type=int, default=0.7,
                        help='Learning rate decay ratio.')
    parser.add_argument('--stop_scheduler', type=int, default=15000,
                        help='When to freeze leraning rate.')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')
    # input noise
    parser.add_argument('--std_min', type=float, default=0.0)
    parser.add_argument('--std_max', type=float, default=0.075)
    parser.add_argument('--test_std_n', type=float, default=0.0)
    parser.add_argument('--test_std_z', type=float, default=1.0)
    parser.add_argument('--std_scale', type=float, default=2)
    # bi-level
    parser.add_argument('--num_neumann_terms', type=int, default=1, help='The maximum number of neumann terms to use')
    parser.add_argument('--use_cg', action='store_true', default=False, help='If we should use CG')
    parser.add_argument('--load_finetune_checkpoint', default='', help='Choose a model')
    parser.add_argument('--load_baseline_checkpoint', default='', help='If we should use CG')
    parser.add_argument('--finetune_margin', type=float, default=3., help='The maximum number of neumann terms to use')
    parser.add_argument('--encoder_data', type=str, default='hs', choices={'hs', 'ss', 'cn'})

    parser.add_argument('--ae_type', type=str, default='ae', choices={'ae', 'ed'})
    parser.add_argument('--ae_input', type=str, default='network', choices={'shape', 'sketch', 'ss', 'network'})
    parser.add_argument('--ae_output', type=str, default='network', choices={'network', 'sketch', 'shape'})
    parser.add_argument('--shape', type=str, default='gaussian', choices={'gaussian', 'sphere', 'plane'})

    parser.add_argument('--freeze_epoch', type=int, default=2, help='The maximum number of neumann terms to use')
    parser.add_argument('--siamese_epoch', type=int, default=50, help='The maximum number of neumann terms to use')

    # training configuration
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (of datasets) for training')
    parser.add_argument('--optimizer', type=str, default='sgd', choices={'adam', 'sgd'})
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate for the Adam optimizer.')
    parser.add_argument('--epochs', type=int, default=15000,
                        help='Number of epochs for training (default: 12000)')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to the checkpoint to be loaded for training.')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                    help='Path to the checkpoint to be loaded for genration and test.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for initializing training. ')

    # triplet configuration
    parser.add_argument('--use_triplet', action='store_true',
                        help='Whether to use triplet training.')
    parser.add_argument('--symmetric', action='store_true',
                        help='Whether to use symmetric triplet loss.')
    parser.add_argument('--margin', type=float, default=0.3,
                        help='margin for triplet loss.')
    parser.add_argument('--hard_negative_mining', action='store_true',
                        help='Whether to use triplet training.')
    parser.add_argument('--beta', type=int, default=1,
                        help='margin for triplet loss.')
    parser.add_argument('--sketch_anchor', action='store_true',
                        help='Whether to use triplet training.')
    parser.add_argument('--use_softplus', action='store_true',
                        help='Whether to use triplet training.')
    parser.add_argument('--flooding_b', type=float, default=0.01,
                        help='margin for triplet loss.')
    parser.add_argument('--loss', type=str, default='tl', choices={'tl', 'cl_1', 'cl_2', 'cl_3', 'rl'})
    parser.add_argument('--tao', type=float, default=0.1,
                        help='margin for triplet loss.')
    parser.add_argument('--m', type=float, default=0.5244,
                        help='margin for triplet loss.')

    # data options
    parser.add_argument('--list_file', type=str, default="hs/{}.txt",
                        help="Path to the namelist file")
    parser.add_argument('--train_list_file', type=str, default="",
                        help="Path to the namelist file")

    parser.add_argument('--aug_list_file', type=str, default="aug/shapenet_702.txt",
                        help="Path to the namelist file")
    parser.add_argument('--aug_dir', type=str, default="ae_sketch",
                        help="Path to the namelist file")
    parser.add_argument('--data_dir', type=str, default="/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch",
                        help="Path to the training data")
    parser.add_argument('--use_aug', action='store_true',
                        help='Whether to randomly rotate each shape.')
    parser.add_argument('--sketch_dir', type=str, default='sketch', choices={'sketch', 'aligned_sketch'})
    parser.add_argument('--test_sketch_dir', type=str, default='aligned_sketch', choices={'aligned_sketch', 'ss_1.0', 'sketch_npy_other', 'hs_id38'})
    parser.add_argument('--percentage', type=float, default=1.0,
                        help='margin for triplet loss.')

    parser.add_argument('--random_rotate', action='store_true',
                        help='Whether to randomly rotate each shape.')
    parser.add_argument('--rotate_mode', default='360', choices={'360', '4_rotations', '360_p0.5'},
                        help='Whether to randomly rotate each shape.')

    parser.add_argument('--transform_anchor', action='store_true',
                        help='Whether to randomly rotate each shape.')

    parser.add_argument('--random_scale', action='store_true',
                        help='Whether to randomly rotate each shape.')
    parser.add_argument('--use_aug_loss', action='store_true',
                        help='Whether to randomly rotate each shape.')

    parser.add_argument('--use_pn2', action='store_true',
                        help='Whether to randomly rotate each shape.')
    parser.add_argument('--use_softmax', action='store_true',
                        help='Whether to randomly rotate each shape.')
    parser.add_argument('--use_KL', action='store_true',
                        help='Whether to randomly rotate each shape.')


    # logging and saving frequency
    parser.add_argument('--save_dir', type=str, default='/vol/research/sketching/projects/FineGrained_3DSketch')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)


    parser.add_argument('--view', type=int, default=1)
    parser.add_argument('--shape_view', type=int, default=11)
    parser.add_argument('--shape_view_type', type=str, default='shape')
    parser.add_argument('--pretraining', action='store_true',
                        help='Whether to randomly rotate each shape.')
    parser.add_argument('--test_list_file', type=str, default="hs/{}.txt",
                        help="Path to the namelist file")

    parser.add_argument('--sketch_data_type', type=str, default='sketch_amateur')

    return parser


def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='SoftPointFlow Experiment')
    parser = add_args(parser)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join("results", "SoftPointFlow")

    return args
