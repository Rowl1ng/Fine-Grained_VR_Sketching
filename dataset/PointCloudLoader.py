import numpy as np
import warnings
from torch.utils.data import Dataset
import os
import pickle

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint, fix=False, seed=0):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    if fix:
        np.random.seed(seed)
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

import random
class PointCloudDataLoader(Dataset):
    def __init__(self, npoints, list_file, split='train', uniform=False, cache_size=15000, data_dir='', data_type='shape', percentage=1.0, debug=False, seed=0):

        self.npoints = npoints
        self.uniform = uniform
        self.split = split
        self.eval = self.split in ['test', 'val']
        if self.eval and data_type == 'shape':
            list_file = list_file.replace('.txt', '_shape.txt')
        self.name_list = [line.rstrip() for line in open(os.path.join(data_dir, 'list', list_file.format(split)))]
        if percentage < 1.0 and not self.eval:
            self.name_list = [line.rstrip() for line in open(os.path.join(data_dir, 'list/hs/train_size/{}_{}.txt'.format(percentage, seed)))]
        self.seed = seed
        self.datapath = []
        self.shape_id = []
        np.random.seed(seed)
        for model_name in self.name_list:
            self.shape_id.append(model_name)
            shape_path = os.path.join(data_dir, data_type, model_name + '.npy')
            self.datapath.append(shape_path)
        # if percentage < 1.0:
        #     random.Random(seed).shuffle(self.datapath)# random.shuffle(self.datapath)
        #     train_num = len(self.datapath)
        #     self.datapath = self.datapath[:int(train_num * percentage)]
        if debug:
            index = 210
            self.datapath = self.datapath[:index]

        print('The size of %s data of type %s is %d' % (split, data_type, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple


    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            point_set = self.cache[index]
        else:
            file_path = self.datapath[index]
            point_set = np.load(file_path).astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints, fix=self.eval, seed=0)
            else:
                if self.eval:
                    np.random.seed(0)
                farthest = np.random.randint(len(point_set), size=self.npoints)
                point_set = point_set[farthest, :]
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if len(self.cache) < self.cache_size:
                self.cache[index] = point_set

        return point_set, self.datapath[index]

class PointCloudDataLoader_aug(Dataset):
    def __init__(self, npoints, list_file=[], uniform=False, cache_size=15000, data_dir='', aug_dir='', data_type='shape', percentage=1.0, debug=False, seed=0):

        self.npoints = npoints
        self.uniform = uniform
        self.datapath = []
        self.shape_id = []
        split = 'train'
        np.random.seed(0)
        hs_list, aug_list = list_file
        if percentage < 1.0:
            hs_name_list = [line.rstrip() for line in open(os.path.join(data_dir, 'list/hs/train_size/{}_{}.txt'.format(percentage, seed)))]
        else:
            hs_name_list = [line.rstrip() for line in open(os.path.join(data_dir, 'list', hs_list.format(split)))]
        for line in hs_name_list:
            self.shape_id.append(line)
            shape_path = os.path.join(data_dir, data_type, line + '.npy')
            self.datapath.append(shape_path)
        aug_name_list = [line.rstrip() for line in open(os.path.join(data_dir, 'list', aug_list))]

        if data_type == 'shape':
            x_dir = 'shape'
        else:
            x_dir = aug_dir
        for line in aug_name_list:
            self.shape_id.append(line)
            shape_path = os.path.join(data_dir, x_dir, line + '.npy')
            self.datapath.append(shape_path)

        if debug:
            index = 100
            self.datapath = self.datapath[:index]

        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple
    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        fix = False

        if index in self.cache:
            point_set = self.cache[index]
        else:
            file_path = self.datapath[index]
            point_set = np.load(file_path).astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints, fix=fix)
            else:
                point_set = point_set[0:self.npoints, :]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if len(self.cache) < self.cache_size:
                self.cache[index] = point_set
        return point_set, self.shape_id[index]

class PointCloudDataLoader_ss(Dataset):
    def __init__(self, npoints, list_file, split='train', uniform=False, cache_size=15000, data_dir='', data_type='shape', debug=False):

        self.npoints = npoints
        self.uniform = uniform
        self.list_file = list_file
        self.split = split
        self.name_list = [line.rstrip().split(' ')[0] for line in open(self.list_file.format(split))]
        self.datapath = []
        self.shape_id = []
        np.random.seed(0)
        ext_dict = {
            'shape': '_opt.npy',
            'network': '_opt_quad_network_20_aggredated.npy',
            'sketch': '_sketch_1.0.npy'
        }
        for model_name in self.name_list:
            self.shape_id.append(model_name)
            shape_path = os.path.join(data_dir, data_type, model_name + ext_dict[data_type])
            self.datapath.append(shape_path)

        if debug:
            index = 10
            self.datapath = self.datapath[:index]

        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple


    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        fix = self.split in ['test', 'val']

        if index in self.cache:
            point_set = self.cache[index]
        else:
            file_path = self.datapath[index]
            point_set = np.load(file_path).astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints, fix=fix)
            else:
                point_set = point_set[0:self.npoints, :]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if len(self.cache) < self.cache_size:
                self.cache[index] = point_set

        return point_set, index

class PointCloudDataLoader_pair(Dataset):
    def __init__(self, args, list_file, split='train', uniform=False,  debug=False):
        self.uniform = uniform
        self.list_file = list_file
        self.split = split
        self.name_list = [line.rstrip() for line in open(self.list_file.format(split))]

        if debug:
            index = 50
            self.name_list = self.name_list[:index]

        self.shape_datapath = []
        self.sketch_datapath = []
        self.pairs = []
        self.shape_id = []
        self.len = len(self.name_list)
        np.random.seed(0)

        self.pairs = [[i, j] for i in range(self.len) for j in range(self.len)]

        print('The size of %s data is %d' % (split, len(self.pairs)))

        self.sketch_cache = []  # from index to (point_set, cls) tuple
        self.shape_cache = []
        def process(npy_path, num_point):
            point_set = np.load(npy_path).astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, num_point, fix=True)
            else:
                point_set = point_set[0:num_point, :]

            point_set = pc_normalize(point_set)
            return point_set

        shape_path = os.path.join(args.save_dir, 'data/shape_{}.pkl'.format(args.shape_point))
        if os.path.exists(shape_path):
            with open(shape_path, 'rb') as file:
                self.shape_cache = pickle.load(file)
        else:
            for line in self.name_list:
                model_name = line.split('_')[1]
                shape_sample_path = os.path.join(args.shape_dir, model_name + '.npy')
                shape = process(shape_sample_path, args.shape_point)
                self.shape_cache.append(shape)
            with open(shape_path, 'wb') as file:
                pickle.dump(self.shape_cache, file)

        sketch_path = os.path.join(args.save_dir, 'data/sketch_{}.pkl'.format(args.sketch_point))
        if os.path.exists(sketch_path):
            with open(sketch_path, 'rb') as file:
                self.sketch_cache = pickle.load(file)
        else:
            for line in self.name_list:
                sketch_sample_path = os.path.join(args.sketch_dir, line + '.npy')
                sketch = process(sketch_sample_path, args.sketch_point)
                self.sketch_cache.append(sketch)
            with open(sketch_path, 'wb') as file:
                pickle.dump(self.sketch_cache, file)


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        sketch_id, shape_id = self.pairs[index]

        return self.sketch_cache[sketch_id], self.shape_cache[shape_id]


if __name__ == '__main__':
    import torch
    list_file = r'C:\Users\ll00931\Documents\chair_1005\list\unique\{}_45participants.txt'
    num_point = 2048
    n_classes = 8
    n_samples = 1
    test_shape_dataset = PointCloudDataLoader(list_file=list_file, npoint=num_point,
                                              uniform=False,
                                              split='train', sketch=False)
    test_sketch_dataset = PointCloudDataLoader(list_file=list_file, npoint=num_point,
                                               uniform=False,
                                               split='train', shape=False)

    from dataset.TripletSampler import BalancedBatchSampler

    test_shape_sampler = BalancedBatchSampler(test_shape_dataset.labels, n_classes=n_classes, n_samples=n_samples, seed=0, n_dataset=len(test_sketch_dataset.labels))
    test_shape_loader = torch.utils.data.DataLoader(test_shape_dataset, batch_sampler=test_shape_sampler, num_workers=4)
    test_sketch_sampler = BalancedBatchSampler(test_sketch_dataset.labels, n_classes=n_classes, n_samples=n_samples, seed=0)
    test_sketch_loader = torch.utils.data.DataLoader(test_sketch_dataset, batch_sampler=test_sketch_sampler, num_workers=4)

    # DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    i = 0
    for i, ((sketches, k_labels), (shapes, p_labels)) in enumerate(zip(test_sketch_loader, test_shape_loader)):
        # print(sketches.shape)
        print(k_labels, p_labels)
        i += 1
        if i > 50:
            break
