import numpy as np
import torch
from torch.utils.data.sampler import Sampler, BatchSampler
from itertools import combinations, permutations
import torch.nn.functional as F

class BatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, seed):

        self.used_indices_count = 0
        self.count = 0
        self.batch_size = batch_size
        self.len = dataset.__len__()
        self.indices = [i for i in range(self.len)]
        self.seed = seed
        np.random.seed(seed)
        np.random.shuffle(self.indices)

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.len:
            np.random.seed(self.seed + int(self.count / self.batch_size))
             # assure any batch contain anchor/other samples from the same classes set
            index = self.used_indices_count
            indices = self.indices[index:index + self.batch_size]

            self.used_indices_count += self.batch_size
            if self.used_indices_count + self.batch_size > self.len:
                np.random.shuffle(self.indices)
                self.used_indices_count = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.len // self.batch_size


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def compute_distance(a, b):
    """cdist (squared euclidean) with pytorch"""
    a_norm = (a**2).sum(1).view(-1, 1)
    b_t = b.permute(1, 0).contiguous()
    b_norm = (b**2).sum(1).view(1, -1)
    dist = a_norm + b_norm - 2.0 * torch.matmul(a, b_t)
    dist[dist != dist] = 0
    return torch.clamp(dist, 0.0, np.inf)

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """
    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

class AllNegativeTripletSelector():
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, symmetric=False, cpu=False):
        super(AllNegativeTripletSelector, self).__init__()
        self.symmetric = symmetric
        self.cpu = cpu

    def get_triplets(self, embeddings, shape_ids):
        if self.cpu:
            embeddings = embeddings.cpu()
        anchor_num = int(embeddings.shape[0]//2)

        triplets = []
        anchor = np.array([i for i in range(anchor_num)])
        if self.symmetric:
            for i in anchor:
                negative_indices = np.where(anchor != i)[0]
                triplets.extend([[i, i + anchor_num, negative + anchor_num] for negative in negative_indices])
                triplets.extend([[i + anchor_num, i, negative] for negative in negative_indices])
        else:
            for i in anchor:
                if shape_ids == None:
                    negative_indices = np.where(anchor != i)[0]
                else:
                    shape_id = shape_ids[i]
                    negative_indices = [j for j in range(anchor_num) if shape_ids[j] != shape_id]
                triplets.extend([[i, i + anchor_num, negative + anchor_num] for negative in negative_indices])

        triplets = np.array(triplets)
        return torch.LongTensor(triplets)


class AllNegativeTripletSelector_network():
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, cpu=False):
        super(AllNegativeTripletSelector_network, self).__init__()
        self.cpu = cpu

    def get_triplets(self, embeddings):
        if self.cpu:
            embeddings = embeddings.cpu()
        B, _ = embeddings.shape
        anchor_num = int(B/3)

        triplets = []
        for i in range(anchor_num):
            positives = [i, i+anchor_num, i+2*anchor_num]
            negative_indices = [i for i in range(anchor_num, B) if i not in positives]
            triplets.extend([[i, i + anchor_num, negative] for negative in negative_indices])
            triplets.extend([[i, i + 2*anchor_num, negative] for negative in negative_indices])

        # anchor = np.array([i for i in range(anchor_num)])

        # for i in anchor:
        #     negative_indices = np.where(anchor != i)[0]
        #     triplets.extend([[i, i + anchor_num, negative + anchor_num] for negative in negative_indices])
        # print(triplets)
        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, sketch_anchor, cpu=False):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.sketch_anchor = sketch_anchor

    def get_triplets(self, embeddings):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        anchor_num = int(embeddings.shape[0]//2)

        triplets = []
        anchor = np.array([i for i in range(anchor_num)])
        if self.sketch_anchor:
            for i in anchor:
                ap_distance = distance_matrix[i, i + anchor_num]
                negative_indices = np.array(np.where(anchor != i)[0]) + anchor_num
                loss_values = ap_distance - distance_matrix[i, torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([i, i + anchor_num, hard_negative])
        else:
            for i in anchor:
                negative_indices_1 = np.array(np.where(anchor != i)[0])
                negative_indices = np.concatenate((negative_indices_1, negative_indices_1+anchor_num), axis=0)
                anchor_positives = [[i, i + anchor_num], [i + anchor_num, i]]  # All anchor-positive pairs
                for item in anchor_positives:
                    ap_distance = distance_matrix[item[0], item[1]]
                    loss_values = ap_distance - distance_matrix[item[0], torch.LongTensor(negative_indices)] + self.margin
                    loss_values = loss_values.data.cpu().numpy()
                    hard_negative = self.negative_selection_fn(loss_values)
                    if hard_negative is not None:
                        hard_negative = negative_indices[hard_negative]
                        triplets.append([i, i + anchor_num, hard_negative])

        if len(triplets) == 0:
            i = 0
            negative_indice = np.random.choice(np.where(anchor != i)[0])
            triplets.append([i, i + anchor_num, negative_indice+anchor_num])

        triplets = np.array(triplets)
        return torch.LongTensor(triplets)

def HardestNegativeTripletSelector(margin, sketch_anchor, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                              sketch_anchor=sketch_anchor,
                                                                                              cpu=cpu)

def RandomNegativeTripletSelector(margin,  cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)

def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)
if __name__ == "__main__":
    list_file = r'C:\Users\ll00931\Documents\chair_1005\list\unique\{}_45participants.txt'  # r'C:\Users\ll00931\Documents\chair_1005\list\unique\{}_45participants.txt'
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args(args=['--debug',
                                   '--use_triplet',
                                   '--use_z',
                                   '--list_file',
                                   r'C:\Users\ll00931\Documents\chair_1005\list\unique\{}_45participants.txt', \
                                   '--sketch_dir', r'C:\Users\ll00931\Documents\chair_1005\pointcloud\final_set', \
                                   '--shape_dir', r"C:\Users\ll00931\Documents\chair_1005\pointcloud\shape", \
                                   '--epoch', '1', \
                                   '--batch_size', '2', \
                                   '--n_flow', '12', \
                                   '--multi_freq', '4', \
                                   '--n_flow_AF', '9', \
                                   '--h_dims_AF', '256-256-256', \
                                   '--save_freq', '1', \
                                   '--valid_freq', '1', \
                                   '--log_freq', '1', \
                                   '--save_dir', r'C:\Users\ll00931\PycharmProjects\FineGrained_3DSketch'
                                   ])

    from dataset.PointCloudLoader import PointCloudDataLoader
    train_shape_dataset = PointCloudDataLoader(args, uniform=True,
                                               data_dir=args.shape_dir,
                                               split='train', data_type='shape')
    train_sketch_dataset = PointCloudDataLoader(args, uniform=True,
                                               data_dir=args.sketch_dir,
                                               split='train', data_type='sketch')

    seed = 0
    shape_train_batch_sampler = BatchSampler(train_shape_dataset, args.batch_size, seed)
    shape_dataloader = torch.utils.data.DataLoader(train_shape_dataset, batch_sampler=shape_train_batch_sampler)

    sketch_train_batch_sampler = BatchSampler(train_sketch_dataset, args.batch_size, seed)
    sketch_dataloader = torch.utils.data.DataLoader(train_sketch_dataset, batch_sampler=sketch_train_batch_sampler)

    # for images, labels in DataLoader:
    for epoch in range(3):
        for i, ([sketches, ida], [shapes, idb]) in enumerate(zip(sketch_dataloader, shape_dataloader)):
            print(i)
            print(sketches.shape, ida, shapes.shape, idb)
            # if i > 3:
            #     print('Epoch {} over'.format(epoch))
            #     break
