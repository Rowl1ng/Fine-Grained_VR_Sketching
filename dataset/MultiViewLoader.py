import numpy as np
import torch.utils.data
from PIL import Image
import torch
from torchvision import transforms, datasets
from tools.misc import load_string_list
import os

class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, set='train', list_file='', data_dir='', test_mode=True, shuffle=False, data_type='', num_views=12, view=1):

        self.test_mode = test_mode
        self.split = set
        self.eval = self.split in ['test', 'val']
        self.shape_id = []
        if self.eval and data_type == 'shape':
            list_file = list_file.replace('.txt', '_shape.txt')

        self.name_list = load_string_list(os.path.join(data_dir, 'list', list_file.format(set)))
        self.file_paths = []

        all_files = []
        if 'shape' in data_type:
            self.num_views = num_views
            if self.num_views == 12:
                for line in self.name_list:
                    shape_paths = [(os.path.join(data_dir, 'view_based', data_type, line + '_{}.png')).format(i) for i in range(12)]
                    all_files.extend(shape_paths)  # Edge
            else:
                for line in self.name_list:
                    shape_paths = [(os.path.join(data_dir, 'view_based', data_type, '{}_{}.png'.format(line, view)))]
                    all_files.extend(shape_paths)  # Edge
            self.shape_id.append(line)
        elif 'sketch' in data_type:
            if num_views == 3:
                self.num_views = 3
                for line in self.name_list:
                    sketch_paths = [
                        os.path.join(data_dir, 'view_based', data_type, '{}_{}.png'.format(line, view)) for i in range(3)]

                    all_files.extend(sketch_paths)
            else:
                self.num_views = 1
                for line in self.name_list:
                    sketch_paths = [(os.path.join(data_dir, 'view_based', data_type, '{}_{}.png'.format(line, view)))]

                    all_files.extend(sketch_paths)
            self.shape_id.append(line)

        ## Select subset for different number of views
        self.file_paths = all_files

        print('The size of %s data is %d' % (set, len(self.name_list)))
        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return int(len(self.file_paths) / self.num_views)

    def __getitem__(self, idx):
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.file_paths[idx * self.num_views + i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        # return (class_id, torch.stack(imgs), self.file_paths[idx * self.num_views:(idx + 1) * self.num_views])
        return (torch.stack(imgs), idx)

if __name__ == "__main__":
    list_file = r'C:\Users\ll00931\OneDrive - University of Surrey\Documents\chair_1005\all_networks\list\hs\{}_view.txt'
    data_dir = r'C:\Users\ll00931\OneDrive - University of Surrey\Documents\chair_1005\all_networks'
    train_shape_dataset = MultiviewImgDataset(set='train', list_file=list_file, data_dir=data_dir, test_mode=False, data_type='shape')
    train_sketch_dataset = MultiviewImgDataset(set='train', list_file=list_file, data_dir=data_dir, test_mode=False, data_type='sketch')
    from dataset.TripletSampler import BatchSampler

    minibatch = 2
    seed = 0
    shape_train_batch_sampler = BatchSampler(train_shape_dataset, minibatch, seed)
    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_sampler=shape_train_batch_sampler)
    sketch_train_batch_sampler = BatchSampler(train_sketch_dataset, minibatch, seed)
    train_sketch_loader = torch.utils.data.DataLoader(train_sketch_dataset, batch_sampler=sketch_train_batch_sampler)

    # for labels, images in DataLoader:
    for _, (shape, sketch) in enumerate(zip(train_shape_loader, train_sketch_loader), 0):
        print(shape[0].shape, sketch[0].shape)

        # print(images.shape)
