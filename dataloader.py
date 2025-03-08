from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms
import pandas as pd
import os.path as path
from sklearn.preprocessing import scale, StandardScaler, MaxAbsScaler
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split


kwargs = {"shuffle": True, "num_workers": 16, "pin_memory": True}


def load_transform(dataname='celeba'):
    if dataname == 'celeba':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_size = 128
        crop_size = 128

        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)

        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif dataname == 'UTK':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_size = 128
        crop_size = 128

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif dataname == 'dnc':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_size = 128
        crop_size = 128

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return transform, transform_test


def ImageLoader(dataname='celeba'):
    if dataname == 'celeba':
        root_dir = './data/CelebA'
        attr_file = os.path.join(root_dir, 'list_attr_celeba.txt')
        partition_file = os.path.join(root_dir, 'list_eval_partition.txt')

        #
        attr_data = pd.read_csv(attr_file, skiprows=1, delim_whitespace=True)
        attr_data.reset_index(inplace=True)
        attr_data.rename(columns={'index': 'image_id'}, inplace=True)

        #
        partition_data = pd.read_csv(partition_file, delim_whitespace=True, header=None,
                                     names=['image_id', 'partition'])

        #
        full_data = pd.merge(attr_data, partition_data, on='image_id')

        #
        train_data = full_data[full_data['partition'] == 0]
        valid_data = full_data[full_data['partition'] == 1]
        test_data = full_data[full_data['partition'] == 2]

        return train_data.reset_index(drop=True), valid_data.reset_index(drop=True), test_data.reset_index(drop=True)

    if dataname == 'UTK':
        root_dir = './data/UTKFace'
        all_files = []

        # traverse root_dir
        filenames = [f for f in os.listdir(root_dir) if f.lower().endswith('.jpg')]
        # random shuffle
        np.random.shuffle(filenames)

        for fname in filenames:
            # fname like: '26_0_2_20170103183828330.jpg'
            parts = fname.split('_')
            if len(parts) < 4:
                continue
            try:
                age = int(parts[0])  # age
                gender = int(parts[1])  # gender(0/1)
            except:
                print('invalid format')
                continue

            # age>=30: label=1, else label=0
            label = 1 if age >= 30 else 0
            # image path
            full_path = os.path.join(root_dir, fname)
            all_files.append((full_path, gender, label))

        # divide into train/valid/test
        N = len(all_files)
        n_train = int(0.8 * N)
        n_valid = int(0.1 * N)

        train_data = all_files[:n_train]
        valid_data = all_files[n_train:n_train + n_valid]
        test_data = all_files[n_train + n_valid:]

        # convert to DataFrame
        cols = ['filename', 'sens', 'label']
        train_data = pd.DataFrame(train_data, columns=cols)
        valid_data = pd.DataFrame(valid_data, columns=cols)
        test_data = pd.DataFrame(test_data, columns=cols)

        return train_data.reset_index(drop=True), valid_data.reset_index(drop=True), test_data.reset_index(drop=True)

    if dataname == 'celebahq':
        data = pd.read_csv('/data/celeba-hq/hq_to_small.csv')
        data['idx'] = data['idx'].apply(lambda x: '/data/celeba-hq/CelebA-HQ/combined/imgHQ' + str(x).zfill(5) + '.npy')

        train_data, test_data = train_test_split(data, random_state=2021, test_size=0.9)

        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


class ImageDataset(Dataset):
    def __init__(self, data, dataname, sens_idx, label_idx, root_dir=None, transform=None):
        self.transform = transform
        self.data = data
        self.dataname = dataname

        if self.dataname == 'celeba':
            self.sens_idx = sens_idx
            self.label_idx = label_idx
            self.root_dir = root_dir
        elif self.dataname == 'UTK':
            self.sens_idx = sens_idx
            self.label_idx = label_idx
            self.root_dir = root_dir
        elif self.dataname == 'celebahq':
            self.label_idx = data.columns.get_loc(label_idx)
            self.sens_idx = data.columns.get_loc(sens_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 0]

        if self.dataname == 'celeba':
            img_name = os.path.join(self.root_dir, img_name)

            image = Image.open(img_name)
            sens = self.data[self.sens_idx][idx]
            label = self.data[self.label_idx][idx]

        elif self.dataname == 'UTK':
            # UTK: DataFrame column: [filename, sens, label]
            img_name = self.data.iloc[idx, 0]  # 'filename'
            sens = self.data.iloc[idx, 1]  # 'sens'
            label = self.data.iloc[idx, 2]  # 'label'

            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)

            # convert to 0/1
            sens = int(sens)
            label = int(label)
            return image, sens, label

        elif self.dataname == 'celebahq':
            image = np.load(img_name)
            image = image.reshape(3, 1024, 1024)
            image = np.transpose(image, (1, 2, 0))
            image = Image.fromarray(image)

            sens = np.array(self.data.iloc[idx][self.sens_idx], dtype=float)
            label = np.array(self.data.iloc[idx][self.label_idx], dtype=float)
            name = self.data['idx'][idx].split('/')[-1].split('.')[0]

        if self.transform:
            image = self.transform(image)

        return image, max(int(sens), 0), max(int(label), 0)
