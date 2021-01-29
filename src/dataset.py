import torch as th
from torch.utils.data import Dataset, DataLoader
import os
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


class CassavaDataset(Dataset):
    def __init__(self, df,
                 data_dir: os.path,
                 task='train',
                 transform=None):

        self.df = df
        self.images_dir = data_dir
        self.task = task
        self.transform = transform

        # print(self.images_dir)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img_id = self.df.iloc[index].image_id
        img = Image.open(os.path.join(self.images_dir, img_id))
        img = np.array(img).transpose((2, 0, 1))
        # transform to tensor and normalize
        img = th.from_numpy(img).float() / 255.
        # apply transforms if not none
        if self.transform is not None:
            img = self.transform(img)

        sample = {
            'images': img,  # image tensor

        }

        if self.task == 'train':
            target = self.df.iloc[index].label
            sample.update({
                'targets': th.tensor(target, dtype=th.long)
            })

        return sample


class CassavaDM(pl.LightningDataModule):

    def __init__(self,
                 df: pd.DataFrame,
                 train_data_dir: str,
                 test_data_dir: str,
                 data_transforms=None,
                 frac: float = 0,
                 train_batch_size: int = 64,
                 test_batch_size: int = 32,
                 test_size: float = .1,
                 n_classes: int = 5
                 ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.frac = frac
        self.df = df
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.test_size = test_size
        self.n_classes = n_classes
        self.data_transforms = data_transforms

    def setup(self, stage=None):
        # datasets
        # if fraction is fed
        train_df, val_df = train_test_split(self.df, test_size=self.test_size)

        if self.frac > 0:
            train_df = train_df.sample(frac=self.frac).reset_index(drop=True)
            self.train_ds = CassavaDataset(df=train_df,
                                           data_dir=self.train_data_dir,
                                           task='train',
                                           transform=self.data_transforms['train'])
        else:
            self.train_ds = CassavaDataset(df=train_df,
                                           data_dir=self.train_data_dir,
                                           task='train',
                                           transform=self.data_transforms['train'])

        self.val_ds = CassavaDataset(df=val_df,
                                     data_dir=self.train_data_dir,
                                     task='train',
                                     transform=self.data_transforms['test'])

        training_data_size = len(self.train_ds)
        validation_data_size = len(self.val_ds)

        print(
            f'[INFO] Training on {training_data_size} samples belonging to {self.n_classes} classes')
        print(
            f'[INFO] Validating on {validation_data_size} samples belonging to {self.n_classes} classes')

    # data loaders
    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=os.cpu_count())
