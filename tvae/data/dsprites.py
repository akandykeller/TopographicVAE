import os
import numpy as np
from sklearn.utils.extmath import cartesian
from torch.utils.data import DataLoader, Dataset
from urllib import request
import torch


class SequenceDataset(Dataset):
    def __init__(self, seq_len=None, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.seq_transform_idxs = None # set in child class
        self.index_manager = None   # set in child class
        self.factor_sizes = None  # set in child class
        self.data = None  # set in child class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x0 = self.data[idx]
        x0_feat = self.index_manager.index_to_features(idx)
        xt_feat = x0_feat.copy()

        transform_idx = self.seq_transform_idxs[torch.randint(len(self.seq_transform_idxs), (1,))]

        img_seq = [x0]
        feat_seq = [x0_feat]

        for t in range(self.seq_len):
            xt_feat[transform_idx] = (x0_feat[transform_idx] + (1+t)) % self.factor_sizes[transform_idx]

            xt_idx = self.index_manager.features_to_index(xt_feat)
            xt = self.data[xt_idx]

            img_seq.append(xt.copy())
            feat_seq.append(xt_feat.copy())

        img_seq = torch.tensor(img_seq).unsqueeze(1)
        feat_seq = torch.tensor(feat_seq)

        return img_seq, feat_seq


class IndexManger(object):
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes):
        """Index to latent (= features) space and vice versa.
        Args:
          factor_sizes: List of integers with the number of distinct values for each
            of the factors.
        """
        self.factor_sizes = np.array(factor_sizes)
        self.num_total = np.prod(self.factor_sizes)
        self.factor_bases = self.num_total / np.cumprod(self.factor_sizes)

        self.index_to_feat = cartesian([np.array(list(range(i))) for i in self.factor_sizes])

    def features_to_index(self, features):
        """Returns the indices in the input space for given factor configurations.
        Args:
          features: Numpy matrix where each row contains a different factor
            configuration for which the indices in the input space should be
            returned.
        """
        assert np.all((0 <= features) & (features <= self.factor_sizes))
        index = np.array(np.dot(features, self.factor_bases), dtype=np.int64)
        assert np.all((0 <= index) & (index < self.num_total))
        return index

    def index_to_features(self, index):
        assert np.all((0 <= index) & (index < self.num_total))
        features = self.index_to_feat[index]
        assert np.all((0 <= features) & (features <= self.factor_sizes))
        return features


class DSpritesDataset(SequenceDataset):
    """
    A PyTorch wrapper for the dSprites dataset by
    Matthey et al. 2017. The dataset provides a 2D scene
    with a sprite under different transformations:
    # dim, type,     #values  avail.-range
    * 0, color       |  1 |     1-1
    * 1, shape       |  3 |     1-3
    * 2, scale       |  6 |     0.5-1.
    * 3, orientation | 40 |     0-2pi
    * 4, x-position  | 32 |     0-1
    * 5, y-position  | 32 |     0-1
    for details see https://github.com/deepmind/dsprites-dataset
    """
    def __init__(self, dir='/home/akeller/repo/TECA/tvae/datasets/dsprites/', 
                 seq_transforms=['orientation'],
                 avail_transforms=None,
                 max_transform_len=18,
                 **kwargs):
        super().__init__(**kwargs)

        self.url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz" 
        self.path = dir
        self.path_npz = os.path.join(self.path, 'dsprites.npz')
    
        print("Loading DSprites...")
        try:
            full_data = np.load(self.path_npz, encoding="latin1", allow_pickle=True)
        except FileNotFoundError:
            os.makedirs(self.path, exist_ok=True)
            print(f'downloading dataset ... saving to {self.path_npz}')
            request.urlretrieve(self.url, self.path_npz)
            full_data = np.load(self.path_npz, encoding="latin1", allow_pickle=True)
        
        self.data = full_data['imgs'].squeeze().astype(np.float32)
        self.latents_values = full_data['latents_values']
        self.latents_classes = full_data['latents_classes']
        self.metadata = full_data['metadata'][()]

        original_factor_sizes = [3, 6, 40, 32, 32]
        speeds = [1, 1, 2, 2, 2]
        start_idx = [0, 1, 0, 0, 0]
        data_by_factor = self.data.reshape((*original_factor_sizes,) + (64, 64))

        # Subselect Avail Transforms
        if avail_transforms is None:
            avail_transforms = seq_transforms
        print(f"Removing all transforms but {avail_transforms}")
        avail_transform_idxs = [self.metadata['latents_names'][1:].index(x) for x in avail_transforms]
        data_indexer = [-1 if i not in avail_transform_idxs else np.s_[start_idx[i]:max_transform_len:speeds[i]] for i in range(len(original_factor_sizes))]
        self.data = data_by_factor[data_indexer]
        self.factor_sizes = [1 if i not in avail_transform_idxs else self.data.shape[i] for i in range(len(original_factor_sizes))]
        self.data = self.data.reshape(-1, 64, 64)

        # Ignore color index
        self.seq_transforms = seq_transforms
        self.seq_transform_idxs = [self.metadata['latents_names'][1:].index(x) for x in self.seq_transforms]

        self.index_manager = IndexManger(self.factor_sizes)
        
        print("Done")

    def __len__(self):
        return len(self.data)


def get_dataloader(dir='./', 
                   seq_transforms=['posX', 'posY'],
                   avail_transforms=None,
                   seq_len=9, 
                   max_transform_len=18,
                   batch_size=20):
    train_data = DSpritesDataset(dir=dir, seq_transforms=seq_transforms, 
                                          avail_transforms=avail_transforms, 
                                          max_transform_len=max_transform_len,
                                          seq_len=seq_len)
    
    return DataLoader(
		train_data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=20,
		pin_memory=True,
		drop_last=True)


if __name__ == "__main__":
    data_loader = get_dataloader()

    for x, f in data_loader:
        print(x,f)
        break
