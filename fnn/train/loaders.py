import torch
from torch.multiprocessing import spawn, Queue
import numpy as np
from tqdm import tqdm


# -------------- Loader Prototype --------------


class Loader:
    """Data Loader"""

    def _init(self, dataset):
        """
        Parameters
        ----------
        dataset : fnn.data.dataset.Dataset
            dataset to load
        """
        self.dataset = dataset

    def __call__(self, training=True, display_progress=True):
        """
        Parameters
        ----------
        training : bool
            training or validation
        display_progress : bool
            display progress

        Yields
        -------
        dict
            training or validation data
        """
        raise NotImplementedError()


# -------------- Loader Types --------------


class RandomBatches(Loader):
    """Randomly Sampled Batches"""

    def __init__(self, sample_size, batch_size, epoch_size, train_fraction=0.95, seed=42):
        """
        Parameters
        ----------
        sample_size : int
            number of samples in a datapoint
        batch_size : int
            number of datapoints in a batch
        epoch_size : int
            number of batches in an epoch
        train_fraction : float
            fraction of the data used for training
        seed : int
            random seed for splitting data into training/validation
        """
        assert sample_size > 0
        assert batch_size > 0
        assert epoch_size > 0
        assert 0 < train_fraction <= 1

        self.sample_size = int(sample_size)
        self.batch_size = int(batch_size)
        self.epoch_size = int(epoch_size)
        self.train_fraction = float(train_fraction)
        self.seed = int(seed)

    def _init(self, dataset):
        """
        Parameters
        ----------
        dataset : fnn.data.dataset.Dataset
            dataset to load
        """
        assert dataset.samples.min() >= self.sample_size

        n = len(dataset)
        train_n = round(n * self.train_fraction)
        val_n = n - train_n

        self.train_size = round(self.epoch_size * train_n / n)
        self.val_size = round(self.epoch_size * val_n / n)

        rng = np.random.default_rng(self.seed)
        train_idx = rng.choice(n, size=train_n, replace=False)

        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_idx] = True

        keys = np.sort(dataset.index)
        self.train_keys = keys[train_mask]
        self.val_keys = keys[~train_mask]

        super()._init(dataset)

    def _random_keys(self, training=True):
        if training:
            keys = self.train_keys
            size = self.batch_size * self.train_size
        elif self.val_size:
            keys = self.val_keys
            size = self.batch_size * self.val_size
        else:
            return []

        idx = torch.randint(high=len(keys), size=(size,)).numpy()

        return keys[idx].tolist()

    def _random_indexes(self, key):
        high = self.dataset.loc[key].samples - self.sample_size

        if high > 0:
            i = torch.randint(high=high, size=(1,)).item()
        else:
            i = 0

        return i + np.arange(self.sample_size)

    def _load(self, i, queue, keys, indexes):
        assert i == 0
        for key, index in zip(keys, indexes):
            data = self.dataset.load(key, index)
            queue.put(data)

    def __call__(self, training=True, display_progress=True):
        """
        Parameters
        ----------
        training : bool
            training or validation
        display_progress : bool
            display progress

        Yields
        -------
        dict
            training or validation data
        """
        keys = self._random_keys(training)

        if keys:
            indexes = [self._random_indexes(key) for key in keys]
        else:
            return

        d = {item: [] for item in self.dataset.dataitems}
        q = Queue(self.batch_size)
        c = spawn(self._load, args=(q, keys, indexes), nprocs=1, join=False)

        if display_progress:
            if training:
                iterbar = tqdm(desc="Training Batches", total=self.train_size)
            else:
                iterbar = tqdm(desc="Validation Batches", total=self.val_size)

        for b, _ in enumerate(keys):

            for k, v in q.get().items():
                d[k].append(v)

            if (b + 1) % self.batch_size:
                continue

            batch = {}
            for k, v in d.items():
                batch[k] = np.stack(v, axis=1)
                v.clear()

            if display_progress:
                iterbar.update(n=1)

            yield batch

        assert c.join()
