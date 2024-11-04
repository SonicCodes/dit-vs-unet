
import numpy as np
import torch
import tqdm
import json
import jax.numpy as jnp
import jax
from torch.utils.data import Dataset, DataLoader, distributed
import numpy as np
class ImageNetDataset(Dataset):
    def __init__(self, data_path, labels_path=None, is_train=False):
        self.data = np.memmap(data_path, dtype='float16', mode='r', shape=(1281152, 32, 32, 4))
        # basically random shuffle self.data with a seed
        rows = self.data.shape[0]
        indices = np.arange(rows)

        shuffle_rng = np.random.RandomState(123) 
        shuffle_rng.shuffle(indices)
        self.indices = indices
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        if is_train:
            # 95% of the data is used for training
            length = int(len(self.indices) * 0.9)
            self.indices = self.indices[:length]
        else:
            # 5% of the data is used for validation
            length = int(len(self.indices) * 0.1)
            self.indices = self.indices[-length:]
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, _idx):
        idx = self.indices[_idx]
        image = self.data[idx]
        label = int(self.labels[idx][0])
        image = (image.astype(np.float32).transpose(2, 0, 1)) * 10.0
        # image = (image / 255.0 - 0.5) * 24.0
        return image, label#, label_text

def loopy(dl):
    while True:
        for x in iter(dl): yield x

def get_dataset(batch_size, train=False):
    seed = 19 if train else 23
    data_path = 'inet.fp16.npy'
    labels_path = 'inet.fp16.json'
    dataset = ImageNetDataset(data_path, labels_path, train)
    num_processes = jax.process_count()
    rank = jax.process_index()
    dataset_sampler = distributed.DistributedSampler(dataset, shuffle=True, drop_last=True, num_replicas=num_processes, rank=rank, seed=seed)
    # dataloader that respects nodes and devices
    # get process id, and number of process from jax
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = np.stack(images)
        labels = np.stack(labels)
        return images*0.0.13025, labels
    
    while True:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, collate_fn=collate_fn, sampler=dataset_sampler)
        for images, labels in iter(dataloader):
            yield jnp.array(images),  jnp.array(labels)


# if __name__ == '__main__':
#     dataloader = get_dataset(256, train=True)
#     import tqdm 
#     prgb = tqdm.trange(10000)
#     for _ in tqdm.trange(10000):
#         images, labels = next(dataloader)
#         # print (images.shape, labels.shape)
#         # break
#         prgb.set_postfix({'images': images.shape, 'labels': labels.shape})
#         # print(images.shape, labels.shape, ltxt)
#         # print(images.shape, labels.shape)
#         # break
#         pass
