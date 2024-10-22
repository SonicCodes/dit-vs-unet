from streaming.base.format.mds.encodings import Encoding, _encodings
import numpy as np
from typing import Any
import torch
from streaming import StreamingDataset
import streaming
import pdb
import jax.numpy as jnp
class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x=  np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0

_encodings["uint8"] = uint8

remote_train_dir = "./vae_mds" # this is the path you installed this dataset.
local_train_dir = "./local_train_dir"


def infinite_iterator(original_loader):
    while True:
        try :
            for data in iter(original_loader):
                yield data
        except StopIteration:
            pass





def get_dataset(batch_size=32, train=False):
    streaming.base.util.clean_stale_shared_memory()
    dataset = StreamingDataset(
        local=local_train_dir,
        remote=remote_train_dir,
        split=None,
        shuffle=True,
        shuffle_algo="naive",
        num_canonical_nodes=1,
        batch_size = batch_size
    )

    # # split 90%10%
    # total_len = len(dataset)
    # train_len = int(total_len * 0.9)
    # test_len = total_len - train_len
    # dataset = dataset[:train_len] if train else dataset[train_len:]

    def collate_fn(batch):
        latents, labels = [], []
        for samples in batch:
            # pdb.set_trace()
            # print(samples.keys())
            latents.append((samples["vae_output"]).reshape(4, 32, 32))
            labels.append(int(samples["label"]))
        stacked_latents = jnp.stack(latents)#.transpose((0, 2, 3, 1))
        stacked_labels = jnp.stack(labels)
        # print (stacked_latents.shape, stacked_labels.shape)
        return stacked_latents*0.13025, stacked_labels

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  collate_fn=collate_fn, drop_last=True)

    return infinite_iterator(dataloader)
