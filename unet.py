import dataclasses

import flax.linen as nn
import flax.struct
import jax.lax
import jax.numpy as jnp
from typing import Optional, Union, Tuple
from common import LabelEmbedder


import math

import jax
import typing
import flax.linen as nn
import jax.numpy as jnp
from typing import OrderedDict
from dataclasses import fields


from jax import numpy as jnp
import jax
from flax import linen as nn
import typing
from dit import DiTBlock

import typing

import jax
from jax import numpy as jnp
from flax import linen as nn

import math

import jax
import typing
import flax.linen as nn
import jax.numpy as jnp
from typing import OrderedDict
from dataclasses import fields

import math
import typing
from functools import partial
from jax.sharding import PartitionSpec
import jax
from jax import numpy as jnp
from flax import linen as nn

from common import TimestepEmbedder
import jax



class Identity(nn.Module):
    """A simple identity module that returns its input."""
    def __call__(self, x):
        return x


class Upsample(nn.Module):
    in_channels: int


    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1))
        )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Downsample(nn.Module):
    in_channels: int

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
        )

    def __call__(self, hidden_states):
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        hidden_states = jnp.pad(hidden_states, pad_width=pad)
        hidden_states = self.conv(hidden_states)
        return hidden_states

class ResnetBlock2D(nn.Module):
    in_c: int
    out_c: int = None
    use_shortcut: bool= None
    dropout_rate: float = 0.0
    epsilon: float = 1e-5


    def setup(self) -> None:
        out_c = self.out_c or self.in_c

        self.c1 = nn.Conv(
            features=out_c,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        )
        self.norm1 = nn.GroupNorm(
            32, epsilon=self.epsilon
        )

        self.c2 = nn.Conv(
            features=out_c,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        )
        self.norm2 = nn.GroupNorm(
            32, epsilon=self.epsilon
        )

        self.time_emb = nn.Dense(
            out_c,
        )
        self.drop = nn.Dropout(
            self.dropout_rate
        )

        cut = self.in_c != out_c if self.use_shortcut is None else self.use_shortcut
        if cut:
            self.cs = nn.Conv(
                out_c,
                kernel_size=(1, 1),
                padding="VALID",
                strides=(1, 1),
            )
        else:
            self.cs=Identity()

    def __call__(self, x, c, deterministic=False):
        # print("Res block, x= ", x.shape, " c= ", c.shape, " in_c", self.in_c, " out_c", self.out_c)
        resid = x
        hidden_state = self.c1(nn.swish(self.norm1(x)))
        c = jnp.expand_dims(jnp.expand_dims(self.time_emb(nn.swish(c)), 1), 1)

        hidden_state += c
        hidden_state = self.c2(self.drop(nn.swish(self.norm2(hidden_state)), deterministic=deterministic))

        if hasattr(self, 'cs'):
            resid = self.cs(resid)
        return hidden_state + resid



class Block(nn.Module):
    in_channels: int
    out_channels: int
    num_attention_heads: int
    num_hidden_layers: int = 1
    add_downsampler: bool = False
    add_upsampler: bool = False
    attention: bool = False
    perv_out_channels: int = None


    def setup(self) -> None:
        blocks =[]

        for index in range(self.num_hidden_layers):
            in_channels = self.in_channels if index == 0 else self.out_channels
            if (index == 0) and self.perv_out_channels:
                in_channels = in_channels + self.perv_out_channels
            if(self.attention):
                blocks.append(
                    (ResnetBlock2D(
                        in_c=in_channels,
                        out_c=self.out_channels,
                        use_shortcut=True,
                    ), DiTBlock(
                        hidden_size=self.out_channels,
                        num_heads=self.num_attention_heads,
                    ))
                )
            else:
                res_n = ResnetBlock2D(
                    in_c=in_channels,
                    out_c=self.out_channels,
                    use_shortcut=True,
                )
                blocks.append(res_n)

        self.blocks = blocks
        if self.add_downsampler:
            self.downsamplers_0 = Downsample(
                self.out_channels
            )
        if self.add_upsampler:
            self.upsamplers_0 = Upsample(
                self.out_channels
            )

    def __call__(self, x, c, skips=None):
        output_states = ()
        
            # make x reach target dim

        for block in self.blocks:
            # print("Block, x= ", x.shape, " c= ", c.shape, " Is DiTBlock= ", block is DiTBlock, " type(block)", type(block))
            if skips:
                skip = skips[-1]
                skips = skips[:-1]
                x = jnp.concatenate([x, skip], axis=-1)
            # check if block is a tuple
            if isinstance(block, tuple):
                (resnet, ditblock) = block
                # basically we need to reshape x to (B, H*W, C)
                x = resnet(x, c)
                B, H, W, C = x.shape
                x = x.reshape(x.shape[0], -1, x.shape[-1])
                # print("Giving to DiT, x=", x.shape, " c=", c.shape)
                x = ditblock(x, c)
                x = x.reshape(B, H, W, C)
            else:
                x = block(x, c)
            output_states += (x,)
        if self.add_downsampler:
            x = self.downsamplers_0(x)
            output_states += (x,)
        if self.add_upsampler:
            x = self.upsamplers_0(x)
            output_states += (x,)
            
        return x, output_states



class Unet2DConditionModel(nn.Module):

    depth: int = 8
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 10


    sample_size: int = 32
    in_channels: int = 4
    out_channels: int = 4

    down_layers: typing.Tuple = ("attention", "attention", "attention", "resnet")
    up_layers: typing.Tuple = ("resnet",  "attention", "attention", "attention")
    channels: typing.Tuple = (320, 640, 640, 1280, 1280)
    num_hidden_layers_per_block: int = 2




    def init_weights(self, rng):

        sample = jnp.zeros((1, self.in_channels, self.sample_size, self.sample_size), dtype=self.dtype)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=self.dtype)

        params_rng, dropout_rng = jax.random.split(rng)

        return self.init({"params": params_rng, "dropout": dropout_rng}, sample, timesteps, encoder_hidden_states)[
            "params"]

    def setup(self) -> None:
        embedding_dimension = self.channels[0] * 4
        num_attention_heads = self.num_heads
        self.conv_in = nn.Conv(
            self.channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1))
        )


        num_attention_heads = [num_attention_heads] * len(self.down_layers)

        output_channel = self.channels[0]
        down_blocks: typing.List[Block] = []
        for i, name in enumerate(self.down_layers):
            in_channels = output_channel
            output_channel = self.channels[i]
            is_final_b = i == len(self.down_layers) - 1
            block = Block(
                num_attention_heads=num_attention_heads[i],
                add_downsampler=not is_final_b,
                num_hidden_layers=self.num_hidden_layers_per_block,
                in_channels=in_channels,
                attention= name=="attention",
                out_channels=output_channel,
                add_upsampler=False,
            )
            down_blocks.append(block)
        self.bottle_neck = Block(
            in_channels=self.channels[-1],
            out_channels=self.channels[-1],
            num_attention_heads=num_attention_heads[-1],
            add_upsampler=False,
            add_downsampler=False
        )
        up_blocks: typing.List[Block] = []
        reversed_block_out_channels = list(self.channels[::-1])
        reversed_num_attention_heads = list(num_attention_heads[::-1])
        output_channel = reversed_block_out_channels[0]
        for i, name in enumerate(self.up_layers):
            perv_output_channel = output_channel
            in_channels = reversed_block_out_channels[min(i + 1, len(self.channels) - 1)]
            output_channel = reversed_block_out_channels[i]
            is_final_b = i == len(reversed_block_out_channels) - 1
            block = Block(
                in_channels=in_channels,
                out_channels=output_channel,
                perv_out_channels=perv_output_channel,
                num_attention_heads=reversed_num_attention_heads[i],
                num_hidden_layers=self.num_hidden_layers_per_block + 1,
                add_upsampler=not is_final_b,
                add_downsampler=False,
                attention = name=="attention"
            )
            up_blocks.append(block)

        self.norm_out = nn.GroupNorm(32)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        )
        self.up_blocks = up_blocks
        self.down_blocks = down_blocks
        self.label_embedder = LabelEmbedder(self.class_dropout_prob, self.num_classes, embedding_dimension)
        self.time_proj = TimestepEmbedder(embedding_dimension)

    def __call__(self,
                 x, t, y, train=False, force_drop_ids=None
                 ):

        # x = (B, C, H, W)
        # t = (B,)
        # y = (B,)
        x = x.transpose((0, 2, 3, 1))

        t = self.time_proj(t)
        y = self.label_embedder(y, train=train, force_drop_ids=force_drop_ids)
        c = t + y
        x = self.conv_in(x)
        hs = (x,)
        for block in self.down_blocks:
            # print("Preparing, Down, ", ", Is attention" ,block.attention, " in:", x.shape, "c:", c.shape)
            x_in = x
            x, res_hidden_states = block(x, c)
            hs += tuple(res_hidden_states)
            # print("Down Block, in:", x_in.shape, "out:", x.shape, " skip: ", res_hidden_states[-1].shape)
        x_in = x
        x = self.bottle_neck(
            x,
            c
        )[0]
        # print("Bottle Neck, in:", x_in.shape, "out:", x.shape)


        for block in self.up_blocks:
            x_in = x
            res_hidden_states = hs[-(self.num_hidden_layers_per_block + 1):]
            hs = hs[: -(self.num_hidden_layers_per_block + 1)]
            x = block(
                x,
                c,
                skips=res_hidden_states
            )[0]
            # print("Up Block, in:", x_in.shape, "out:", x.shape, " skip: ", res_hidden_states[0].shape)

        x = self.conv_out(nn.swish(self.norm_out(x)))
        x = x.transpose((0, 3, 1, 2))
        return x

# # we have 4 classes of models, 10M, 50M, 100M, 500M
def UNet10M(patch_size, num_classes, class_dropout_prob):
    model = Unet2DConditionModel(
        in_channels=4,
        out_channels=4,
        down_layers=("attention", "attention", "resnet"),
        up_layers=("resnet",  "attention", "attention"),
        num_hidden_layers_per_block=1,
        channels=(64, 128, 256),
        num_heads=16,
        class_dropout_prob=0.1,
    )
    return model

def UNet50M(patch_size, num_classes, class_dropout_prob):
    model = Unet2DConditionModel(
        in_channels=4,
        out_channels=4,
        down_layers=("attention", "attention",  "resnet"),
        up_layers=("resnet",  "attention", "attention"),
        num_hidden_layers_per_block=2,
        channels=(128, 224, 448),
        num_heads=16,
        class_dropout_prob=0.1,
    )
    return model



def UNet100M(patch_size, num_classes, class_dropout_prob):
    model = Unet2DConditionModel(
        in_channels=4,
        out_channels=4,
        down_layers=("attention", "attention","attention",  "resnet"),
        up_layers=("resnet",  "attention","attention", "attention"),
        num_hidden_layers_per_block=2,
        channels=(224, 256, 384, 384),
        num_heads=16,
        class_dropout_prob=0.1,
    )
    return model

# try
#
def test_flax_unet():
    x = jnp.ones((1, 32, 32, 4))
    y = jnp.ones((1,)).astype(jnp.int32)
    t = jnp.ones((1, ))
    model = UNet100M(patch_size=1, num_classes=10, class_dropout_prob=0.1)
    tabulated_output = nn.tabulate(model, jax.random.key(0), compute_flops=True, depth=1)
    print(tabulated_output(x=x, y=y, t=t))
    params = model.init(x=x, y=y, t=t, rngs=jax.random.PRNGKey(0))
    out = model.apply(params, x=x, y=y, t=t, rngs=jax.random.PRNGKey(0))
    assert out.shape == (1, 32, 32, 4)
    # number of params
    num_params = sum([p.size for p in jax.tree_flatten(params)[0]])
    print(f"Number of parameters in UNet2DConditionModel: {num_params}")

if __name__ == '__main__':
    test_flax_unet()
