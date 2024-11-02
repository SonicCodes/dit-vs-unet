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

class FlaxResnetBlock2D(nn.Module):
    in_c: int
    out_c: int = None
    use_shortcut: bool= None
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:
        out_c = self.out_c or self.in_c

        self.c1 = nn.Conv(
            features=out_c,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm1 = nn.GroupNorm(
            32, epsilon=self.epsilon
        )

        self.c2 = nn.Conv(
            features=out_c,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm2 = nn.GroupNorm(
            32, epsilon=self.epsilon
        )

        self.time_emb = nn.Dense(
            out_c,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
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
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        else:
            self.cs=Identity()

    def __call__(self, hidden_state, time, deterministic=False):
        residual = hidden_state
        hidden_state = self.c1(nn.swish(self.norm1(hidden_state)))
        time = jnp.expand_dims(jnp.expand_dims(self.time_emb(nn.swish(time)), 1), 1)

        hidden_state += time
        hidden_state = self.c2(self.drop(nn.swish(self.norm2(hidden_state)), deterministic=deterministic))

        if hasattr(self, 'cs'):
            residual = self.cs(residual)
        return hidden_state + residual


class FlaxResnetBlock2DNTime(nn.Module):
    in_c: int
    out_c: int = None
    use_shortcut: bool = None
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:
        out_c = self.out_c or self.in_c
        self.c1 = nn.Conv(
            features=out_c,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm1 = nn.GroupNorm(32, epsilon=self.epsilon)

        self.c2 = nn.Conv(
            features=out_c,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm2 = nn.GroupNorm(32, epsilon=self.epsilon)

        self.drop = nn.Dropout(self.dropout_rate)

        cut = self.in_c != out_c if self.use_shortcut is None else self.use_shortcut
        self._cut = cut
        if cut:
            self.cs = nn.Conv(
                out_c,
                kernel_size=(1, 1),
                padding="VALID",
                strides=(1, 1),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self, hidden_state, deterministic=False):
        residual = hidden_state
        hidden_state = self.c1(nn.swish(self.norm1(hidden_state)))
        # print(f"HIDDEN : {hidden_state.shape} | IN_C : {self.in_c} | OUT_C : {self.out_c}")
        hidden_state = self.c2(self.drop(nn.swish(self.norm2(hidden_state)), deterministic=deterministic))
        # print(f"C2 : {hidden_state.shape} | CUT : {self._cut}")
        if hasattr(self, 'cs'):
            residual = self.cs(residual)
        # print(f"CS : {hidden_state.shape} | RESIDUAL : {residual.shape}")
        # print('*' * 15)
        return hidden_state + residual


# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def get_gradient_checkpointing_policy(name):
    return {
        "everything_saveable": jax.checkpoint_policies.everything_saveable,
        "nothing_saveable": jax.checkpoint_policies.nothing_saveable,
        "dots_saveable": jax.checkpoint_policies.dots_saveable,
        "checkpoint_dots": jax.checkpoint_policies.dots_saveable,
        "dots_with_no_batch_dims_saveable": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        "checkpoint_dots_with_no_batch_dims": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        "save_anything_except_these_names": jax.checkpoint_policies.save_anything_except_these_names,
        "save_any_names_but_these": jax.checkpoint_policies.save_any_names_but_these,
        "save_only_these_names": jax.checkpoint_policies.save_only_these_names,
        "save_from_both_policies": jax.checkpoint_policies.save_from_both_policies
    }[name]

class FlaxBaseAttn(nn.Module):
    query_dim: int
    num_attention_heads: int = 8
    heads_dim: int = 64
    dropout_rate: float = 0.0
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        inner_dim = self.heads_dim * self.num_attention_heads
        self.scale = self.heads_dim ** -0.5
        self.q = nn.Dense(inner_dim,
                          dtype=self.dtype,
                          param_dtype=self.param_dtype,
                          precision=self.precision,
                          kernel_init=jax.nn.initializers.normal(),
                          use_bias=False)
        self.k = nn.Dense(inner_dim,
                          dtype=self.dtype,
                          param_dtype=self.param_dtype,
                          precision=self.precision,
                          kernel_init=jax.nn.initializers.normal(),
                          use_bias=False)
        self.v = nn.Dense(inner_dim,
                          dtype=self.dtype,
                          param_dtype=self.param_dtype,
                          precision=self.precision,
                          kernel_init=jax.nn.initializers.normal(),
                          use_bias=False)
        self.out = nn.Dense(
            self.query_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(),
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def split(self, x):
        # print(x.shape)
        batch, sq, hidden_size = x.shape
        x = x.reshape(batch, sq, self.num_attention_heads, hidden_size // self.num_attention_heads)
        x = jnp.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch * self.num_attention_heads, sq, hidden_size // self.num_attention_heads)

    def merge(self, x):
        batch, sq, hidden_size = x.shape
        x = x.reshape(batch // self.num_attention_heads, self.num_attention_heads, sq, hidden_size)
        x = jnp.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch // self.num_attention_heads, sq, self.num_attention_heads * hidden_size)

    def __call__(self, hidden_state,
                 context = None,
                 deterministic = False):
        context = hidden_state if context is None else context
        # context = jax.lax.with_sharding_constraint(context, PartitionSpec(('dp', 'fsdp')))
        # hidden_state = jax.lax.with_sharding_constraint(hidden_state, PartitionSpec(('dp', 'fsdp')))
        q = self.q(hidden_state)
        v = self.v(context)
        k = self.k(context)
        q, k, v = self.split(q), self.split(k), self.split(v)
        attn = jax.nn.softmax(jnp.einsum('b i d,b j d-> b i j', q, k) * self.scale, axis=-1)
        attn = self.merge(jnp.einsum('b i j,b j d -> b i d', attn, v))
        return self.dropout(self.out(attn), deterministic=deterministic)


class FlaxFeedForward(nn.Module):
    features: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self):
        self.net_0 = FlaxGEGLU(features=self.features,
                               dropout_rate=self.dropout_rate,
                               dtype=self.dtype,
                               param_dtype=self.param_dtype,
                               precision=self.precision)
        self.net_2 = nn.Dense(self.features,
                              dtype=self.dtype,
                              param_dtype=self.param_dtype,
                              precision=self.precision)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.net_0(hidden_states, deterministic=deterministic)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class FlaxGEGLU(nn.Module):
    features: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self):
        inner_features = self.features * 4
        self.proj = nn.Dense(inner_features * 2,
                             dtype=self.dtype,
                             param_dtype=self.param_dtype,
                             precision=self.precision)
        self.dropout_layer = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
        return self.dropout_layer(hidden_linear * nn.gelu(hidden_gelu), deterministic=deterministic)


class FlaxEncoderBaseTransformerBlock(nn.Module):
    features: int
    num_attention_heads: int
    heads_dim: int
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    only_cross_attn: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:
        self.attn1 = FlaxBaseAttn(
            self.features,
            num_attention_heads=self.num_attention_heads,
            heads_dim=self.heads_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.attn2 = FlaxBaseAttn(
            self.features,
            num_attention_heads=self.num_attention_heads,
            dropout_rate=self.dropout_rate,
            heads_dim=self.heads_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.ff = FlaxFeedForward(
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            dropout_rate=self.dropout_rate,
            precision=self.precision
        )
        self.norm1 = nn.LayerNorm(epsilon=self.epsilon,
                                  dtype=self.dtype,
                                  param_dtype=self.param_dtype)
        self.norm2 = nn.LayerNorm(epsilon=self.epsilon,
                                  dtype=self.dtype,
                                  param_dtype=self.param_dtype)
        self.norm3 = nn.LayerNorm(epsilon=self.epsilon,
                                  dtype=self.dtype,
                                  param_dtype=self.param_dtype)
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def __call__(self, hidden_state, context, deterministic: bool = True):

        if self.only_cross_attn:
            hidden_state = self.attn1(self.norm1(hidden_state), context=context,
                                      deterministic=deterministic) + hidden_state
        else:
            hidden_state = self.attn1(self.norm1(hidden_state), context=None,
                                      deterministic=deterministic) + hidden_state

        hidden_state = self.attn2(self.norm2(hidden_state), context=context, deterministic=deterministic) + hidden_state
        hidden_state = self.ff(self.norm3(hidden_state), deterministic=deterministic) + hidden_state
        return self.dropout_layer(hidden_state, deterministic=deterministic)


class FlaxEncoderBaseTransformerBlockCollection(nn.Module):
    features: int
    num_attention_heads: int
    heads_dim: int
    num_hidden_layers: int
    dropout_rate: float
    epsilon: float
    only_cross_attn: bool
    gradient_checkpointing: str
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:

        self.blocks = [
            FlaxEncoderBaseTransformerBlock(
                features=self.features,
                heads_dim=self.heads_dim,
                num_attention_heads=self.num_attention_heads,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i)
            ) for i in range(self.num_hidden_layers)
        ]

    def __call__(self, hidden_state, context, deterministic: bool = True):
        for block in self.blocks:
            hidden_state = block(
                hidden_state=hidden_state,
                context=context,
                deterministic=deterministic
            )
        return hidden_state


class FlaxTransformerBlock2D(nn.Module):
    num_attention_heads: int
    heads_dim: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    only_cross_attn: bool = False
    use_linear_proj: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self) -> None:

        features = self.heads_dim * self.num_attention_heads
        self.norm = nn.GroupNorm(
            32, epsilon=self.epsilon
        )
        if self.use_linear_proj:
            self.proj_in = nn.Dense(
                features,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        else:
            self.proj_in = nn.Conv(
                features,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='VALID',
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        self.blocks = FlaxEncoderBaseTransformerBlockCollection(
            features=features,
            num_attention_heads=self.num_attention_heads,
            heads_dim=self.heads_dim,
            num_hidden_layers=self.num_hidden_layers,
            dropout_rate=self.dropout_rate,
            epsilon=self.epsilon,
            only_cross_attn=self.only_cross_attn,
            gradient_checkpointing=self.gradient_checkpointing
        )
        if self.use_linear_proj:
            self.proj_out = nn.Dense(
                features,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        else:
            self.proj_out = nn.Conv(
                features,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='VALID',
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        self.dropout_layer = nn.Dropout(
            self.dropout_rate
        )

    def __call__(self, hidden_state, context, deterministic: bool = True):
        batch, height, width, channels = hidden_state.shape
        residual = hidden_state
        hidden_state = self.norm(hidden_state)
        if self.use_linear_proj:
            hidden_state = self.proj_in(
                hidden_state.reshape(batch, height * width, channels)
            )
        else:
            hidden_state = self.proj_in(
                hidden_state
            ).reshape(batch, height * width, channels)
        hidden_state = self.blocks(hidden_state=hidden_state, context=context, deterministic=deterministic)
        if self.use_linear_proj:

            hidden_state = self.proj_out(
                hidden_state
            ).reshape(batch, height, width, channels)
        else:
            hidden_state = self.proj_in(
                hidden_state.reshape(batch, height, width, channels)
            )
        return self.dropout_layer(hidden_state + residual, deterministic=deterministic)


class FlaxAttentionBlock(nn.Module):
    channels: int
    num_attention_heads = None
    num_groups: int = 32
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self):
        self.num_heads = self.channels // self.num_attention_heads if self.num_attention_heads is not None else 1

        dense = partial(nn.Dense, self.channels, dtype=self.dtype, param_dtype=self.param_dtype,
                        precision=self.precision)

        self.group_norm = nn.GroupNorm(num_groups=self.num_groups, epsilon=1e-6)
        self.query, self.key, self.value = dense(), dense(), dense()
        self.proj_attn = dense()

    def transpose_for_scores(self, projection):
        new_projection_shape = projection.shape[:-1] + (self.num_heads, -1)
        new_projection = projection.reshape(new_projection_shape)
        new_projection = jnp.transpose(new_projection, (0, 2, 1, 3))
        return new_projection

    def __call__(self, hidden_states):
        residual = hidden_states
        batch, height, width, channels = hidden_states.shape
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.reshape((batch, height * width, channels))
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attn_weights = jnp.einsum("...qc,...kc->...qk", query * scale, key * scale)
        attn_weights = nn.softmax(attn_weights, axis=-1)
        hidden_states = jnp.einsum("...kc,...qk->...qc", value, attn_weights)
        hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))
        new_hidden_states_shape = hidden_states.shape[:-2] + (self.channels,)
        hidden_states = hidden_states.reshape(new_hidden_states_shape)
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.reshape((batch, height, width, channels))
        hidden_states = hidden_states + residual
        return hidden_states


class Upsample(nn.Module):
    in_channels: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
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
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(self, hidden_states):
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        hidden_states = jnp.pad(hidden_states, pad_width=pad)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class BaseOutput(OrderedDict):
    """
    from HuggingFace
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and isinstance(first_field, dict):
            for key, value in first_field.items():
                self[key] = value
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):

        super().__setitem__(key, value)

        super().__setattr__(key, value)

    def to_tuple(self):

        return tuple(self[k] for k in self.keys())

class FlaxDownBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    add_downsampler: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self) -> None:
        resnet = []

        for index in range(self.num_hidden_layers):
            in_channels = self.in_channels if index == 0 else self.out_channels
            res_n = FlaxResnetBlock2D(
                in_c=in_channels,
                out_c=self.out_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

            resnet.append(res_n)

        self.resnets = resnet
        if self.add_downsampler:
            self.downsamplers_0 = Downsample(
                self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self, hidden_state, time, deterministic: bool = True):
        output_states = []
        for resnet in self.resnets:
            hidden_state = resnet(hidden_state=hidden_state, time=time, deterministic=deterministic)
            output_states.append(hidden_state)
        if self.add_downsampler:
            hidden_state = self.downsamplers_0(hidden_state)
            output_states.append(hidden_state)
        return hidden_state, output_states


class FlaxCrossAttnUpBlock(nn.Module):
    in_channels: int
    out_channels: int
    perv_out_channels: int
    num_attention_heads: int
    num_hidden_layers: int = 2
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    only_cross_attn: bool = False
    use_linear_proj: bool = False
    add_upsampler: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self) -> None:
        resnet = []
        attentions = []
        resnet_block = nn.remat(FlaxResnetBlock2D,
                                policy=get_gradient_checkpointing_policy(
                                    name=self.gradient_checkpointing)) \
            if self.gradient_checkpointing != '' else FlaxResnetBlock2D

        attention_block = nn.remat(FlaxTransformerBlock2D,
                                   policy=get_gradient_checkpointing_policy(
                                       name=self.gradient_checkpointing)) \
            if self.gradient_checkpointing != '' else FlaxTransformerBlock2D

        for index in range(self.num_hidden_layers):
            in_channel = self.in_channels if (index == self.num_hidden_layers - 1) else self.out_channels
            resnet_skip_in_channel = self.perv_out_channels if index == 0 else self.out_channels
            res_n = resnet_block(
                in_c=in_channel + resnet_skip_in_channel,
                out_c=self.out_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            resnet.append(res_n)
            atn_n = attention_block(
                num_hidden_layers=1,
                heads_dim=self.out_channels // self.num_attention_heads,
                num_attention_heads=self.num_attention_heads,
                use_linear_proj=self.use_linear_proj,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                only_cross_attn=self.only_cross_attn,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                gradient_checkpointing=self.gradient_checkpointing
            )
            attentions.append(atn_n)

        self.resnets = resnet
        self.attentions = attentions
        if self.add_upsampler:
            self.upsampler = Upsample(
                self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self,
                 hidden_state,
                 time,
                 output_states: list,
                 encoder_hidden_states,
                 deterministic: bool = True
                 ):
        output_states = tuple(output_states)
        for res, atn in zip(self.resnets, self.attentions):
            enc = output_states[-1]
            output_states = output_states[:-1]
            hidden_state = jnp.concatenate([hidden_state, enc], axis=-1)
            hidden_state = res(hidden_state, time, deterministic=deterministic)
            hidden_state = atn(hidden_state, encoder_hidden_states, deterministic=deterministic)
        if self.add_upsampler:
            hidden_state = self.upsampler(hidden_state)
        return hidden_state


class FlaxUpBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    perv_out_channels: int
    num_attention_heads: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    only_cross_attn: bool = False
    use_linear_proj: bool = False
    add_upsampler: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self) -> None:
        resnet = []
        resnet_block = nn.remat(FlaxResnetBlock2D,
                                policy=get_gradient_checkpointing_policy(
                                    name=self.gradient_checkpointing)) \
            if self.gradient_checkpointing != '' else FlaxResnetBlock2D

        for index in range(self.num_hidden_layers):
            in_channel = self.in_channels if (index == self.num_hidden_layers - 1) else self.out_channels
            resnet_skip_in_channel = self.perv_out_channels if index == 0 else self.out_channels
            res_n = resnet_block(
                in_c=in_channel + resnet_skip_in_channel,
                out_c=self.out_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            resnet.append(res_n)

        self.resnets = resnet
        if self.add_upsampler:
            self.upsampler = Upsample(
                self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self,
                 hidden_state,
                 time,
                 output_states: list,
                 deterministic: bool = True
                 ):
        output_states = tuple(output_states)
        for res in self.resnets:
            enc = output_states[-1]
            output_states = output_states[:-1]
            hidden_state = jnp.concatenate([hidden_state, enc], axis=-1)
            hidden_state = res(hidden_state, time, deterministic=deterministic)

        if self.add_upsampler:
            hidden_state = self.upsampler(hidden_state)
        return hidden_state


class FlaxUNetMidBlock2DCrossAttn(nn.Module):
    in_channels: int
    num_attention_heads: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    only_cross_attn: bool = False
    use_linear_proj: bool = False
    add_upsampler: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self):
        attention_block = FlaxTransformerBlock2D
        resnet_block = FlaxResnetBlock2D
        # attention_block = nn.remat(
        #     FlaxTransformerBlock2D,
        #     policy=get_gradient_checkpointing_policy(self.gradient_checkpointing)
        # )
        # resnet_block = nn.remat(
        #     FlaxResnetBlock2D,
        #     policy=get_gradient_checkpointing_policy(self.gradient_checkpointing)
        # )
        resnets = [
            resnet_block(
                in_c=self.in_channels,
                out_c=self.in_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        ]

        attentions = []

        for _ in range(self.num_hidden_layers):
            attn_block = attention_block(
                num_hidden_layers=1,
                heads_dim=self.in_channels // self.num_attention_heads,
                num_attention_heads=self.num_attention_heads,
                use_linear_proj=self.use_linear_proj,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                only_cross_attn=self.only_cross_attn,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                gradient_checkpointing=self.gradient_checkpointing
            )
            attentions.append(attn_block)

            res_block = resnet_block(
                in_c=self.in_channels,
                out_c=self.in_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self,
                 hidden_states,
                 time,
                 encoder_hidden_states,
                 deterministic=True
                 ):
        hidden_states = self.resnets[0](hidden_states, time)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)
            hidden_states = resnet(hidden_states, time, deterministic=deterministic)

        return hidden_states


class FlaxUNetMidBlock2D(nn.Module):
    in_channels: int
    num_attention_heads: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self):
        resnets = [
            FlaxResnetBlock2DNTime(
                in_c=self.in_channels,
                out_c=self.in_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        ]

        attentions = []

        for _ in range(self.num_hidden_layers):
            attn_block = FlaxAttentionBlock(
                channels=self.in_channels,
                num_attention_heads=self.num_attention_heads,
                num_groups=32,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            attentions.append(attn_block)

            res_block = FlaxResnetBlock2DNTime(
                in_c=self.in_channels,
                out_c=self.in_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self,
                 hidden_states,
                 deterministic=True
                 ):
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        return hidden_states




class Upsample(nn.Module):
    in_channels: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
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
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(self, hidden_states):
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        hidden_states = jnp.pad(hidden_states, pad_width=pad)
        hidden_states = self.conv(hidden_states)
        return hidden_states



class Block(nn.Module):
    in_channels: int
    out_channels: int
    num_attention_heads: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    use_linear_proj: bool = False
    add_downsampler: bool = False
    add_upsampler: bool = False
    attention: boolean = False
    perv_out_channels: int = None


    def setup(self) -> None:
        resnets = []
        attentions = []



        for index in range(self.num_hidden_layers):
            in_channels = self.in_channels if index == 0 else self.out_channels
            res_n = FlaxResnetBlock2D(
                in_c=in_channels,
                out_c=self.out_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            atn_n = FlaxTransformerBlock2D(
                num_hidden_layers=1,
                heads_dim=self.out_channels // self.num_attention_heads,
                num_attention_heads=self.num_attention_heads,
                use_linear_proj=self.use_linear_proj,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                only_cross_attn=self.only_cross_attn,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                gradient_checkpointing=self.gradient_checkpointing
            )
            attentions.append(atn_n)
            resnets.append(res_n)
        self.attentions = attentions
        self.resnets = resnets
        if self.add_downsampler:
            self.downsamplers_0 = Downsample(
                self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        if self.add_upsampler:
            self.upsamplers_0 = Upsample(
                self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self, x, c, skip):
        output_states = ()
        for resnet, attention in zip(self.resnets, self.attentions):
            hidden_state = resnet(hidden_state=hidden_state, time=time, deterministic=deterministic)
            hidden_state = attention(hidden_state=hidden_state, context=encoder_hidden_states,
                                     deterministic=deterministic)
            output_states += (hidden_state,)
        if self.add_downsampler:
            hidden_state = self.downsamplers_0(hidden_state)
            output_states += (hidden_state,)
        if self.add_upsampler:
            hidden_state = self.upsamplers_0(hidden_state)
            output_states += (hidden_state,)
        return hidden_state, output_states



class Unet2DConditionModel(nn.Module):
    sample_size: int = 32
    in_channels: int = 4
    out_channels: int = 4

    down_layers = ("attention", "attention", "attention", "attention", "resnet")
    up_layers = ("resnet", "attention", "attention", "attention", "attention")
    channels = (320, 640, 640, 1280, 1280)
    num_hidden_layers_per_block: int = 2

    depth: int
    num_heads: int
    mlp_ratio: float
    class_dropout_prob: float
    num_classes: int


    def init_weights(self, rng):

        sample = jnp.zeros((1, self.in_channels, self.sample_size, self.sample_size), dtype=self.dtype)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=self.dtype)

        params_rng, dropout_rng = jax.random.split(rng)

        return self.init({"params": params_rng, "dropout": dropout_rng}, sample, timesteps, encoder_hidden_states)[
            "params"]

    def setup(self) -> None:
        assert len(self.down_block_types) == len(self.up_block_types)
        embedding_dimension = self.channels[0] * 4
        num_attention_heads = self.num_attention_heads
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
                use_linear_proj=self.use_linear_proj,
                add_downsampler=not is_final_b,
                num_hidden_layers=self.num_hidden_layers_per_block,
                in_channels=in_channels,
                attention= name=="attention",
                out_channels=output_channel,
                add_upsampler=False,
            )
            down_blocks.append(block)
        self.bottle_neck = Block(
            in_channels=self.block_out_channels[-1],
            out_channels=self.block_out_channels[-1],
            num_attention_heads=num_attention_heads[-1],
            use_linear_proj=self.use_linear_proj,
            add_upsampler=False,
            add_downsampler=False
        )
        up_blocks: typing.List[Block] = []
        reversed_block_out_channels = list(self.block_out_channels[::-1])
        reversed_num_attention_heads = list(num_attention_heads[::-1])
        output_channel = reversed_block_out_channels[0]
        for i, name in enumerate(self.up_block_types):
            perv_output_channel = output_channel
            in_channels = reversed_block_out_channels[min(i + 1, len(self.block_out_channels) - 1)]
            output_channel = reversed_block_out_channels[i]
            is_final_b = i == len(reversed_block_out_channels) - 1
            block = Block(
                in_channels=in_channels,
                out_channels=output_channel,
                perv_out_channels=perv_output_channel,
                num_attention_heads=reversed_num_attention_heads[i],
                num_hidden_layers=self.num_hidden_layers_per_block + 1,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                use_linear_proj=self.use_linear_proj,
                add_upsampler=not is_final_b,
                add_downsampler=False,
                attention = name=="attention"
            )
            up_blocks.append(block)

        self.norm_out = nn.GroupNorm(32, epsilon=self.epsilon)
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

        # x = (B, H, W, C)
        # t = (B,)
        # y = (B,)

        t = self.time_proj(t)
        y = self.label_embedder(y, train=train, force_drop_ids=force_drop_ids)
        c = t + y
        x = self.conv_in(x)
        hs = (x,)
        for block in self.down_blocks:
            hidden_states, res_hidden_states = block(x, c)
            hs += tuple(res_hidden_states)

        hidden_states = self.bottle_neck(
            x,
            c
        )[0]


        for block in self.up_blocks:
            res_hidden_states = hs[-(self.num_hidden_layers_per_block + 1):]
            down_block_res_hidden_states = hs[: -(self.num_hidden_layers_per_block + 1)]
            hidden_states = block(
                x,
                c,
                skip=res_hidden_states[0]
            )[0]

        hidden_states = self.conv_out(nn.swish(self.norm_out(hidden_states)))
        return hidden_states

# # we have 4 classes of models, 10M, 50M, 100M, 500M
def UNet10M(patch_size, num_classes, class_dropout_prob):
    model = Unet2DConditionModel(
        in_channels=4,
        out_channels=4,
        num_hidden_layers_per_block=1,
        block_out_channels=(32, 64, 128, 128, 128),
        num_attention_heads=1,
        use_linear_proj=False,
        dropout_rate=0.1,
        epsilon=1e-6,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        gradient_checkpointing=""
    )
    return model

# try
#
def test_flax_unet():
    hidden_states = jnp.ones((1, 4, 16, 16))
    encoder_hidden_states = jnp.ones((1, 11, 768))
    timestep = jnp.ones((1, ))
    model = Unet2DConditionModel(
        in_channels=4,
        out_channels=4,
        num_hidden_layers_per_block=1,
        block_out_channels=(32, 64, 128, 128, 128),
        num_attention_heads=1,
        use_linear_proj=False,
        dropout_rate=0.1,
        epsilon=1e-6,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        gradient_checkpointing=""
    )
    params = model.init(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep, rngs=jax.random.PRNGKey(0))
    out = model.apply(params, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep, rngs=jax.random.PRNGKey(0))
    assert out.sample.shape == (1, 4, 16, 16)
    # number of params
    num_params = sum([p.size for p in jax.tree_flatten(params)[0]])
    print(f"Number of parameters in UNet2DConditionModel: {num_params}")

if __name__ == '__main__':
    test_flax_unet()
