import math
import json
import re
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
from typing import Callable, Tuple
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

import xformers.ops as xops

from huggingface_hub import PyTorchModelHubMixin

from lightning_attn import lightning_attn_func as lightning_attn_ops

from open_lm.attention import get_attn_func, xformers_attn, torch_attn
from open_lm.norms import get_norm_class
from open_lm.positional_embedding.head_rotary import HeadRotaryWithCast
from open_lm.positional_embedding.rotary import RotaryWithCast
from open_lm.positional_embedding.llama_rotary import LLaMARotaryWithCast
from open_lm.positional_embedding.none import identity_with_cast

# from open_lm.moe.mixture_of_experts import MoE
try:
    from megablocks.layers.moe import MoE
    from megablocks.layers.arguments import Arguments as MoEArgs
except ImportError:
    import logging

    logging.warning(f"Megablocks not installed. To train MoE, install with pip install megablocks.")

try:  # optional import
    from mamba_ssm import MambaLMHeadModel
except ImportError:
    MambaLMHeadModel = None

# from openclip
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs(model_config_paths=None):
    global _MODEL_CONFIGS

    config_iter = None
    if model_config_paths is not None:
        config_iter = [
            Path(model_config_paths),
        ]
    else:
        config_iter = _MODEL_CONFIG_PATHS

    config_ext = (".json",)
    config_files = []
    for config_path in config_iter:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(Path(config_path))
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


# args and default params follow llama (except with LayerNorm instead of RmsNorm)
@dataclass
class Params:
    dim: int = 512
    qk_head_dim: int = None
    v_head_dim: int = None
    intermediate_dim_ffn: int = None
    n_layers: int = 8
    n_heads: int = 8
    n_heads_kv: int = 8
    vocab_size: int = -1
    norm_eps: float = 1e-5
    seq_len: int = 2048
    post_embed_norm: bool = False
    weight_tying: bool = False
    norm_type: nn.Module = nn.LayerNorm
    attn_func: Callable = xformers_attn if torch.cuda.is_available() else torch_attn
    attn_name: str = "xformers_attn"
    apply_qk_norm: bool = False
    moe_loss_weight: float = 0.1
    moe_capacity_factor: float = 1.25
    moe_expert_model_parallelism: bool = False
    moe_weight_parallelism: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_freq: int = 0
    positional_embedding_type: str = "rotary"
    ffn_type: str = "swiglu"
    decay_start: float = None
    rotary_base_frequency: int = 10000
    rotary_scale: float = 1.0
    use_retnet_slopes: bool = False
    use_decay: bool = False


def get_pos_embed(args: Params):
    head_dim = args.dim // args.n_heads
    if args.positional_embedding_type == "rotary":
        return RotaryWithCast(head_dim, args.seq_len, args.rotary_base_frequency, args.rotary_scale)
    elif args.positional_embedding_type == "llama_rotary":
        return LLaMARotaryWithCast(head_dim, args.n_heads, args.seq_len, args.rotary_base_frequency, args.rotary_scale)
    elif args.positional_embedding_type == "head_rotary":
        return HeadRotaryWithCast(head_dim, args.seq_len)
    elif args.positional_embedding_type == "none":
        return identity_with_cast
    else:
        raise RuntimeError(f"Unknown positional embedding type {args.positional_embedding_type}")


class CustomAttn(nn.Module):
    def __init__(self, layer_id, args: Params):
        super().__init__()
        self.n_heads = args.n_heads
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.n_heads_kv = args.n_heads_kv
        self.in_proj = nn.Linear(
            args.dim,
            (args.n_heads * self.qk_head_dim + self.n_heads_kv * self.qk_head_dim + self.n_heads_kv * self.v_head_dim),
            bias=False,
        )
        self.out_proj = nn.Linear(args.n_heads * self.v_head_dim, args.dim, bias=False)
        self.pos_embed = get_pos_embed(args)
        self.attn_fn = args.attn_func
        self.apply_qk_norm = args.apply_qk_norm

        # initialize norm layers for queries and keys if needed
        NormClass = args.norm_type
        self.q_norm = (
            NormClass(
                args.n_heads * self.qk_head_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            NormClass(
                args.n_heads * self.qk_head_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )

        self.layer_id = layer_id
        self.dim = args.dim
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weights by trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = std / math.sqrt(2 * (self.layer_id + 1))
        torch.nn.init.trunc_normal_(self.out_proj.weight, std=std, a=-3 * std, b=3 * std)

    def repeat_kv(self, hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states
        hidden_states2 = hidden_states.transpose(1, 2)
        batch, num_key_value_heads, slen, head_dim = hidden_states2.shape
        hidden_states2 = hidden_states2[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states2.reshape(batch, num_key_value_heads * n_rep, slen, head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, is_causal=True, past_key_value=None, use_cache=False, attention_mask=None):
        batchsize, q_len, _ = x.shape
        queries, keys, vals = self.in_proj(x).split(
            [self.n_heads * self.qk_head_dim, self.n_heads_kv * self.qk_head_dim, self.n_heads_kv * self.v_head_dim],
            dim=-1,
        )

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.view(batchsize, q_len, self.n_heads, self.qk_head_dim)
        keys = keys.view(batchsize, q_len, self.n_heads_kv, self.qk_head_dim)
        vals = vals.view(batchsize, q_len, self.n_heads_kv, self.v_head_dim)

        past_length = 0 if past_key_value is None else past_key_value[0].shape[1]
        queries, keys, vals = self.pos_embed(queries, keys, vals, offset=past_length)

        keys = self.repeat_kv(keys, n_rep=self.n_heads // self.n_heads_kv)
        vals = self.repeat_kv(vals, n_rep=self.n_heads // self.n_heads_kv)

        if past_key_value is not None and use_cache:
            keys = torch.cat([past_key_value[0], keys], dim=1)
            vals = torch.cat([past_key_value[1], vals], dim=1)

        if use_cache:
            past_key_value = [keys, vals]

        output = self.attn_fn(
            queries,
            keys,
            vals,
            is_causal=is_causal,
            attention_mask=attention_mask,
        )

        output = output.view(batchsize, q_len, -1)

        return self.out_proj(output), past_key_value


############### Linear attention addition ################


def no_slope_tensor(n_attention_heads: int, device: torch.device, dtype: torch.dtype):
    """
    This function returns a tensor of zeros, which is equivalent to not using any decay.
    n_attention_heads: number of attention heads
    device: device to use
    dtype: data type to use
    """
    slopes = torch.zeros(n_attention_heads, 1, 1, device=device, dtype=dtype)

    return slopes


def get_slopes_power_of_2(n, start):
    """
    This function returns a list of slopes for the linear attention function given a power of 3 number of heads.
    It is taken from the lightning attention code.
    n: number of attention heads
    start: (optional) start value for the slope tensor
    """
    ratio = 2 ** (-(2 ** -(math.log2(n) - 3)))
    if start is None:
        start = ratio
    return [start * ratio ** i for i in range(n)]


def get_slopes(n, start):
    """
    This function returns a list of slopes for the linear attention function.
    It is taken from the lightning attention code.
    n: number of attention heads
    start: (optional) start value for the slope tensor
    """
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(
            n, start
        )  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return (
            get_slopes_power_of_2(closest_power_of_2, start)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def get_slope_tensor(
    n_attention_heads: int,
    start: float = None,
    use_retnet_slopes: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    """
    This function returns a tensor of slopes for the linear attention function. This determines the decay of the attention function.
    n_attention_heads: number of attention heads
    start: (optional) start value for the slope tensor
    use_retnet_slopes: (optional) use the RetNet slopes instead of the default lightning attention ones
    device: (optional) device to use
    dtype: (optional) data type to use
    """
    if use_retnet_slopes:
        head_count = torch.arange(n_attention_heads, device=device, dtype=dtype)
        gamma = 1 - torch.exp2(-5 - head_count.float())
        slopes = -torch.log(gamma.unsqueeze(-1))
    else:
        # h, 1, 1
        slopes = torch.tensor(get_slopes(n_attention_heads, start), dtype=dtype, device=device).reshape(
            n_attention_heads,
            1,
        )
    return slopes


def recurrent_forward(
    queries, keys, vals, s, qk_scale=1, start=None, use_decay=False, use_retnet_slopes=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function computes the output of the linear attention function in a recurrent manner.
    Its result is equivalent to the parallel computation of the linear attention function.
    queries: queries, shape (batch_size, num_heads, seq_len, dim_qk)
    keys: keys, shape (batch_size, num_heads, seq_len, dim_qk)
    vals: values, shape (batch_size, num_heads, seq_len, dim_v)
    s: current state of the RNN, shape (batch_size, num_heads, dim_qk, dim_v)
    use_decay: (optional) use the decaying factor on the distance between queries and keys
    use_retnet_slopes: (optional) use the RetNet slopes instead of the default lightning attention ones
    qk_scale: scale factor for queries and keys
    start: (optional) start value for the slope tensor in case decay is used
    """
    if use_decay:
        slope = get_slope_tensor(queries.shape[1], start, use_retnet_slopes, queries.device, queries.dtype)
        gamma = torch.exp(-slope).reshape(1, queries.shape[1], 1, 1)
    else:
        gamma = 1.0
    s_n = s + (keys.transpose(-1, -2) * qk_scale) @ vals
    output = queries @ s_n
    return output, gamma * s_n


def lightning_attn_func(
    q, k, v, qk_scale: float, start: float = None, use_decay: bool = True, use_retnet_slopes=False
) -> torch.Tensor:
    """
    This is the lightning attention function, which is a kernel linear approximation of the softmax function
    Almost the same as linear_attn_func but using the triton kernel from lightning_attn and a decaying factor (from RetNet paper https://arxiv.org/pdf/2307.08621.pdf)
    as defined by the depth_slope_tensor function (using no_slope_tensor is equivalent to linear_attn_func).
    Args:
        q: queries, shape (batch_size, num_heads, seq_len, dim_qk)
        k: keys, shape (batch_size, num_heads, seq_len, dim_qk)
        v: values, shape (batch_size, num_heads, seq_len, dim_v)
        qk_scale: scale factor for queries and keys
        start: (optional) start value for the slope tensor in case decay is used
        use_decay: (optional) use the decaying factor on the distance between queries and keys
        use_retnet_slopes: (optional) use the RetNet slopes instead of the default lightning attention ones
    """
    h = q.shape[1]
    if use_decay:
        s = get_slope_tensor(h, start, use_retnet_slopes, q.device, torch.float32)
    else:
        s = no_slope_tensor(h, q.device, q.dtype)
    output = lightning_attn_ops(q, k * qk_scale, v, s)

    return output


class LinearAttn(nn.Module):
    """
    This class implements the linear attention layer.
    It can be used as a drop-in replacement for the CustomAttn class.
    The forward method can be run in parallel or recurrent mode depending on the use_cache parameter,
    which folows the same logic as the CustomAttn class with qk_cache or without it.
    """

    def __init__(self, layer_id, args: Params):
        super().__init__()
        self.params = args
        self.n_heads = args.n_heads
        self.n_heads_kv = args.n_heads_kv
        self.qk_head_dim = args.qk_head_dim
        self.qk_in_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim

        self.qk_in_head_dim = args.dim // args.n_heads
        self.in_proj = nn.Linear(
            args.dim,
            (
                args.n_heads * self.qk_in_head_dim
                + self.n_heads_kv * self.qk_in_head_dim
                + self.n_heads_kv * self.v_head_dim
            ),
            bias=False,
        )

        self.out_proj = nn.Linear(args.n_heads * self.v_head_dim, args.dim, bias=False)

        self.pos_embed = get_pos_embed(args)

        self.apply_qk_norm = args.apply_qk_norm

        self._totrain_gn = nn.GroupNorm(
            num_groups=args.n_heads, num_channels=args.n_heads * self.v_head_dim, affine=False
        )

        self._totrain_embed = nn.Linear(
            args.n_heads * self.qk_in_head_dim,
            args.n_heads * self.qk_head_dim,
        )

        if args.n_heads != args.n_heads_kv:
            self._totrain_embed_kv = nn.Linear(
                args.n_heads_kv * self.qk_in_head_dim,
                args.n_heads_kv * self.qk_head_dim,
            )

        self.linear_attn_fn = partial(
            lightning_attn_func,
            use_decay=args.use_decay,
            use_retnet_slopes=args.use_retnet_slopes,
            start=args.decay_start,
        )
        self.recurrent_forward_fn = partial(
            recurrent_forward,
            use_decay=args.use_decay,
            use_retnet_slopes=args.use_retnet_slopes,
            start=args.decay_start,
        )

        self.mask = None

        # initialize weights by trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(args.dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)

        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(self.out_proj.weight, std=std, a=-3 * std, b=3 * std)

        norm_dim = self.n_heads * self.qk_head_dim

        # initialize norm layers for queries and keys if needed
        NormClass = args.norm_type
        self._totrain_q_norm = (
            NormClass(
                norm_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self._totrain_k_norm = (
            NormClass(
                norm_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self.qk_scale = 1.0 / math.sqrt(self.qk_head_dim)

    def repeat_kv(self, hidden_states, n_rep):
        """
        This function repeats the key and value tensors to match the number of queries.
        This is needed when the number of key-value heads is different from the number of query heads (GQA or MQA).
        """
        if n_rep == 1:
            return hidden_states
        hidden_states2 = hidden_states.transpose(1, 2)
        batch, num_key_value_heads, slen, head_dim = hidden_states2.shape
        hidden_states2 = hidden_states2[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states2.reshape(batch, num_key_value_heads * n_rep, slen, head_dim).transpose(1, 2)

    def _set_mask(self, seqlen: int, device):
        if self.mask is None or self.mask.shape[-1] < seqlen:
            self.mask = torch.tril(torch.ones(1, 1, seqlen, seqlen, requires_grad=False), diagonal=0).to(device)

    def _get_qkv(self, x: torch.Tensor, offset=0):
        """
        This function computes the queries, keys, and values for the linear attention function.
        It re-uses the projection layer from a usual transformer model and applies the kernels to the queries and keys (one layer + relu).
        """
        batchsize, seqlen, _ = x.shape
        queries, keys, vals = self.in_proj(x).split(
            [
                self.n_heads * self.qk_in_head_dim,
                self.n_heads_kv * self.qk_in_head_dim,
                self.n_heads_kv * self.v_head_dim,
            ],
            dim=-1,
        )
        vals = vals.view(batchsize, seqlen, self.n_heads_kv, self.v_head_dim)

        queries = F.relu(self._totrain_embed(queries.view(batchsize, seqlen, self.n_heads * self.qk_in_head_dim))).view(
            batchsize, seqlen, self.n_heads, self.qk_head_dim
        )
        if self.n_heads != self.n_heads_kv:
            keys = F.relu(
                self._totrain_embed_kv(keys.view(batchsize, seqlen, self.n_heads_kv * self.qk_in_head_dim))
            ).view(batchsize, seqlen, self.n_heads_kv, self.qk_head_dim)
        else:
            keys = F.relu(
                self._totrain_embed(keys.view(batchsize, seqlen, self.n_heads_kv * self.qk_in_head_dim))
            ).view(batchsize, seqlen, self.n_heads_kv, self.qk_head_dim)

        keys = self.repeat_kv(keys, n_rep=self.n_heads // self.n_heads_kv)
        vals = self.repeat_kv(vals, n_rep=self.n_heads // self.n_heads_kv)

        queries = self._totrain_q_norm(queries.view(batchsize, seqlen, self.n_heads * self.qk_head_dim)).view(
            batchsize, seqlen, self.n_heads, self.qk_head_dim
        )
        keys = self._totrain_k_norm(keys.reshape(batchsize, seqlen, self.n_heads * self.qk_head_dim)).view(
            batchsize, seqlen, self.n_heads, self.qk_head_dim
        )

        queries, keys, vals = self.pos_embed(
            queries,
            keys,
            vals,
            offset=offset,
        )

        queries = queries.transpose(1, 2).contiguous()
        keys = keys.transpose(1, 2).contiguous()
        vals = vals.transpose(1, 2).contiguous()
        return queries, keys, vals

    def _output(self, output: torch.Tensor):
        """
        This function computes the output of the linear attention function.
        It applies the group normalization and the output projection layer.
        """
        output = output.transpose(1, 2).contiguous()
        batchsize, seqlen = output.shape[:2]

        output = self._totrain_gn(output.reshape(batchsize * seqlen, self.v_head_dim * self.n_heads))

        output = output.view(batchsize, seqlen, self.v_head_dim * self.n_heads)

        output = self.out_proj(output)

        return output

    def forward(
        self, x: torch.Tensor, is_causal: bool = True, past_key_value=None, use_cache=False, attention_mask=None
    ):
        """
        Run the linear attention function either in parallel (use_cache=False) or recurrent mode (use_cache=True).
        x: [B, T, D]
        is_causal: bool must be True
        past_key_value: None or tuple of (state, offset), this is a hack to repurpose the key_value cache for recurrent inference
        use_cache: bool if set to true, run the model in recurrent mode, else run parallel mode
        attention_mask: None,
        """
        assert is_causal, "LinearAttn class only supports causal mode"
        if attention_mask is not None and attention_mask.all():
            attention_mask = None
        increment = x.shape[1] if attention_mask is None else attention_mask.sum(dim=1)

        if not use_cache:
            output = self.forward_parallel(x, is_causal, attention_mask=attention_mask)
        else:
            if past_key_value is None:
                past_key_value = (None, 0)
            output, s = self.forward_recurrent(x, past_key_value[0], past_key_value[1])
            past_key_value = (s, past_key_value[1] + increment)

        return output, past_key_value

    def forward_parallel(self, x: torch.Tensor, causal, attention_mask=None):
        """
        Use the linear attention function to compute the output in parallel.
        x: [B, T, D]
        causal: bool must be True
        attention_mask: None, not supported for linear attention in parallel mode
        """
        assert attention_mask is None, "Attention mask not supported for linear attention"
        queries, keys, vals = self._get_qkv(x)
        output = self.linear_attn_fn(queries, keys, vals, self.qk_scale)
        return self._output(output)

    def forward_recurrent(
        self,
        x: torch.Tensor,
        s: torch.Tensor = None,
        offset=0,
    ):
        """
        Use the linear attention function to compute the output in recurrent mode.
        Loops over the sequence length and computes the output and the state update at each step.
        x: [B, sequence_length, D] input features
        s: [B, head, h_dim, h_dim] (optional) input recurrent state
        offset: int or [B,] (optional) sequence offset for positional embedding, encodes the current position of x in the sequence.
        """
        if s is None:
            s = torch.zeros(
                x.shape[0],
                self.n_heads,
                self.qk_head_dim,
                self.v_head_dim,
                device=x.device,
                dtype=x.dtype,
            )
        queries, keys, vals = self._get_qkv(x, offset)

        out = []
        for i in range(x.shape[1]):
            output, s = self.recurrent_forward_fn(
                queries[:, :, i : i + 1], keys[:, :, i : i + 1], vals[:, :, i : i + 1], s, qk_scale=self.qk_scale
            )
            out.append(output)

        output = torch.cat(out, dim=2)
        return self._output(output), s


##########################################################


class GemmaMLP(nn.Module):
    """Google's Gemma model MLP (aka GeGLU).

    Modified from https://github.com/google/gemma_pytorch/blob/01062c9ef4cf89ac0c985b25a734164ede017d0b/gemma/model.py#L182-L201
    """

    def __init__(self, dim: int, hidden_dim: int, layer_id: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)
        self._layer_id = layer_id

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate)
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.gate_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.up_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self.hidden_dim)
        std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.down_proj.weight, std=std, a=-3 * std, b=3 * std)


# Same as pseudocode provided from xformers SwiGLU
# https://github.com/facebookresearch/xformers
class SwiGLUTorch(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=True):
        super().__init__()
        self.w12 = nn.Linear(in_dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x):
        gate, x = self.w12(x).chunk(2, dim=-1)
        x = F.silu(gate) * x
        return self.w3(x)


class Block(nn.Module):
    def __init__(self, layer_id, args: Params):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim

        self.head_dim = args.dim // args.n_heads

        if args.attn_name == "linear_attn":
            self.attention = LinearAttn(layer_id, args)
        else:
            self.attention = CustomAttn(layer_id, args)

        self._ffn_type = args.ffn_type
        # this follows llama / lit llama -- go to multiple of 256
        self.hidden_dim = (
            256 * ((int(2 * 4 * args.dim / 3) + 256 - 1) // 256)
            if args.intermediate_dim_ffn is None
            else args.intermediate_dim_ffn
        )
        if args.ffn_type == "swiglu":
            self.feed_forward = xops.SwiGLU(args.dim, self.hidden_dim, args.dim, bias=False)
        elif args.ffn_type == "swiglu_torch":
            self.feed_forward = SwiGLUTorch(args.dim, self.hidden_dim, args.dim, bias=False)
        elif args.ffn_type == "gelu":
            # Follows mosaic mpt7b, but without a bias.
            self.hidden_dim = args.dim * 4
            self._ff_w1 = nn.Linear(args.dim, self.hidden_dim, bias=False)
            self._ff_w2 = nn.Linear(self.hidden_dim, args.dim, bias=False)
            self.feed_forward = nn.Sequential(self._ff_w1, nn.GELU(approximate="none"), self._ff_w2)
        elif args.ffn_type == "gemma_geglu":
            self.feed_forward = GemmaMLP(args.dim, self.hidden_dim, layer_id)
        elif args.ffn_type == "moe":
            moe_args = MoEArgs(
                hidden_size=args.dim,
                ffn_hidden_size=args.dim * 4,
                moe_num_experts=args.moe_num_experts,
                moe_weight_parallelism=args.moe_weight_parallelism,
                moe_expert_model_parallelism=args.moe_expert_model_parallelism,
                moe_top_k=args.moe_top_k,
                moe_capacity_factor=args.moe_capacity_factor,
                moe_loss_weight=args.moe_loss_weight,
                device=torch.cuda.current_device(),
                bf16=False,
                fp16=False,
            )
            self.feed_forward = MoE(moe_args)

        self.layer_id = layer_id
        self.attention_norm = args.norm_type(
            args.dim,
            eps=args.norm_eps,
        )
        self.ffn_norm = args.norm_type(
            args.dim,
            eps=args.norm_eps,
        )
        self.attention.seq_len = args.seq_len
        self.reset_parameters()

    def reset_parameters(self):
        if self._ffn_type == "swiglu" or self._ffn_type == "swiglu_torch":
            # initialize weights trunc_normal(1/sqrt(fan_in))
            std = 1.0 / math.sqrt(self.dim)
            torch.nn.init.trunc_normal_(self.feed_forward.w12.weight, std=std, a=-3 * std, b=3 * std)
            # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
            std = 1.0 / math.sqrt(self.hidden_dim)
            std = std / math.sqrt(2 * (self.layer_id + 1))
            torch.nn.init.trunc_normal_(self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std)
        elif self._ffn_type == "gelu":
            std = 1.0 / math.sqrt(self.dim)
            torch.nn.init.trunc_normal_(self._ff_w1.weight, std=std, a=-3 * std, b=3 * std)

            std = 1.0 / math.sqrt(self.hidden_dim)
            std = std / math.sqrt(2 * (self.layer_id + 1))
            torch.nn.init.trunc_normal_(self._ff_w2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x, past_key_value=None, use_cache=False, attention_mask=None):
        h, past_key_value = self.attention(
            self.attention_norm(x),
            is_causal=True,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        h = x + h
        if self._ffn_type == "moe":
            ffn_out, _ = self.feed_forward(self.ffn_norm(h))
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
        out = h + ffn_out
        return out, past_key_value


class Transformer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, params):
        super().__init__()
        # for convenience we often share param names with llama
        self.params = params
        self.dim = params.dim
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.moe_num_experts = params.moe_num_experts
        self.seq_len = params.seq_len
        self.post_embed_norm = (
            params.norm_type(
                params.dim,
                eps=params.norm_eps,
            )
            if params.post_embed_norm
            else nn.Identity()
        )
        self.weight_tying = params.weight_tying

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        ffn_type_ = params.ffn_type
        for layer_id in range(params.n_layers):
            if params.moe_freq > 0 and layer_id % params.moe_freq == 0:
                params.ffn_type = "moe"
            else:
                params.ffn_type = ffn_type_
            self.layers.append(Block(layer_id, params))

        # get class for normalization layers
        self.norm = params.norm_type(
            params.dim,
            eps=params.norm_eps,
        )
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        if self.weight_tying:
            self.tok_embeddings.weight = self.output.weight
        self.grad_checkpointing = False
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weight 1/sqrt(dim)
        # this is 1/fan_in for output, as is default, and Maciej Kilian tried another option
        # for the embed layer (from RWKV paper) but this was better.
        std = 1.0 / math.sqrt(self.params.dim)
        torch.nn.init.trunc_normal_(self.output.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.tok_embeddings.weight, std=std, a=-3 * std, b=3 * std)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, use_cache=False, attention_mask=None):
        """
        Args:
            input
            past_key_values
            use_cache (bool)
            attention_mask (torch.Tensor): Shape (batch_size, sequence_len), indicates tokens that should not be
                attended to. attention_mask[s, i] = False indicates that token i should not be attended to by any other
                token for sequence s.
        """
        if input_ids is not None:
            x = self.tok_embeddings(input_ids)
        elif inputs_embeds is not None:
            x = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        x = self.post_embed_norm(x)

        if past_key_values is None:
            past_key_values = [None] * self.n_layers
        elif isinstance(past_key_values, tuple):
            past_key_values = list(past_key_values)
        for i, layer in enumerate(self.layers):
            if self.grad_checkpointing:
                x, past_key_values[i] = checkpoint(layer, x, past_key_values[i], use_cache, attention_mask)
            else:
                x, past_key_values[i] = layer(x, past_key_values[i], use_cache=use_cache, attention_mask=attention_mask)
        if past_key_values[0] is None:
            past_key_values = None
        x = self.norm(x)
        output = self.output(x)
        # follow llama in casting this to float.
        return output.float(), x, past_key_values

    def get_input_embeddings(self):
        return self.tok_embeddings

    def get_output_embeddings(self):
        return self.output


def create_params(args):
    cfg = None

    if args.model.endswith(".json"):
        _rescan_model_configs(model_config_paths=args.model)
        args.model = Path(args.model).stem

    if args.model in _MODEL_CONFIGS:
        cfg = deepcopy(_MODEL_CONFIGS[args.model])
    else:
        raise ValueError("Pass a pre-defined open_lm model name or a json config")

    # Note: here all the parameters should come from the config file
    # but for retro-compatibility, we add new model parameters to the args (with a default value that matches the old version)
    # These args are managed separately by the argparser
    # If a parameter is in the model config, regardless of the args, we use the config parameters
    # If a parameter is not in the model config, we use the args parameter

    if "mamba" in args.model:
        return {
            "d_model": cfg["d_model"],
            "n_layer": cfg["n_layer"],
            "vocab_size": cfg["vocab_size"],
            "seq_len": cfg["seq_len"],
        }
    else:
        if cfg.get("qk_head_dim", args.qk_head_dim) is None:
            qk_head_dim = cfg["hidden_dim"] // cfg["n_heads"]
        else:
            qk_head_dim = cfg.get("qk_head_dim", args.qk_head_dim)

        if cfg.get("v_head_dim", args.v_head_dim) is None:
            v_head_dim = cfg["hidden_dim"] // cfg["n_heads"]
        else:
            v_head_dim = cfg.get("v_head_dim", args.v_head_dim)
        return Params(
            dim=cfg["hidden_dim"],
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            intermediate_dim_ffn=cfg.get("intermediate_dim_ffn", args.intermediate_dim_ffn),
            n_heads_kv=cfg.get("n_heads_kv", cfg["n_heads"]),
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            seq_len=cfg["seq_len"],
            vocab_size=cfg["vocab_size"],
            post_embed_norm=cfg["post_embed_norm"],
            weight_tying=cfg["weight_tying"],
            norm_type=get_norm_class(cfg.get("model_norm", args.model_norm)),
            attn_func=get_attn_func(
                args.attn_name, args.attn_activation, args.attn_seq_scalar, args.attn_seq_scalar_alpha
            ),
            attn_name=cfg.get("attn_name", args.attn_name),
            apply_qk_norm=cfg.get("qk_norm", args.qk_norm),
            positional_embedding_type=cfg.get("positional_embedding_type", args.positional_embedding_type),
            rotary_base_frequency=cfg.get("rotary_base_frequency", args.rotary_base_frequency),
            rotary_scale=cfg.get("rotary_scale", args.rotary_scale),
            ffn_type=cfg.get("ffn_type", args.ffn_type),
            moe_num_experts=cfg.get("moe_num_experts", args.moe_num_experts),
            moe_loss_weight=cfg.get("moe_loss_weight", args.moe_loss_weight),
            moe_expert_model_parallelism=cfg.get("moe_expert_model_parallelism", args.moe_expert_model_parallelism),
            moe_weight_parallelism=cfg.get("moe_weight_parallelism", args.moe_weight_parallelism),
            moe_capacity_factor=cfg.get("moe_capacity_factor", args.moe_capacity_factor),
            moe_freq=cfg.get("moe_freq", args.moe_freq),
            moe_top_k=cfg.get("moe_top_k", args.moe_top_k),
            use_decay=cfg.get("use_decay", args.use_decay),
            use_retnet_slopes=cfg.get("use_retnet_slopes", args.use_retnet_slopes),
            decay_start=cfg.get("decay_start", args.decay_start),
        )


class Mamba(nn.Module):
    # Experimental architecture, please "pip install mamba-ssm"
    # https://arxiv.org/abs/2312.00752
    def __init__(self, params):
        if MambaLMHeadModel is None:
            raise ImportError(
                "MambaLMHeadModel is not available. Please install the 'mamba_ssm' package by running 'pip install mamba-ssm'."
            )

        super().__init__()
        self.seq_len = params.pop("seq_len")
        self.vocab_size = params["vocab_size"]

        self.model = MambaLMHeadModel(**params)

    def reset_parameters(self):
        return

    def forward(self, x):
        out = self.model(x).logits
        return out, None, None


def create_model(args):
    if "mamba" in args.model:
        model = Mamba(create_params(args))
        return model
    else:
        model = Transformer(create_params(args))
        return model
