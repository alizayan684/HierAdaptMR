import torch
import torch.nn as nn
import fastmri
import math
import torch.nn.functional as F
from typing import Optional, Tuple
import torch.utils.checkpoint as checkpoint
from fastmri.data import transforms
from typing import List
from torch import Tensor
from einops import rearrange
from mri_utils.math import complex_abs, complex_mul, complex_conj
from mri_utils.coil_combine import rss, rss_complex
from mri_utils.fftc import fft2c, ifft2c


def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor, num_adj_slices: int = 1) -> torch.Tensor:
    _, c, _, _, _ = sens_maps.shape
    # 添加数值检查
    x_expanded = x.repeat_interleave(c // num_adj_slices, dim=1)

    # 检查输入是否有NaN/Inf
    if torch.isnan(x_expanded).any() or torch.isinf(x_expanded).any():
        print("Warning: NaN/Inf detected in sens_expand input")
        x_expanded = torch.nan_to_num(x_expanded, nan=0.0, posinf=1e6, neginf=-1e6)

    if torch.isnan(sens_maps).any() or torch.isinf(sens_maps).any():
        print("Warning: NaN/Inf detected in sens_maps")
        sens_maps = torch.nan_to_num(sens_maps, nan=0.0, posinf=1e6, neginf=-1e6)

    result = fft2c(complex_mul(x_expanded, sens_maps))

    # 检查输出
    if torch.isnan(result).any() or torch.isinf(result).any():
        print("Warning: NaN/Inf detected in sens_expand output")
        result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

    return result


def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor, num_adj_slices: int = 1) -> torch.Tensor:
    b, c, h, w, _ = x.shape
    x = ifft2c(x)

    # 检查中间结果
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("Warning: NaN/Inf detected in sens_reduce after ifft2c")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    x = complex_mul(x, complex_conj(sens_maps))

    # 检查复数乘法后的结果
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("Warning: NaN/Inf detected after complex_mul")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    result = x.view(b, num_adj_slices, c // num_adj_slices, h, w, 2).sum(dim=2, keepdim=False)
    return result

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, stride=stride)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, no_use_ca=False):
        super().__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        if not no_use_ca:
            self.CA = CALayer(n_feat, reduction, bias=bias)
        else:
            self.CA = nn.Identity()
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class PromptBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, learnable_prompt=False):
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size),
                                         requires_grad=learnable_prompt)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))

        # 添加数值检查
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            print("Warning: NaN/Inf in PromptBlock embedding")
            emb = torch.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)

        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = torch.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.dec_conv3x3(prompt)

        return prompt


class DownBlock(nn.Module):
    def __init__(self, input_channel, output_channel, n_cab, kernel_size, reduction, bias, act,
                 no_use_ca=False, first_act=False):
        super().__init__()
        if first_act:
            self.encoder = [CAB(input_channel, kernel_size, reduction,bias=bias, act=nn.PReLU(), no_use_ca=no_use_ca)]
            self.encoder = nn.Sequential(
                    *(self.encoder+[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
                                    for _ in range(n_cab-1)]))
        else:
            self.encoder = nn.Sequential(
                *[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
                  for _ in range(n_cab)])
        self.down = nn.Conv2d(input_channel, output_channel,kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        enc = self.encoder(x)
        x = self.down(enc)
        return x, enc


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, n_cab, kernel_size, reduction, bias, act,
                 no_use_ca=False, n_history=0):
        super().__init__()
        # momentum layer
        self.n_history = n_history
        if n_history > 0:
            self.momentum = nn.Sequential(
                nn.Conv2d(in_dim*(n_history+1), in_dim, kernel_size=1, bias=bias),
                CAB(in_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
            )
            self.attn_proj = nn.Conv2d(in_dim, in_dim *n_history , kernel_size=1, bias=bias)
            self.history_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_dim, in_dim // reduction, kernel_size=1, bias=bias),
                act() if isinstance(act, type) else act,
                nn.Conv2d(in_dim // reduction, in_dim, kernel_size=1, bias=bias),
                nn.Sigmoid()
            )

        self.fuse = nn.Sequential(*[CAB(in_dim+prompt_dim, kernel_size, reduction,
                                        bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        self.reduce = nn.Conv2d(in_dim+prompt_dim, in_dim, kernel_size=1, bias=bias)

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False))

        self.ca = CAB(out_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

    def forward(self, x, prompt_dec, skip, history_feat: Optional[torch.Tensor] = None):
        # momentum layer
        if self.n_history > 0:
            if history_feat is None:
                # 没有历史特征时，重复当前特征 (n_history+1) 次
                x = torch.tile(x, (1, self.n_history + 1, 1, 1))
            else:
                # 有历史特征时的处理
                # x: [1, in_dim, H, W] 例如 [1, 120, H, W]
                # history_feat: [1, in_dim*n_history, H, W] 例如 [1, 1320, H, W]

                # 计算注意力权重并调整维度
                attn_weight = self.history_attn(x)  # [B, in_dim, 1, 1]
                attn_weight = self.attn_proj(attn_weight)  # [B, in_dim, 1, 1] → [B, in_dim*n_history, 1, 1]

                # 应用注意力权重到历史特征
                weighted_history = history_feat * attn_weight

                # 拼接：x + weighted_history = [1, 120+1320, H, W] = [1, 1440, H, W]
                x = torch.cat([x, weighted_history], dim=1)

            # 现在 x 的通道数应该是 in_dim * (n_history + 1)
            x = self.momentum(x)

        x = torch.cat([x, prompt_dec], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)

        x = self.up(x) + skip
        x = self.ca(x)

        return x


class SkipBlock(nn.Module):
    def __init__(self, enc_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        super().__init__()
        if n_cab == 0:
            self.skip_attn = nn.Identity()
        else:
            self.skip_attn = nn.Sequential(*[CAB(enc_dim, kernel_size, reduction, bias=bias, act=act,
                                                 no_use_ca=no_use_ca) for _ in range(n_cab)])

    def forward(self, x):
        x = self.skip_attn(x)
        return x


class PromptUnet(nn.Module):
    def __init__(self,
                in_chans: int,
                out_chans: int,
                n_feat0: int,
                feature_dim: List[int],
                prompt_dim: List[int],
                len_prompt: List[int],
                prompt_size: List[int],
                n_enc_cab: List[int],
                n_dec_cab: List[int],
                n_skip_cab: List[int],
                n_bottleneck_cab: int,
                kernel_size=3,
                reduction=4,
                act=nn.PReLU(),
                bias=False,
                no_use_ca=False,
                learnable_prompt=False,
                adaptive_input=False,
                n_buffer=0,
                n_history=0,
                 ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_history = n_history
        self.n_buffer = n_buffer if adaptive_input else 0

        in_chans = in_chans * (1+self.n_buffer) if adaptive_input else in_chans
        out_chans = out_chans * (1+self.n_buffer) if adaptive_input else in_chans

        # Feature extraction
        self.feat_extract = conv(in_chans, n_feat0, kernel_size, bias=bias)

        # Encoder - 3 DownBlocks
        self.enc_level1 = DownBlock(n_feat0, feature_dim[0], n_enc_cab[0], kernel_size, reduction, bias, act, no_use_ca, first_act=True)
        self.enc_level2 = DownBlock(feature_dim[0], feature_dim[1], n_enc_cab[1], kernel_size, reduction, bias, act, no_use_ca)
        self.enc_level3 = DownBlock(feature_dim[1], feature_dim[2], n_enc_cab[2], kernel_size, reduction, bias, act, no_use_ca)

        # Skip Connections - 3 SkipBlocks
        self.skip_attn1 = SkipBlock(n_feat0, n_skip_cab[0], kernel_size, reduction, bias, act, no_use_ca)
        self.skip_attn2 = SkipBlock(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias, act, no_use_ca)
        self.skip_attn3 = SkipBlock(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias, act, no_use_ca)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[CAB(feature_dim[2], kernel_size, reduction, bias, act, no_use_ca)
                                          for _ in range(n_bottleneck_cab)])
        # Decoder - 3 UpBlocks
        self.prompt_level3 = PromptBlock(prompt_dim[2], len_prompt[2], prompt_size[2], feature_dim[2], learnable_prompt)
        self.dec_level3 = UpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], n_dec_cab[2], kernel_size, reduction, bias, act, no_use_ca, n_history)

        self.prompt_level2 = PromptBlock(prompt_dim[1], len_prompt[1], prompt_size[1], feature_dim[1], learnable_prompt)
        self.dec_level2 = UpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], n_dec_cab[1], kernel_size, reduction, bias, act, no_use_ca, n_history)

        self.prompt_level1 = PromptBlock(prompt_dim[0], len_prompt[0], prompt_size[0], feature_dim[0], learnable_prompt)
        self.dec_level1 = UpBlock(feature_dim[0], n_feat0, prompt_dim[0], n_dec_cab[0], kernel_size, reduction, bias, act, no_use_ca, n_history)

        # OutConv
        self.conv_last = conv(n_feat0, out_chans, 5, bias=bias)

    def forward(self, x, history_feat: Optional[List[torch.Tensor]] = None):
        if history_feat is None:
            history_feat = [None, None, None]

        history_feat3, history_feat2, history_feat1 = history_feat
        current_feat = []

        # 0. featue extraction
        x = self.feat_extract(x)

        # 1. encoder
        x, enc1 = self.enc_level1(x)
        x, enc2 = self.enc_level2(x)
        x, enc3 = self.enc_level3(x)

        # 2. bottleneck
        x = self.bottleneck(x)

        # 3. decoder
        current_feat.append(x.clone())
        dec_prompt3 = self.prompt_level3(x)
        x = self.dec_level3(x, dec_prompt3, self.skip_attn3(enc3), history_feat3)

        current_feat.append(x.clone())
        dec_prompt2 = self.prompt_level2(x)
        x = self.dec_level2(x, dec_prompt2, self.skip_attn2(enc2), history_feat2)

        current_feat.append(x.clone())
        dec_prompt1 = self.prompt_level1(x)
        x = self.dec_level1(x, dec_prompt1, self.skip_attn1(enc1), history_feat1)

        # 4. last conv
        if self.n_history > 0:
            for i, history_feat_i in enumerate(history_feat):
                if history_feat_i is None:  # for the first cascade, repeat the current feature
                    history_feat[i] = torch.cat([torch.tile(current_feat[i], (1, self.n_history, 1, 1))], dim=1)
                else:  # for the rest cascades: pop the oldest feature and append the current feature
                    history_feat[i] = torch.cat([current_feat[i], history_feat[i][:, :-self.feature_dim[2-i]]], dim=1)
        return self.conv_last(x), history_feat


class NormPromptUnet(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        no_use_ca: bool = False,
        learnable_prompt=False,
        adaptive_input=False,
        n_buffer=0,
        n_history=0,
    ):

        super().__init__()
        self.n_history = n_history
        self.n_buffer = n_buffer
        self.unet = PromptUnet(in_chans=in_chans,
                               out_chans=out_chans,
                               n_feat0=n_feat0,
                               feature_dim=feature_dim,
                               prompt_dim=prompt_dim,
                               len_prompt=len_prompt,
                               prompt_size=prompt_size,
                               n_enc_cab=n_enc_cab,
                               n_dec_cab=n_dec_cab,
                               n_skip_cab=n_skip_cab,
                               n_bottleneck_cab=n_bottleneck_cab,
                               no_use_ca=no_use_ca,
                               learnable_prompt=learnable_prompt,
                               adaptive_input=adaptive_input,
                               n_buffer=n_buffer,
                               n_history=n_history,
                               )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return rearrange(x, 'b c h w two -> b (two c) h w')

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        return rearrange(x, 'b (two c) h w -> b c h w two', two=2).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        # 添加epsilon防止除零
        eps = 1e-8
        std = torch.clamp(std, min=eps)  # 防止std为0

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor,
              h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor,
                history_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                buffer: torch.Tensor = None):
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        cc = x.shape[1]
        if buffer is not None:
            x = torch.cat([x, buffer], dim=1)

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x, history_feat = self.unet(x, history_feat)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        if buffer is not None:
            x, _, latent, _ = torch.split(x, [cc, cc, cc, x.shape[1] - 3*cc], dim=1)
        else:
            latent = None
        return x, latent, history_feat


class PromptMRBlock(nn.Module):

    def __init__(self, model: nn.Module, num_adj_slices=5):

        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
            self,
            current_img: torch.Tensor,
            img_zf: torch.Tensor,
            latent: torch.Tensor,
            mask: torch.Tensor,
            sens_maps: torch.Tensor,
            history_feat: Optional[Tuple[torch.Tensor, ...]] = None
    ):
        mask = mask.bool()
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_img)
        current_kspace = sens_expand(current_img, sens_maps, self.num_adj_slices)
        ffx = sens_reduce(torch.where(mask, current_kspace, zero), sens_maps, self.num_adj_slices)
        if self.model.n_buffer > 0:
            # adaptive input. buffer: A^H*A*x_i, s_i, x0, A^H*A*x_i-x0
            buffer = torch.cat([ffx, latent, img_zf] + [ffx - img_zf] * (self.model.n_buffer - 3), dim=1)
        else:
            buffer = None

        soft_dc = (ffx - img_zf) * self.dc_weight
        model_term, latent, history_feat = self.model(current_img, history_feat, buffer)
        img_pred = current_img - soft_dc - model_term
        return img_pred, latent, history_feat

class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        num_adj_slices: int = 5,
        n_feat0: int = 24,
        feature_dim: List[int] = [36, 48, 60],
        prompt_dim: List[int] = [12, 24, 36],
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        mask_center: bool = True,
        low_mem: bool = False,

    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in
                BottleneckBlock.
            no_use_ca: not using channel attention.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.num_adj_slices = num_adj_slices
        self.low_mem = low_mem
        self.norm_unet = NormPromptUnet(in_chans=in_chans,
                                out_chans = out_chans,
                                n_feat0=n_feat0,
                                feature_dim = feature_dim,
                                prompt_dim = prompt_dim,
                                len_prompt = len_prompt,
                                prompt_size = prompt_size,
                                n_enc_cab = n_enc_cab,
                                n_dec_cab = n_dec_cab,
                                n_skip_cab = n_skip_cab,
                                n_bottleneck_cab = n_bottleneck_cab,
                                no_use_ca = no_use_ca)

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        b, adj_coil, h, w, two = x.shape
        coil = adj_coil//self.num_adj_slices
        x = x.view(b, self.num_adj_slices, coil, h, w, two)
        # 计算RSS时添加epsilon
        rss_result = rss_complex(x, dim=2)
        eps = 1e-8
        rss_result = torch.clamp(rss_result, min=eps)  # 防止除零

        x = x / rss_result.unsqueeze(-1).unsqueeze(2)

        return x.view(b, adj_coil, h, w, two)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def compute_sens(self, model: nn.Module, images: torch.Tensor, compute_per_coil: bool) -> torch.Tensor:
        bc = images.shape[0]  # batch_size * n_coils
        if compute_per_coil:
            output = []
            for i in range(bc):
                output.append(model(images[i].unsqueeze(0))[0])
            output = torch.cat(output, dim=0)
        else:
            output = model(images)[0]
        return output

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor,
                num_low_frequencies: Optional[int] = None) -> torch.Tensor:

        # Determine the type of mask: Uniform, Gaussian, or Radial
        mask_type = self._identify_mask_type(mask)

        # Step 1: Handle different mask types correctly
        if self.mask_center:
            if mask_type in ['ktUniform', 'ktGaussian']:
                pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask, num_low_frequencies)
                masked_kspace = transforms.batched_mask_center(masked_kspace, pad, pad + num_low_freqs)
            elif mask_type == 'ktRadial':
                # Special handling for radial masks (central 16x16 regions)
                masked_kspace = self._apply_radial_center_mask(masked_kspace, mask)

        # Step 2: Convert k-space to image space, potentially adapting to mask type
        images, batches = self.chans_to_batch_dim(ifft2c(masked_kspace))

        # Step 3: Adjust for different k-space trajectories
        computed_sens = self.compute_sens(self.norm_unet, images, compute_per_coil=False)

        # Step 4: Return divided root sum of squares result
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(computed_sens, batches)
        )

    def _identify_mask_type(self, mask: torch.Tensor) -> str:
        center_lines = mask[..., mask.shape[3] // 2 - 10:mask.shape[3] // 2 + 10, 0] #mask[1,50,416,168,1]

        if torch.all(center_lines == 1):
            return 'ktUniform'
        else:
            return 'ktRadial'

    def _apply_radial_center_mask(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Apply central 16x16 region mask for radial k-space trajectory
        # Implement the logic to zero-out all k-space data except for the central 16x16 region
        center_x = mask.shape[-2] // 2
        center_y = mask.shape[-3] // 2

        masked_kspace[..., center_y - 10:center_y + 10, center_x - 10:center_x + 10, :] = masked_kspace[...,
                                                                                      center_y - 10:center_y + 10,
                                                                                      center_x - 10:center_x + 10, :]

        return masked_kspace


class PromptMR(nn.Module):
    """
    An prompt-learning based unrolled model for multi-coil MR reconstruction,
    see https://arxiv.org/abs/2309.13839.

    """

    def __init__(
        self,
        num_cascades: int = 12,
        num_adj_slices: int = 5,
        n_feat0: int = 48,
        feature_dim: List[int] = [72, 96, 120],
        prompt_dim: List[int] = [24, 48, 72],
        sens_n_feat0: int =24,
        sens_feature_dim: List[int] = [36, 48, 60],
        sens_prompt_dim: List[int] = [12, 24, 36],
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        sens_len_prompt: Optional[List[int]] = None,
        sens_prompt_size: Optional[List[int]] = None,
        sens_n_enc_cab: Optional[List[int]] = None,
        sens_n_dec_cab: Optional[List[int]] = None,
        sens_n_skip_cab: Optional[List[int]] = None,
        sens_n_bottleneck_cab: Optional[List[int]] = None,
        sens_no_use_ca: Optional[bool] = None,
        mask_center: bool = True,
        use_checkpoint: bool = False,
        low_mem: bool = False,
        adaptive_input: bool = False,
        n_buffer: int = 4,
        n_history: int = 0,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational network.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            sens_n_feat0: Number of top-level feature channels for sense map
                estimation PromptUnet in PromptMR.
            sens_feature_dim: feature dim for each level in PromptUnet for
                sensitivity map estimation (SME) network.
            sens_prompt_dim: prompt dim for each level in PromptUnet in
                sensitivity map estimation (SME) network.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in
                BottleneckBlock.
            no_use_ca: not using channel attention.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
            use_checkpoint: Whether to use checkpointing to trade compute for GPU memory.
            low_mem: Whether to compute sensitivity map coil by coil to save GPU memory.
        """
        super().__init__()
        assert num_adj_slices % 2 == 1, "num_adj_slices must be odd"
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices//2
        self.n_history = n_history
        self.n_buffer = n_buffer
        self.sens_net = SensitivityModel(
            num_adj_slices=num_adj_slices,
            n_feat0=sens_n_feat0,
            feature_dim= sens_feature_dim,
            prompt_dim = sens_prompt_dim,
            len_prompt = sens_len_prompt if sens_len_prompt is not None else len_prompt,
            prompt_size = sens_prompt_size if sens_prompt_size is not None else prompt_size,
            n_enc_cab = sens_n_enc_cab if sens_n_enc_cab is not None else n_enc_cab,
            n_dec_cab = sens_n_dec_cab if sens_n_dec_cab is not None else n_dec_cab,
            n_skip_cab = sens_n_skip_cab if sens_n_skip_cab is not None else n_skip_cab,
            n_bottleneck_cab = sens_n_bottleneck_cab if sens_n_bottleneck_cab is not None else n_bottleneck_cab,
            no_use_ca = sens_no_use_ca if sens_no_use_ca is not None else no_use_ca,
            mask_center=mask_center,
            low_mem=low_mem,

        )
        self.cascades = nn.ModuleList([
            PromptMRBlock(
                NormPromptUnet(
                    in_chans=2 * num_adj_slices,
                    out_chans=2 * num_adj_slices,
                    n_feat0=n_feat0,
                    feature_dim=feature_dim,
                    prompt_dim=prompt_dim,
                    len_prompt=len_prompt,
                    prompt_size=prompt_size,
                    n_enc_cab=n_enc_cab,
                    n_dec_cab=n_dec_cab,
                    n_skip_cab=n_skip_cab,
                    n_bottleneck_cab=n_bottleneck_cab,
                    adaptive_input=adaptive_input,
                    n_buffer=n_buffer,
                    n_history=n_history
                ),
                num_adj_slices=num_adj_slices
            ) for _ in range(num_cascades)
        ])
        self.num_cascades = num_cascades
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = 20,
    ) -> tuple[Tensor, Tensor]:

        if self.use_checkpoint and self.training:
            sens_maps = torch.utils.checkpoint.checkpoint(
                self.sens_net, masked_kspace, mask, num_low_frequencies, use_reentrant=False)
        else:
            sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        # kspace_pred = masked_kspace.clone()
        img_zf = sens_reduce(masked_kspace, sens_maps, self.num_adj_slices)
        img_pred = img_zf.clone()
        latent = img_zf.clone()
        history_feat = None

        for cascade in self.cascades:
            if self.use_checkpoint:  # and self.training:
                img_pred, latent, history_feat = torch.utils.checkpoint.checkpoint(
                    cascade, img_pred, img_zf, latent, mask, sens_maps, history_feat, use_reentrant=False)
            else:
                img_pred, latent, history_feat = cascade(img_pred, img_zf, latent, mask, sens_maps, history_feat)
        img_pred = torch.chunk(img_pred, self.num_adj_slices, dim=1)[self.center_slice]
        sens_maps = torch.chunk(sens_maps, self.num_adj_slices, dim=1)[self.center_slice]
        img_pred = rss(complex_abs(complex_mul(img_pred, sens_maps)), dim=1)

        return img_pred