# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

from einops import rearrange

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b.to(x.dtype), act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        use_text_encoder= False,
        use_cross_attn  = False,
        text_kwargs     = {},
        use_encoder_decoder=False,
        text_concat=False,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        self.use_encoder_decoder = use_encoder_decoder
        if use_text_encoder:
            text_out_dim = w_dim
            if text_concat:
                text_out_dim = w_dim - z_dim
                w_dim = self.w_dim = z_dim
            self.text_encoder = TextEncoder(
                w_dim=text_out_dim,
                use_encoder_decoder=use_encoder_decoder,
                return_sequences=use_cross_attn,
                **text_kwargs
            )
        else:
            self.text_encoder = None

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, txt=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        if x is not None:
            # Main layers.
            for idx in range(self.num_layers):
                layer = getattr(self, f'fc{idx}')
                x = layer(x)

            # Update moving average of W.
            if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
                with torch.autograd.profiler.record_function('update_w_avg'):
                    self.w_avg.copy_(x.detach().mean(dim=0).to(self.w_avg.dtype).lerp(self.w_avg, self.w_avg_beta))

        ws_txt = None
        if txt is not None and self.text_encoder is not None:
            # TODO: do this after truncate
            ws_txt = self.text_encoder(txt)
            # x = x + ws_txt if x is not None else ws_txt

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                if x is not None:
                    x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
                if ws_txt is not None:
                    ws_txt = ws_txt.unsqueeze(1).repeat([1, self.num_ws] + [1]*(ws_txt.ndim-1))

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x, ws_txt

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

#----------------------------------------------------------------------------

from axial_positional_embedding import AxialPositionalEmbedding

@persistence.persistent_class
class CrossAttention(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads,
        text_dim=512,
        clamp=None,
        init_gain=5e-1
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.text_dim = text_dim
        self.clamp = clamp

        self.q = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.kv = torch.nn.Linear(self.text_dim, 2*self.dim, bias=False)
        self.attn = torch.nn.MultiheadAttention(self.dim, self.heads, batch_first=True)

        self.src_ln = torch.nn.LayerNorm(self.text_dim)
        self.tgt_ln = torch.nn.LayerNorm(self.dim)

        self.gain = torch.nn.Parameter(torch.as_tensor(np.log(init_gain)))

        torch.nn.init.orthogonal_(self.q.weight)
        torch.nn.init.orthogonal_(self.kv.weight)
        torch.nn.init.orthogonal_(self.attn.out_proj.weight)

    def forward(self, src, tgt):
        dtype = tgt.dtype
        q = self.q(self.tgt_ln(tgt.to(torch.float32)))
        kv = self.kv(self.src_ln(src))

        k, v = kv.chunk(2, dim=-1)

        attn_output, attn_output_weights = self.attn(q, k, v)
        attn_output = self.gain.exp() * attn_output
        if self.clamp is not None:
            attn_output = attn_output.clamp(min=-self.clamp, max=self.clamp)
        return attn_output.to(dtype)


@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        use_bf16            = False,
        use_encoder_decoder = False,
        w_txt_res           = 32,
        w_txt_dim           = 512,
        use_cross_attn      = False,
        cross_attn_heads    = 2,
        cross_attn_dim       = None,  # default: out_channels
        txt_gain=1.,
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

        self.use_encoder_decoder = use_encoder_decoder
        self.use_cross_attn = use_cross_attn

        if self.use_encoder_decoder:
            down = max(1, w_txt_res // self.resolution)
            up   = max(1, self.resolution // w_txt_res)
            self.txt_conv = Conv2dLayer(
                w_txt_dim, out_channels, kernel_size=3, bias=True,
                up=up, down=down,
                resample_filter=resample_filter, channels_last=self.channels_last,
                activation='relu'
            )

        if self.use_cross_attn:
            if cross_attn_dim is None:
                cross_attn_dim = out_channels
            self.cross_attn = CrossAttention(dim=cross_attn_dim,
                                             heads=cross_attn_heads,
                                             text_dim=w_txt_dim,
                                             clamp=conv_clamp)
            self.pos_emb = AxialPositionalEmbedding(dim=out_channels,
                                                    axial_shape=(self.resolution, self.resolution),
                                                    axial_dims=(out_channels//2, out_channels//2)
                                                    )

    def forward(self, x, img, ws, ws_txt=None, txt_gain=1., force_fp32=False, fused_modconv=None, autocasting=False, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        if self.use_bf16:
            dtype = torch.bfloat16
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            if autocasting:
                x = self.const.to(memory_format=memory_format)
            else:
                x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            if autocasting:
                x = x.to(memory_format=memory_format)
            else:
                x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            if self.use_encoder_decoder:
                ws_txt = ws_txt.to(dtype=dtype, memory_format=memory_format)
                ws_txt = ws_txt.transpose(1, 3)
                ws_txt_out = self.txt_conv(ws_txt)
                x = x + ws_txt_out
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            if self.use_cross_attn:
                tgt = rearrange(x, 'b c h w -> b (h w) c', h=x.shape[2])
                tgt = tgt + self.pos_emb(tgt)
                attn_out = self.cross_attn(src=ws_txt, tgt=tgt)
                attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=x.shape[2])
                x = x/np.sqrt(2) + txt_gain*attn_out/np.sqrt(2)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            if self.use_encoder_decoder:
                ws_txt = ws_txt.to(dtype=dtype, memory_format=memory_format)
                ws_txt = ws_txt.transpose(1, 3)
                ws_txt_out = self.txt_conv(ws_txt)
                x = x + ws_txt_out
            elif self.use_cross_attn:
                tgt = rearrange(x, 'b c h w -> b (h w) c', h=x.shape[2])
                tgt = tgt + self.pos_emb(tgt)
                attn_out = self.cross_attn(src=ws_txt, tgt=tgt)
                attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=x.shape[2])
                x = x/np.sqrt(2) + txt_gain*attn_out/np.sqrt(2)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            if self.use_encoder_decoder:
                ws_txt = ws_txt.to(dtype=dtype, memory_format=memory_format)
                ws_txt = ws_txt.transpose(1, 3)
                ws_txt_out = self.txt_conv(ws_txt)
                x = x + ws_txt_out
            elif self.use_cross_attn:
                tgt = rearrange(x, 'b c h w -> b (h w) c', h=x.shape[2])
                tgt = tgt + self.pos_emb(tgt)
                attn_out = self.cross_attn(src=ws_txt, tgt=tgt)
                attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=x.shape[2])
                x = x/np.sqrt(2) + txt_gain*attn_out/np.sqrt(2)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            if autocasting:
                y = y.to(memory_format=torch.contiguous_format)
            else:
                y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        # assert x.dtype == dtype
        # assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        use_bf16        = False,
        text_concat     = False,
        use_encoder_decoder = False,
        use_cross_attn  = False,
        cross_attn_resolution = 128,
        w_txt_dim       = 512,
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.text_concat = text_concat
        self.use_encoder_decoder = use_encoder_decoder
        self.use_cross_attn = use_cross_attn

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block_use_cross_attn = use_cross_attn and (res <= cross_attn_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, use_bf16=use_bf16,
                use_encoder_decoder=use_encoder_decoder,
                use_cross_attn=block_use_cross_attn,
                w_txt_dim=w_txt_dim,
                **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, ws_txt, txt_gain=1., autocasting=False, **block_kwargs):
        block_ws = []
        block_ws_txt = []
        with torch.autograd.profiler.record_function('split_ws'):
            # misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            if not autocasting:
                ws = ws.to(torch.float32)
                if ws_txt is not None:
                    ws_txt = ws_txt.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_w = ws.narrow(1, w_idx, block.num_conv + block.num_torgb)

                if ws_txt is None:
                    block_ws.append(block_w)
                    block_ws_txt.append(None)
                else:
                    if self.use_encoder_decoder or self.use_cross_attn:
                        block_w_txt = txt_gain * ws_txt.narrow(1, w_idx, 1).squeeze(1)
                    else:
                        block_w_txt = txt_gain * ws_txt.narrow(1, w_idx, block.num_conv + block.num_torgb)


                    if self.use_encoder_decoder or self.use_cross_attn:
                        pass
                    elif self.text_concat:
                        block_w = torch.cat([block_w,
                                             block_w_txt
                                             ],
                                            dim=-1)
                    else:
                        block_w += txt_gain * block_w_txt
                    block_ws.append(block_w)
                    block_ws_txt.append(block_w_txt)
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws, cur_ws_txt in zip(self.block_resolutions, block_ws, block_ws_txt):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, ws_txt=cur_ws_txt, txt_gain=txt_gain, autocasting=autocasting, **block_kwargs)
        return img

#----------------------------------------------------------------------------

from x_transformers import TransformerWrapper, Encoder, XTransformer


@persistence.persistent_class
class TextEncoder(torch.nn.Module):
    def __init__(self,
        w_dim,                      # output dim
        inner_dim = None,           # model dim (default = w_dim)
        depth = 2,
        head_dim = 128,
        num_tokens = 2500,
        max_seq_len = 130,
        rotary_pos_emb = True,
        ff_glu = True,
        use_scalenorm = True,
        use_rezero = False,
        use_encoder_decoder = False,
        decoder_sqrt_ntok = 32,
        encoder_kwargs = {},
        return_sequences=False
    ):
        super().__init__()

        if inner_dim is None:
            inner_dim = w_dim

        assert inner_dim % head_dim == 0
        n_heads = inner_dim // head_dim

        self.use_encoder_decoder = use_encoder_decoder
        self.return_sequences = return_sequences

        if self.use_encoder_decoder:
            enc_kwargs = dict(
                depth = depth,
                heads = n_heads,
                rotary_pos_emb = rotary_pos_emb,
                ff_glu = ff_glu,
                use_scalenorm = use_scalenorm,
                use_rezero = use_rezero,
            )
            enc_kwargs = {k: encoder_kwargs.get(k, v) for k, v in enc_kwargs.items()}
            enc_kwargs = {'enc_' + k: v for k, v in enc_kwargs.items()}

            self.decoder_sqrt_ntok = decoder_sqrt_ntok
            self.dec_max_seq_len = decoder_sqrt_ntok ** 2

            self.model = XTransformer(
                enc_num_tokens = num_tokens,
                dec_num_tokens = 1,
                enc_max_seq_len = max_seq_len,
                dec_max_seq_len = self.dec_max_seq_len,
                dim = inner_dim,
                dec_depth = depth,
                dec_heads = n_heads,
                dec_rotary_pos_emb = rotary_pos_emb,
                dec_ff_glu = ff_glu,
                **enc_kwargs
            )
        else:
            self.model = TransformerWrapper(
                num_tokens = num_tokens,
                max_seq_len = max_seq_len,
                attn_layers = Encoder(
                    dim = inner_dim,
                    depth = depth,
                    heads = n_heads,
                    rotary_pos_emb = rotary_pos_emb,
                    ff_glu = ff_glu,
                    use_scalenorm = use_scalenorm,
                    use_rezero = use_rezero,
                )
            )
        self.proj = torch.nn.Linear(inner_dim, w_dim)

    def forward(self, tokens):
        if self.use_encoder_decoder:
            tgt = torch.zeros((tokens.shape[0], self.dec_max_seq_len), device=tokens.device, dtype=torch.int)
            enc = self.model.encoder(tokens, return_embeddings = True)
            out = self.model.decoder.net(tgt, context=enc, return_embeddings=True)
            out = rearrange(out, 'b (h w) c -> b h w c', h=self.decoder_sqrt_ntok)
            out = self.proj(out)
            return out
        else:
            out = self.model(tokens, return_embeddings=True)
            if not self.return_sequences:
                out = out[:, 0, :]
            out = self.proj(out)
            return out


#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        text_concat         = False,
        use_encoder_decoder = False,
        use_cross_attn      = False,
        w_txt_res           = 32,
        w_txt_dim           = 512,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
            text_concat=text_concat,
            use_encoder_decoder=use_encoder_decoder,
            use_cross_attn=use_cross_attn,
            w_txt_res=w_txt_res,
            w_txt_dim=w_txt_dim,
            **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.text_concat = text_concat
        self.use_encoder_decoder = use_encoder_decoder
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws,
                                      text_concat=text_concat,
                                      use_encoder_decoder=use_encoder_decoder,
                                      use_cross_attn=use_cross_attn,
                                      **mapping_kwargs)

    def forward(self, z, c, txt=None, txt_gain=1., autocasting=False, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws, ws_txt = self.mapping(z, c, txt, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

        img = self.synthesis(ws, ws_txt, txt_gain=txt_gain, autocasting=autocasting,
                             **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        use_bf16            = False,
        use_ws              = False,
        w_dim               = 512,
        use_encoder_decoder = False,
        w_txt_res           = 32,
        use_cross_attn      = False,
        cross_attn_heads    = 2,
        cross_attn_dim      = None,  # default: tmp_channels
        cross_attn_pdrop    = 0,
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.use_bf16 = use_bf16
        self.use_ws = use_ws
        self.use_encoder_decoder = use_encoder_decoder
        self.use_cross_attn = use_cross_attn
        self.cross_attn_pdrop = cross_attn_pdrop
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        if self.use_ws and (not self.use_encoder_decoder) and (not self.use_cross_attn):
            self.conv0 = SynthesisLayer(tmp_channels, tmp_channels, w_dim=w_dim, resolution=resolution,
                                        kernel_size=3, activation=activation,
                                        conv_clamp=conv_clamp, channels_last=self.channels_last)
        else:
            self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

        if self.use_encoder_decoder:
            down = max(1, w_txt_res // self.resolution)
            up   = max(1, self.resolution // w_txt_res)
            self.txt_conv = Conv2dLayer(
                w_dim, tmp_channels, kernel_size=3, bias=True,
                up=up, down=down,
                resample_filter=resample_filter, channels_last=self.channels_last,
                activation='relu'
            )
        if self.use_cross_attn:
            if cross_attn_dim is None:
                cross_attn_dim = tmp_channels
            self.cross_attn = CrossAttention(dim=cross_attn_dim,
                                             heads=cross_attn_heads,
                                             text_dim=w_dim,
                                             clamp=conv_clamp)
            self.pos_emb = AxialPositionalEmbedding(dim=tmp_channels,
                                                    axial_shape=(self.resolution, self.resolution),
                                                    axial_dims=(tmp_channels//2, tmp_channels//2)
                                                    )


    def forward(self, x, img, w=None, txt_gain=1., force_fp32=False, autocasting=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        if self.use_bf16:
            dtype = torch.bfloat16
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            if autocasting:
                x = x.to(memory_format=memory_format)
            else:
                x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            if autocasting:
                img = img.to(dtype=dtype, memory_format=memory_format)
            else:
                img = img.to(memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            if self.use_cross_attn:
                x = self.conv0(x)
                tgt = rearrange(x, 'b c h w -> b (h w) c', h=x.shape[2])
                tgt = tgt + self.pos_emb(tgt)
                attn_out = self.cross_attn(src=w, tgt=tgt)
                attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=x.shape[2])
                if self.cross_attn_pdrop > 0:
                    dropmask = torch.rand((attn_out.shape[0],))
                    attn_out = torch.where(dropmask < self.cross_attn_pdrop, torch.zeros_like(attn_out), attn_out)
                x = x/np.sqrt(2) + txt_gain*attn_out/np.sqrt(2)
            elif self.use_ws and self.use_encoder_decoder:
                x = self.conv0(x)
                w = w.to(dtype=dtype, memory_format=memory_format)
                w = w.transpose(1, 3)
                ws_txt_out = self.txt_conv(w)
                x = x + ws_txt_out
            elif self.use_ws:
                x = self.conv0(x, w)
            else:
                x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            if self.use_cross_attn:
                x = self.conv0(x)
                tgt = rearrange(x, 'b c h w -> b (h w) c', h=x.shape[2])
                tgt = tgt + self.pos_emb(tgt)
                attn_out = self.cross_attn(src=w, tgt=tgt)
                attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=x.shape[2])
                if self.cross_attn_pdrop > 0:
                    dropmask = torch.rand((attn_out.shape[0],))
                    attn_out = torch.where(dropmask < self.cross_attn_pdrop, torch.zeros_like(attn_out), attn_out)
                x = x/np.sqrt(2) + txt_gain*attn_out/np.sqrt(2)
            elif self.use_ws and self.use_encoder_decoder:
                x = self.conv0(x)
                w = w.to(dtype=dtype, memory_format=memory_format)
                w = w.transpose(1, 3)
                ws_txt_out = self.txt_conv(w)
                x = x + ws_txt_out
            elif self.use_ws:
                x = self.conv0(x, w)
            else:
                x = self.conv0(x)
            x = self.conv1(x)

        # assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        # assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        use_bf16            = False,
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        use_text_encoder    = False,
        use_ws              = False,
        use_encoder_decoder = False,
        use_cross_attn      = False,
        cross_attn_resolution = 128,
        cross_attn_pdrop    = 0,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0 and (not use_text_encoder):
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block_use_cross_attn = use_cross_attn and (res <= cross_attn_resolution)
            block_use_ws = use_ws and ((not use_cross_attn) or (res <= cross_attn_resolution))
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, use_bf16=use_bf16,
                use_ws=block_use_ws,
                use_encoder_decoder=use_encoder_decoder,
                use_cross_attn=block_use_cross_attn,
                cross_attn_pdrop=cross_attn_pdrop,
                w_dim=cmap_dim,
                **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        self.use_ws = use_ws
        self.use_encoder_decoder = use_encoder_decoder
        self.use_cross_attn = use_cross_attn

        self.num_ws = None
        if c_dim > 0 or use_text_encoder:
            if self.use_ws:
                self.num_ws = len(self.block_resolutions)
            self.mapping = MappingNetwork(
                z_dim=0, c_dim=c_dim,
                w_dim=cmap_dim,
                num_ws=self.num_ws,
                w_avg_beta=None,
                use_text_encoder=use_text_encoder,
                use_encoder_decoder=use_encoder_decoder,
                use_cross_attn=use_cross_attn,
                **mapping_kwargs)
        else:
            self.mapping = None
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=0 if self.use_ws else cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, txt=None, txt_gain=1., **block_kwargs):
        x = None

        if self.mapping is not None and self.use_ws:
            _, ws = self.mapping(None, c, txt)

            block_ws = []
            with torch.autograd.profiler.record_function('split_ws'):
                ws = ws.to(torch.float32)
                for w_idx, res in enumerate(self.block_resolutions):
                    block = getattr(self, f'b{res}')
                    block_w = txt_gain * ws[:, w_idx, ...]
                    block_ws.append(block_w)
        else:
            block_ws = [None for _ in self.block_resolutions]

        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, txt_gain=txt_gain, **block_kwargs)

        # TODO: use txt at lower res somehow
        cmap = None
        if self.mapping is not None and not self.use_ws:
            cmap, _ = self.mapping(None, c, txt)
        x = self.b4(x, img, cmap)
        return x

#----------------------------------------------------------------------------
