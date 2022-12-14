import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np
from einops import rearrange, repeat

import vol_utils.general as utils
from vol_utils import rend_util
from model.density import LaplaceDensity, AbsDensity
from model.ray_sampler import ErrorBoundSampler
from model.embedder import get_embedder

"""
For modeling more complex backgrounds, we follow the inverted sphere parametrization from NeRF++ 
https://github.com/Kai-46/nerfplusplus 
"""


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[..., :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[..., :1]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, dim=-1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[..., 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[..., :1]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, dim=-1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == "idr":
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == "nerf":
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise ValueError(f"Currently, it only support mode with [`idr`, `nerf`].")

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def create_activation(name, **kwargs):
    if name == "relu":
        act = nn.ReLU(inplace=True)

    elif name == "softplus":
        beta = kwargs.get("beta", 100)
        act = nn.Softplus(beta=beta)

    elif name == "gelu":
        act = nn.GELU()

    elif name == "GEGLU":
        act = GEGLU()

    else:
        raise ValueError(f"{name} is invalid. Currently, it only supports [`relu`, `softplus`, `sine`, `gaussian`]")

    return act


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        return self.to_out(out)


class CodebookAttention(nn.Module):
    def __init__(self, *,
                 codebook_dim,
                 depth: int = 1,
                 num_latents: int = 512,
                 latent_dim: int = 256,
                 latent_heads: int = 8,
                 latent_dim_head: int = 64,
                 cross_heads: int = 1,
                 cross_dim_head: int = 64):

        super().__init__()

        self.latents = nn.Parameter(torch.randn((num_latents, latent_dim), dtype=torch.float32))

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, codebook_dim, heads=cross_heads,
                                          dim_head=cross_dim_head), context_dim=codebook_dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        self.self_attend_blocks = nn.ModuleList([])
        for i in range(depth):
            self_attn = PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
            self_ff = PreNorm(latent_dim, FeedForward(latent_dim))

            self.self_attend_blocks.append(nn.ModuleList([self_attn, self_ff]))

    def forward(self, codebook):
        """ Useful code items selection.

        Args:
            codebook (torch.Tensor): [b, n, d]

        Returns:
            x (torch.Tensor): [b, k, d]
        """

        b = codebook.shape[0]

        x = repeat(self.latents, "k d -> b k d", b=b)

        cross_attn, cross_ff = self.cross_attend_blocks

        # cross attention only happens once for Perceiver IO
        x = cross_attn(x, context=codebook) + x
        x = cross_ff(x) + x

        # self attention
        for self_attn, self_ff in self.self_attend_blocks:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return x


class CoordinateAttention(nn.Module):
    def __init__(self, *,
                 queries_dim,
                 depth: int = 1,
                 activation: str = "geglu",
                 latent_dim: int = 256,
                 cross_heads: int = 1,
                 cross_dim_head: int = 64,
                 decoder_ff: bool = True):

        super().__init__()

        self.cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads,
                                                         dim_head=cross_dim_head), context_dim=latent_dim)

        if activation == "geglu":
            hidden_dim = queries_dim * 2
        else:
            hidden_dim = queries_dim

        self.cross_attend_blocks = nn.ModuleList()

        for i in range(depth):
            cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads,
                                                        dim_head=cross_dim_head), context_dim=latent_dim)

            ffn = nn.Sequential(
                nn.Linear(queries_dim, hidden_dim),
                create_activation(name=activation),
                nn.Linear(hidden_dim, queries_dim)
            )

            if i == depth - 1 and decoder_ff:
                cross_ff = PreNorm(queries_dim, ffn)
            else:
                cross_ff = None

            self.cross_attend_blocks.append(nn.ModuleList([cross_attn, cross_ff]))

    def forward(self, queries, latents):
        """ Query points features from the latents codebook.

        Args:
            queries (torch.Tensor): [b, n, c], the sampled points.
            latents (torch.Tensor): [b, n, k]

        Returns:
            x (torch.Tensor): [b, n, c]

        """

        x = queries

        # cross attend from queries to latents
        for cross_attn, cross_ff in self.cross_attend_blocks:
            x = cross_attn(x, context=latents) + x

            if cross_ff is not None:
                x = cross_ff(x) + x

        return x


class ImplicitAttentionNetwork(nn.Module):
    def __init__(
            self,
            codebook,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,

            depth: int = 1,
            num_latents: int = 512,
            latent_dim: int = 256,
            latent_heads: int = 8,
            latent_dim_head=64,

            num_cross_depth: int = 1,
            activation: str = "softplus",
            cross_heads: int = 1,
            cross_dim_head: int = 64,
            decoder_ff: bool = True
    ):
        super().__init__()

        self.register_buffer("codebook", codebook)

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale

        self.embed_fn = None
        input_dim = d_in
        if multires > 0:
            self.embed_fn, input_dim = get_embedder(multires, input_dims=d_in)

        n_embed, codebook_dim = codebook.shape
        self.codebook_attn = CodebookAttention(
            codebook_dim=codebook_dim,
            depth=depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            latent_heads=latent_heads,
            latent_dim_head=latent_dim_head,
            cross_heads=cross_heads,
            cross_dim_head=cross_dim_head
        )

        self.coordinate_attn = CoordinateAttention(
            queries_dim=input_dim,
            depth=num_cross_depth,
            activation=activation,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            cross_dim_head=cross_dim_head,
            decoder_ff=decoder_ff
        )

        dims = [input_dim + input_dim] + dims + [d_out + feature_vector_size]

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.mlp = nn.ModuleList()

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            self.mlp.append(lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, points):
        if self.embed_fn is not None:
            points = self.embed_fn(points)

        x = points

        b, n_rays, n_pts, c = x.shape

        codebook = self.codebook
        if codebook.ndim == 2:
            codebook = repeat(codebook, "n d -> b n d", b=b)

        latents = self.codebook_attn(codebook)

        x = x.view((b, n_rays * n_pts, c))
        x = self.coordinate_attn(x, latents)
        x = x.view((b, n_rays, n_pts, -1))

        x = torch.cat([points, x], dim=-1)

        for l in range(self.num_layers - 1):
            lin = self.mlp[l]

            if l in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[..., :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[..., :1]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, dim=-1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[..., 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[..., :1]
        """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, dim=-1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


class RenderingAttentionNetwork(nn.Module):
    def __init__(
            self,
            codebook,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            multires_view=4,

            num_layers: int = 3,
            hidden_dim: int = 256,

            depth: int = 1,
            num_latents: int = 512,
            latent_dim: int = 256,
            latent_heads: int = 8,
            latent_dim_head=64,

            num_cross_depth: int = 1,
            activation: str = "gelu",
            cross_heads: int = 1,
            cross_dim_head: int = 64,
            decoder_ff: bool = True
    ):
        super().__init__()

        self.register_buffer("codebook", codebook)

        self.mode = mode

        input_dim = d_in + feature_vector_size
        self.embedview_fn = None
        if multires_view > 0:
            self.embedview_fn, input_ch = get_embedder(multires_view)
            input_dim += (input_ch - 3)

        n_embed, codebook_dim = codebook.shape
        self.codebook_attn = CodebookAttention(
            codebook_dim=codebook_dim,
            depth=depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            latent_heads=latent_heads,
            latent_dim_head=latent_dim_head,
            cross_heads=cross_heads,
            cross_dim_head=cross_dim_head
        )

        self.coordinate_attn = CoordinateAttention(
            queries_dim=input_dim,
            depth=num_cross_depth,
            activation=activation,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            cross_dim_head=cross_dim_head,
            decoder_ff=decoder_ff
        )

        color_mlp = []
        c_in = input_dim
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(c_in, hidden_dim),
                nn.ReLU(inplace=True)
            )
            color_mlp.append(layer)
            c_in = hidden_dim

        color_mlp.append(nn.Linear(c_in, d_out))
        self.color_mlp = nn.Sequential(*color_mlp)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == "idr":
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == "nerf":
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise ValueError(f"Currently, it only support mode with [`idr`, `nerf`].")

        x = rendering_input
        b, n_rays, n_pts, c = x.shape

        codebook = self.codebook
        if codebook.ndim == 2:
            codebook = repeat(codebook, "n d -> b n d", b=b)

        latents = self.codebook_attn(codebook)

        x = x.view((b, n_rays * n_pts, c))
        x = self.coordinate_attn(x, latents)
        x = x.view((b, n_rays, n_pts, -1))
        x = self.color_mlp(x)
        x = self.sigmoid(x)

        return x




class VolSDFNetworkBG(nn.Module):
    def __init__(self, conf):
        super().__init__()

        # load codebook
        vq_cfg = conf["vq_cfg"]
        vq_path = vq_cfg["ckpt_path"]
        ckpt_data = torch.load(vq_path, map_location="cpu")
        state_dict = ckpt_data.get("state_dict", {})
        codebook = state_dict["quantize.embedding.weight"]

        # Foreground object"s networks
        self.scene_bounding_sphere = conf.get_float("scene_bounding_sphere")
        self.feature_vector_size = conf.get_int("feature_vector_size")

        self.implicit_network = ImplicitAttentionNetwork(codebook, self.feature_vector_size, 0.0,
                                                         **conf.get_config("implicit_network"))
        self.rendering_network = RenderingAttentionNetwork(codebook, self.feature_vector_size,
                                                           **conf.get_config("rendering_network"))

        self.density = LaplaceDensity(**conf.get_config("density"))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, inverse_sphere_bg=True,
                                             **conf.get_config("ray_sampler"))

        # Background"s networks
        bg_feature_vector_size = conf.get_int("bg_network.feature_vector_size")
        self.bg_implicit_network = ImplicitNetwork(bg_feature_vector_size, 0.0,
                                                   **conf.get_config("bg_network.implicit_network"))
        self.bg_rendering_network = RenderingNetwork(bg_feature_vector_size,
                                                     **conf.get_config("bg_network.rendering_network"))
        self.bg_density = AbsDensity(**conf.get_config("bg_network.density", default={}))

    def geometric_init(self):
        pass

    def forward(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        # b_r_rays_dirs: [bs, n_rays, 3]
        # b_cam_loc: [bs, 3]
        b_r_rays_dirs, b_cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = b_r_rays_dirs.shape

        # b_r_cam_loc: [bs, n_rays, 3]
        b_r_cam_loc = b_cam_loc.unsqueeze(1).repeat(1, num_pixels, 1)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(b_r_rays_dirs, b_r_cam_loc, self)

        z_vals, z_vals_bg = z_vals
        z_max = z_vals[..., -1]
        z_vals = z_vals[..., :-1]
        N_samples = z_vals.shape[-1]

        points = b_r_cam_loc.unsqueeze(-2) + z_vals.unsqueeze(-1) * b_r_rays_dirs.unsqueeze(-2)
        view_dirs = b_r_rays_dirs.unsqueeze(-2).expand((batch_size, num_pixels, N_samples, 3))

        b_r_p_points = points.view((batch_size, num_pixels, N_samples, -1))
        b_r_p_view_dirs = view_dirs.view((batch_size, num_pixels, N_samples, -1))
        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(b_r_p_points)
        rgb = self.rendering_network.forward(b_r_p_points, gradients, b_r_p_view_dirs, feature_vectors)

        weights, bg_transmittance = self.volume_rendering(z_vals, z_max, sdf)

        fg_rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, -2)

        # Background rendering
        N_bg_samples = z_vals_bg.shape[-1]
        z_vals_bg = torch.flip(z_vals_bg, dims=[-1, ])  # 1--->0

        bg_dirs = b_r_rays_dirs.unsqueeze(-2).expand((batch_size, num_pixels, N_bg_samples, 3))
        bg_locs = b_r_cam_loc.unsqueeze(-2).expand((batch_size, num_pixels, N_bg_samples, 3))

        bg_points = self.depth2pts_outside(bg_locs, bg_dirs, z_vals_bg)  # [..., N_samples, 4]

        output = self.bg_implicit_network(bg_points)
        bg_sigma = output[..., :1]
        bg_feature_vectors = output[..., 1:]
        bg_rgb = self.bg_rendering_network(None, None, bg_dirs, bg_feature_vectors)

        bg_weights = self.bg_volume_rendering(z_vals_bg, bg_sigma)
        bg_rgb_values = torch.sum(bg_weights.unsqueeze(-1) * bg_rgb, -2)

        # Composite foreground and background
        bg_rgb_values = bg_transmittance.unsqueeze(-1) * bg_rgb_values
        rgb_values = fg_rgb_values + bg_rgb_values

        output = {
            "rgb_values": rgb_values,
        }

        if self.training:
            # Sample points for the eikonal loss
            device = b_r_rays_dirs.device
            eikonal_points = torch.empty((batch_size, num_pixels, 1, 3)).uniform_(-self.scene_bounding_sphere,
                                                                                  self.scene_bounding_sphere).to(device)

            # add some of the near surface points
            eik_near_points = b_r_rays_dirs.unsqueeze(-2) + z_samples_eik.unsqueeze(-1) * b_r_rays_dirs.unsqueeze(-2)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], -2)

            grad_theta = self.implicit_network.gradient(eikonal_points)
            output["grad_theta"] = grad_theta

        else:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, -2)

            output["normal_map"] = normal_map

        return output

    def volume_rendering(self, z_vals, z_max, sdf):
        """

        Args:
            z_vals:
            z_max: [bs, n_rays, 1]
            sdf: [bs, n_rays, n_pts, 1]

        Returns:

        """

        # [bs, n_rays, n_pts]
        density = self.density(sdf).squeeze(-1)

        # included also the dist from the sphere intersection
        # [bs, n_rays, n_pts - 1]
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # [bs, n_rays, n_pts]
        dists = torch.cat([dists, z_max.unsqueeze(-1) - z_vals[..., -1:]], -1)

        # LOG SPACE, [bs, n_rays, n_pts]
        free_energy = dists * density

        # add 0 for transperancy 1 at t_0, [bs, n_rays, 1 + n_pts]
        shifted_free_energy = torch.cat([torch.zeros_like(free_energy[..., 0:1]), free_energy], dim=-1)

        # probability of it is not empty here, [bs, n_rays, 1 + n_pts]
        alpha = 1 - torch.exp(-free_energy)

        # probability of everything is empty up to now, [bs, n_rays, 1 + n_pts]
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))

        # [bs, n_rays, n_pts]
        fg_transmittance = transmittance[..., :-1]

        # probability of the ray hits something here, [bs, n_rays, n_pts]
        weights = alpha * fg_transmittance

        # factor to be multiplied with the bg volume rendering, [bs, n_rays]
        bg_transmittance = transmittance[..., -1]

        return weights, bg_transmittance

    def bg_volume_rendering(self, z_vals_bg, bg_sigma):
        """

        Args:
            z_vals_bg: [bs, n_rays, n_pts]
            bg_sigma: [bs, n_rays, n_pts, 1]

        Returns:

        """

        # [bs, n_rays, n_pts]
        bg_density = self.bg_density(bg_sigma).squeeze(-1)

        # [bs, n_rays, n_pts - 1]
        bg_dists = z_vals_bg[..., :-1] - z_vals_bg[..., 1:]

        # [bs, n_rays, 1]
        shifted_zeros = torch.zeros_like(bg_dists[..., 0:1])

        # [bs, n_rays, n_pts]
        bg_dists = torch.cat([bg_dists, shifted_zeros + 1e10], -1)

        # LOG SPACE, [bs, n_rays, n_pts]
        bg_free_energy = bg_dists * bg_density

        # shift one step, [bs, n_rays, n_pts]
        bg_shifted_free_energy = torch.cat([shifted_zeros, bg_free_energy[..., :-1]], dim=-1)

        # probability of it is not empty here, [bs, n_rays, n_pts]
        bg_alpha = 1 - torch.exp(-bg_free_energy)

        # probability of everything is empty up to now, [bs, n_rays, n_pts]
        bg_transmittance = torch.exp(-torch.cumsum(bg_shifted_free_energy, dim=-1))

        # probability of the ray hits something here, [bs, n_rays, n_pts]
        bg_weights = bg_alpha * bg_transmittance

        return bg_weights

    def depth2pts_outside(self, ray_o, ray_d, depth):
        """

        Args:
            ray_o: [..., 3]
            ray_d: [..., 3]
            depth: [..., n_pts], inverse of distance to sphere origin.

        Returns:

        """

        o_dot_d = torch.sum(ray_d * ray_o, dim=-1)
        under_sqrt = o_dot_d ** 2 - ((ray_o ** 2).sum(-1) - self.scene_bounding_sphere ** 2)
        d_sphere = torch.sqrt(under_sqrt) - o_dot_d
        p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
        p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)

        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        phi = torch.asin(p_mid_norm / self.scene_bounding_sphere)
        theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
        rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

        # now rotate p_sphere
        # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                       torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                       rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
        p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        return pts


class VolSDFNetworkPlusplus(VolSDFNetworkBG):
    def __init__(self, conf):
        nn.Module.__init__(self)

        # load codebook
        vq_cfg = conf["vq_cfg"]
        vq_path = vq_cfg["ckpt_path"]
        ckpt_data = torch.load(vq_path, map_location="cpu")
        state_dict = ckpt_data.get("state_dict", {})
        codebook = state_dict["quantize.embedding.weight"]

        # Foreground object"s networks
        self.scene_bounding_sphere = conf.get_float("scene_bounding_sphere")
        self.feature_vector_size = conf.get_int("feature_vector_size")

        self.implicit_network = ImplicitAttentionNetwork(codebook, self.feature_vector_size, 0.0,
                                                         **conf.get_config("implicit_network"))
        self.rendering_network = RenderingAttentionNetwork(codebook, self.feature_vector_size,
                                                           **conf.get_config("rendering_network"))

        self.density = LaplaceDensity(**conf.get_config("density"))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, inverse_sphere_bg=True,
                                             **conf.get_config("ray_sampler"))

        # Background"s networks
        bg_feature_vector_size = conf.get_int("bg_network.feature_vector_size")
        self.bg_implicit_network = ImplicitAttentionNetwork(codebook, bg_feature_vector_size, 0.0,
                                                            **conf.get_config("bg_network.implicit_network"))
        self.bg_rendering_network = RenderingAttentionNetwork(codebook, bg_feature_vector_size,
                                                              **conf.get_config("bg_network.rendering_network"))
        self.bg_density = AbsDensity(**conf.get_config("bg_network.density", default={}))

