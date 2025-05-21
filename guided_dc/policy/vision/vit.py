"""
Custom ViT image encoder implementation from IBRL, https://github.com/hengyuan-hu/ibrl

"""

import math
from typing import ClassVar

import einops
import torch
from torch import nn
from torch.nn.init import trunc_normal_


def validate_crop_size(aug, img_size, patch_size):
    from guided_dc.policy.vision.aug import CropRandomizer

    for i in range(len(aug)):
        if isinstance(aug[i], CropRandomizer):
            if aug[i].crop_height % patch_size != 0:
                nearest_crop_height_pct = (
                    aug[i].crop_height // patch_size * patch_size
                ) / img_size[-1]
                raise ValueError(
                    f"crop_height should be divisible by patch_size. Nearest crop_height percentage: {nearest_crop_height_pct}"
                )
            if aug[i].crop_width % patch_size != 0:
                nearest_crop_width_pct = (
                    aug[i].crop_width // patch_size * patch_size
                ) / img_size[-2]
                raise ValueError(
                    f"crop_width should be divisible by patch_size. Nearest crop_width percentage: {nearest_crop_width_pct}"
                )
            break


class VitEncoder2(nn.Module):
    def __init__(
        self,
        img_size,
        img_cond_steps=1,
        patch_size=8,
        depth=1,
        embed_dim=128,
        num_heads=4,
        # act_layer=nn.GELU,
        embed_style="embed2",
        embed_norm=0,
        share_embed_head=False,
        num_views=3,
        use_large_patch=False,
        use_cls_token=False,
        **kwargs,
    ):
        super().__init__()
        vits = []
        for _ in range(num_views):
            vits.append(
                MinVit2(
                    embed_style=embed_style,
                    embed_dim=embed_dim,
                    embed_norm=embed_norm,
                    num_head=num_heads,
                    depth=depth,
                    num_channel=img_size[0],
                    img_h=img_size[1],
                    img_w=img_size[2],
                    patch_size=patch_size,
                    img_cond_steps=img_cond_steps,
                    use_cls_token=use_cls_token,
                )
            )
        self.vits = nn.Sequential(*vits)

        self.img_h = img_size[1]
        self.img_w = img_size[2]
        self.num_patch = sum([vit.num_patch for vit in self.vits])
        self.embed_dim = embed_dim

    def forward(self, obs) -> torch.Tensor:
        feats = []
        for v, vit in zip(obs.values(), self.vits):
            feats.append(vit(v))
        feats = torch.cat(feats, dim=1)
        return feats


class VitEncoder(nn.Module):
    def __init__(
        self,
        img_size,
        img_cond_steps=1,
        patch_size=8,
        depth=1,
        embed_dim=128,
        num_heads=4,
        # act_layer=nn.GELU,
        embed_style="embed2",
        embed_norm=0,
        share_embed_head=False,
        num_views=3,
        use_large_patch=False,
        **kwargs,
    ):
        super().__init__()
        self.vit = MinVit(
            embed_style=embed_style,
            embed_dim=embed_dim,
            embed_norm=embed_norm,
            num_head=num_heads,
            depth=depth,
            num_channel=img_size[0],
            img_h=img_size[1],
            img_w=img_size[2],
            share_embed_head=share_embed_head,
            num_views=num_views,
            img_cond_steps=img_cond_steps,
            patch_size=patch_size,
            use_large_patch=use_large_patch,
        )
        self.img_h = img_size[1]
        self.img_w = img_size[2]
        self.num_patch = self.vit.num_patch
        self.embed_dim = embed_dim

    def forward(self, obs) -> torch.Tensor:
        feats: torch.Tensor = self.vit.forward(obs)  # [batch, num_patch, embed_dim]
        return feats


class PatchEmbed1(nn.Module):
    def __init__(
        self, embed_dim, num_channel=3, img_h=240, img_w=320, patch_size=8, **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channel, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.num_patch = math.ceil(img_h / patch_size) * math.ceil(img_w / patch_size)
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor):
        y = self.conv(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c")
        return y


class PatchEmbed2(nn.Module):
    def __init__(
        self, embed_dim, use_norm, num_channel=3, img_h=240, img_w=320, patch_size=8
    ):
        super().__init__()
        coef = patch_size // 8
        ks1 = 8 * coef
        stride1 = 4 * coef
        ks2 = 3
        stride2 = 2
        layers = [
            nn.Conv2d(num_channel, embed_dim, kernel_size=ks1, stride=stride1),
            nn.GroupNorm(embed_dim, embed_dim) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=ks2, stride=stride2),
        ]
        self.embed = nn.Sequential(*layers)
        H1 = math.ceil((img_h - ks1) / stride1) + 1
        W1 = math.ceil((img_w - ks1) / stride1) + 1
        H2 = math.ceil((H1 - ks2) / stride2) + 1
        W2 = math.ceil((W1 - ks2) / stride2) + 1
        self.num_patch = H2 * W2
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor):
        y = self.embed(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c")
        return y


class MultiViewPatchEmbed(nn.Module):
    """
    A multi-view patch embedding module to process image patches across multiple views.

    Args:
        embed_dim (int): Dimension of the embedding space.
        num_channel (int): Number of input channels per image.
        img_h (int): Image height.
        img_w (int): Image width.
        embed_style (str): Embedding style, either 'embed1' or 'embed2'.
        num_views (int): Number of views.
        use_norm (bool): Whether to use normalization in embedding layers.
        share_embed_head (bool): If True, shares embedding layers across views.
        img_cond_steps (int): Number of conditioning steps for image inputs.
        patch_size (int): Size of each patch.
        use_large_patch (bool): If True, uses a large patch embedding strategy.
    """

    VALID_EMBED_STYLES: ClassVar[set] = {"embed1", "embed2"}

    def __init__(
        self,
        embed_dim: int,
        num_channel: int = 3,
        img_h: int = 240,
        img_w: int = 320,
        embed_style: str = "embed2",
        num_views: int = 3,
        use_norm: bool = True,
        share_embed_head: bool = False,
        img_cond_steps: int = 1,
        patch_size: int = 8,
        use_large_patch: bool = False,
    ):
        super().__init__()

        if embed_style not in self.VALID_EMBED_STYLES:
            raise ValueError(f"Invalid patch embedding style: {embed_style}")

        self.embed_dim = embed_dim
        self.img_h = img_h
        self.img_w = img_w
        self.num_views = num_views
        self.img_cond_steps = img_cond_steps
        self.use_large_patch = use_large_patch
        self.embed_style = embed_style
        self.share_embed_head = share_embed_head

        # Initialize embedding layers
        self.embed_layers = self._create_embed_layers(
            embed_dim=embed_dim,
            num_channel=num_channel,
            img_h=img_h,
            img_w=img_w,
            patch_size=patch_size,
            use_norm=use_norm,
        )

        self.num_patch = self._compute_num_patches()
        self.test_patch_embed()

    def _compute_num_patches(self) -> int:
        """Compute the total number of patches based on the embedding strategy."""
        base_num_patches = (
            self.embed_layers[0].num_patch
            if isinstance(self.embed_layers, nn.ModuleList)
            else self.embed_layers.num_patch
        )
        if self.use_large_patch:
            return base_num_patches * self.num_views * self.img_cond_steps
        return base_num_patches

    def _create_embed_layers(
        self,
        embed_dim: int,
        num_channel: int,
        img_h: int,
        img_w: int,
        patch_size: int,
        use_norm: bool,
    ) -> nn.Module:
        """Create embedding layers for the views."""
        embed_class = PatchEmbed1 if self.embed_style == "embed1" else PatchEmbed2
        if self.share_embed_head:
            return embed_class(
                embed_dim,
                num_channel=num_channel,
                img_h=img_h,
                img_w=img_w,
                patch_size=patch_size,
                use_norm=use_norm if self.embed_style == "embed2" else None,
            )
        return nn.ModuleList(
            [
                embed_class(
                    embed_dim,
                    num_channel=num_channel,
                    img_h=img_h,
                    img_w=img_w,
                    patch_size=patch_size,
                    use_norm=use_norm if self.embed_style == "embed2" else None,
                )
                for _ in range(self.num_views)
            ]
        )

    def forward_loop(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for individual embedding layers."""
        patch_embeddings = [
            embed(input_tensor)
            for input_tensor, embed in zip(inputs.values(), self.embed_layers)
        ]
        if self.use_large_patch:
            return einops.rearrange(
                torch.cat(patch_embeddings, dim=0),
                "(v b cs) p d -> b (cs v p) d",
                v=self.num_views,
                cs=self.img_cond_steps,
            )
        return torch.cat(patch_embeddings, dim=0)

    def forward_batch(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for shared embedding layers."""
        concatenated_inputs = torch.cat(list(inputs.values()), dim=0)
        embeddings = self.embed_layers(concatenated_inputs)
        if self.use_large_patch:
            return einops.rearrange(
                embeddings,
                "(v b cs) p d -> b (cs v p) d",
                v=self.num_views,
                cs=self.img_cond_steps,
            )
        return embeddings

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the module."""
        if self.share_embed_head:
            return self.forward_batch(inputs)
        return self.forward_loop(inputs)

    def test_patch_embed(self):
        """Run debug assertions."""
        inputs = {
            f"{i}": torch.rand(2 * self.img_cond_steps, 3, self.img_h, self.img_w)
            for i in range(self.num_views)
        }
        output = self.forward(inputs)
        expected_shape = (
            (2, self.num_patch, self.embed_dim)
            if self.use_large_patch
            else (
                2 * self.img_cond_steps * self.num_views,
                self.num_patch,
                self.embed_dim,
            )
        )
        assert output.size() == expected_shape, (
            f"Unexpected output shape: {output.size()}"
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head):
        super().__init__()
        assert embed_dim % num_head == 0

        self.num_head = num_head
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask):
        """
        x: [batch, seq, embed_dim]
        """
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(
            qkv, "b t (k h d) -> b k h t d", k=3, h=self.num_head
        ).unbind(1)
        # force flash/mem-eff attention, it will raise error if flash cannot be applied
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            attn_v = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, attn_mask=attn_mask
            )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_head, dropout):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_head)

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.mha(self.layer_norm1(x), attn_mask))
        x = x + self.dropout(self._ff_block(self.layer_norm2(x)))
        return x

    def _ff_block(self, x):
        x = self.linear2(nn.functional.gelu(self.linear1(x)))
        return x


class MinVit(nn.Module):
    def __init__(
        self,
        embed_style,
        embed_dim,
        embed_norm,
        num_head,
        depth,
        num_channel=3,
        img_h=240,
        img_w=320,
        num_views=3,
        share_embed_head=False,
        img_cond_steps=1,
        patch_size=8,
        use_large_patch=False,
    ):
        super().__init__()

        self.patch_embed = MultiViewPatchEmbed(
            embed_dim,
            num_channel=num_channel,
            img_h=img_h,
            img_w=img_w,
            embed_style=embed_style,
            num_views=num_views,
            share_embed_head=share_embed_head,
            use_norm=embed_norm,
            img_cond_steps=img_cond_steps,
            patch_size=patch_size,
            use_large_patch=use_large_patch,
        )
        self.img_cond_steps = img_cond_steps
        self.num_views = num_views
        self.use_large_patch = use_large_patch

        self.num_patch = self.patch_embed.num_patch

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patch, embed_dim))

        layers = [
            TransformerLayer(embed_dim, num_head, dropout=0) for _ in range(depth)
        ]

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        # weight init
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_vit_timm, self)

    def forward(self, x):
        x = self.patch_embed(
            x
        )  # (v*b*cs, num_patch, embed_dim) or (b, v*cs*num_patch, embed_dim)
        x = x + self.pos_embed
        x = self.net(x)
        x = self.norm(x)
        if not self.use_large_patch:
            x = einops.rearrange(
                x,
                "(v b cs) p d -> b (cs v p) d",
                v=self.num_views,
                cs=self.img_cond_steps,
            )
        return x


class MinVit2(nn.Module):
    def __init__(
        self,
        embed_style,
        embed_dim,
        embed_norm,
        num_head,
        depth,
        num_channel=3,
        img_h=240,
        img_w=320,
        num_views=3,
        share_embed_head=False,
        img_cond_steps=1,
        patch_size=8,
        use_large_patch=False,
        use_cls_token=False,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed2(
            embed_dim,
            embed_norm,
            num_channel=num_channel,
            img_h=img_h,
            img_w=img_w,
            patch_size=patch_size,
        )
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.img_cond_steps = img_cond_steps
        self.num_views = num_views
        self.use_large_patch = use_large_patch

        self.num_patch = (
            self.patch_embed.num_patch + 1
            if use_cls_token
            else self.patch_embed.num_patch
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patch, embed_dim))

        layers = [
            TransformerLayer(embed_dim, num_head, dropout=0) for _ in range(depth)
        ]

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        # weight init
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_vit_timm, self)

    def forward(self, x):
        x = self.patch_embed(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.net(x)
        x = self.norm(x)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def named_apply(
    fn, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def test_patch_embed():
    print("embed 1")
    embed = PatchEmbed1(128)
    x = torch.rand(10, 3, 96, 96)
    y = embed(x)
    print(y.size())

    print("embed 2")
    embed = PatchEmbed2(128, True)
    x = torch.rand(10, 3, 96, 96)
    y = embed(x)
    print(y.size())


def test_transformer_layer():
    embed = PatchEmbed1(128)
    x = torch.rand(10, 3, 96, 96)
    y = embed(x)
    print(y.size())

    transformer = TransformerLayer(128, 4, False, 0)
    z = transformer(y)
    print(z.size())


if __name__ == "__main__":
    # obs_shape = [3, 480, 640]
    # num_views = 1

    # enc = VitEncoder(
    #     obs_shape,
    #     num_channel=obs_shape[0],
    #     img_h=obs_shape[1],
    #     img_w=obs_shape[2],
    # )
    # x = {key: torch.rand(64, *obs_shape) * 255 for key in [f"view{i}" for i in range(num_views)]}

    # print("input size:", x.keys())
    # print("embed_size", enc.vit.patch_embed(x).size())
    # print("output size:", enc(x, flatten=False).size())
    # print("repr dim:", enc.repr_dim, ", real dim:", enc(x, flatten=True).size())
    pm = MultiViewPatchEmbed(
        128,
        num_channel=3,
        img_h=88,
        img_w=88,
        embed_style="embed2",
        num_views=2,
        use_norm=False,
        share_embed_head=False,
        img_cond_steps=2,
        patch_size=8,
        use_large_patch=True,
    )
    # pm = PatchEmbed2(128, use_norm=0, num_channel=3, img_h=240, img_w=320, patch_size=8)
    x = {f"{i}": torch.rand(10, 3, 88, 88) for i in range(3)}
    y = pm(x)
    print(y.size())
    breakpoint()
