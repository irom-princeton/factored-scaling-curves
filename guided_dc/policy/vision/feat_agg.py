"""
Additional implementation of the ViT image encoder from https://github.com/hengyuan-hu/ibrl/tree/main and https://github.com/real-stanford/diffusion_policy/tree/main

"""

import einops
import torch
import torch.nn as nn


def reshape_to_batch(x, img_cond_steps, num_views):
    x = einops.rearrange(
        x,
        "(v b cs) d-> b (cs v d)",
        cs=img_cond_steps,
        v=num_views,
    )
    return x


def reshape_resnet_output(x, img_cond_steps, num_views):
    if len(x.shape) == 4:
        x = torch.flatten(
            x, start_dim=-2
        )  # (batch*img_cond_steps*num_views, emb_dim, h*w)
    x = torch.transpose(x, 1, 2)  # (batch*img_cond_steps*num_views, h*w, emb_dim)
    x = einops.rearrange(
        x,
        "(v b cs) hw d-> b (cs v hw) d",
        cs=img_cond_steps,
        v=num_views,
    )
    return x


class FeatAgg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def init_model(self, x):
        pass


class Compress(FeatAgg):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

    def init_model(self, x):
        input_dim = x.shape[-1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
        )

    def forward(self, x):
        raise NotImplementedError


##### ViT #####


class SpatialEmb(FeatAgg):
    def __init__(
        self,
        prop_dim,
        proj_dim,
        dropout,
    ):
        super().__init__()

        self.prop_dim = prop_dim
        self.proj_dim = proj_dim
        self.dropout = dropout

    def init_model(self, x):
        token_num = x.shape[1]
        num_proj = x.shape[-1]

        proj_in_dim = token_num + self.prop_dim

        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU(inplace=True),
        )
        self.weight = nn.Parameter(torch.zeros(1, num_proj, self.proj_dim))
        self.dropout = nn.Dropout(self.dropout)
        nn.init.normal_(self.weight)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor):
        feat = feat.transpose(1, 2)

        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated_prop), dim=-1)

        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        z = self.dropout(z)
        return z


class CompressViT(Compress):
    def __init__(self, embed_dim, dropout):
        super().__init__(embed_dim, dropout)

    def forward(self, x):
        x = x.flatten(1, -1)
        return self.model(x)


class Concat(FeatAgg):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.flatten(1)


class ClsToken(FeatAgg):
    def __init__(self, num_views, img_cond_steps):
        super().__init__()
        self.num_views = num_views
        self.img_cond_steps = img_cond_steps

    def init_model(self, x):
        self.token_num = x.shape[1]

    def forward(self, x):
        cls_token_idx_list = [
            self.token_num // self.num_views // self.img_cond_steps * i
            for i in range(self.num_views * self.img_cond_steps)
        ]
        x = x[:, cls_token_idx_list, :]
        x = x.flatten(1)
        return x


class Identity(FeatAgg):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


##### ResNet #####


class AvgPool(FeatAgg):
    def __init__(self, img_cond_steps, num_views, flatten=True):
        super().__init__()
        if flatten:
            self.model = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        else:
            self.model = nn.AdaptiveAvgPool2d((1, 1))
        self.img_cond_steps = img_cond_steps
        self.num_views = num_views

    def forward(self, x):
        assert len(x.shape) == 4  # (batch*img_cond_steps*num_views, emb_dim, h, w)
        return reshape_to_batch(self.model(x), self.img_cond_steps, self.num_views)


class CompressResnet(Compress):
    def __init__(self, embed_dim, dropout):
        super().__init__(embed_dim, dropout)

    def forward(self, x):
        assert len(x.shape) == 4  # (batch*img_cond_steps*num_views, emb_dim, h, w)
        x = reshape_resnet_output(x).mean(dim=1)
        return self.model(x)


class Max(FeatAgg):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert len(x.shape) == 4  # (batch*img_cond_steps*num_views, emb_dim, h, w)
        x = reshape_resnet_output(x)
        return x.max(dim=1)


class Mean(FeatAgg):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=1)


class ToToken(FeatAgg):
    def __init__(self, img_cond_steps, num_views, avgpool_first=True):
        super().__init__()
        self.img_cond_steps = img_cond_steps
        self.num_views = num_views
        self.avgpool_first = avgpool_first
        if avgpool_first:
            self.model = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

    def forward(self, x):
        if self.avgpool_first:
            x = self.model(x)[..., None]
        y = reshape_resnet_output(x, self.img_cond_steps, self.num_views)
        return y


if __name__ == "__main__":
    totoken = ToToken(2, 3, avgpool_first=True)
    x = torch.randn(12, 512, 12, 13)
    breakpoint()
