from typing import Callable, Dict, Tuple, Union

import einops
import timm
import torch
from torch import nn


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


class TimmEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",  # or 'resnet18'
        pretrained: bool = False,
        share_img_model: bool = False,
        num_views: int = 3,
        img_cond_steps: int = 1,
        frozen: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.img_cond_steps = img_cond_steps
        self.num_views = num_views
        self.model_name = model_name
        self.share_img_model = share_img_model

        # Validate model name
        if not ("resnet" in model_name or "dino" in model_name):
            raise ValueError(f"Unsupported model name: {model_name}")

        # Initialize model(s)
        self.model = self._create_models(
            model_name=model_name,
            pretrained=pretrained,
            share_img_model=share_img_model,
            num_views=num_views,
            **kwargs,
        )

        # Handle frozen parameters
        if frozen:
            assert pretrained, "Frozen models must be pretrained."
            self._freeze_model()

        # Optionally apply LoRA
        if use_lora:
            self._apply_lora(lora_rank)

    def _create_models(
        self,
        model_name: str,
        pretrained: bool,
        share_img_model: bool,
        num_views: int,
        **kwargs,
    ) -> Union[nn.Module, nn.ModuleList]:
        if share_img_model:
            return self._build_model(model_name, pretrained, **kwargs)
        else:
            return nn.ModuleList(
                [
                    self._build_model(model_name, pretrained, **kwargs)
                    for _ in range(num_views)
                ]
            )

    def _build_model(self, **kwargs) -> nn.Module:
        raise NotImplementedError

    def _freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _apply_lora(self, lora_rank: int):
        import peft

        lora_config = peft.LoraConfig(
            r=lora_rank,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["qkv"],
        )
        self.model = peft.get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.share_img_model:
            return self.forward_batch(x)
        else:
            return self.forward_loop(x)

    def forward_loop(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        y = [model(v) for v, model in zip(x.values(), self.model)]
        return torch.cat(
            y, dim=0
        )  # resnet: num_views*bs*img_cond_steps, embed_dim, h, w

    def forward_batch(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # x is a dict with keys as view names and values as images. Concatenate the images along the batch dimension and reshape to (batch, patch_nums, embed_dim) after passing through the embedding layer.
        y = torch.cat(list(x.values()), dim=0)  # num_views*bs*img_cond_steps, c, h, w
        return self.model(y)  # resnet: num_views*bs*img_cond_steps, embed_dim, h, w


class DINOv2Encoder(TimmEncoder):
    def __init__(
        self,
        model_name="vit_base_patch14_dinov2.lvd142m",  # 'resnet18
        pretrained=False,
        share_img_model=False,
        num_views=3,
        img_cond_steps=1,
        frozen=False,
        use_lora=False,
        drop_path_rate=0.0,
        img_size=(3, 96, 96),
        lora_rank=8,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            share_img_model=share_img_model,
            num_views=num_views,
            img_cond_steps=img_cond_steps,
            frozen=frozen,
            use_lora=use_lora,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
            lora_rank=lora_rank,
        )

        print("Extra kwargs:", kwargs)

    def _build_model(
        self,
        model_name: str,
        pretrained: bool,
        img_size: Tuple[int, int, int],
        drop_path_rate: float,
    ) -> nn.Module:
        return timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool="",
            num_classes=0,
            img_size=img_size[1],
            drop_path_rate=drop_path_rate,
        )

    def forward_loop(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        y = [model(v) for v, model in zip(x.values(), self.model)]
        # vit: bs*img_cond_steps, patch_nums, embed_dim; resnet: bs*img_cond_steps, embed_dim
        y = torch.cat(y, dim=0)
        y = einops.rearrange(
            y,
            "(v b cs) p d -> b (cs v p) d",
            cs=self.img_cond_steps,
            v=self.num_views,
        )
        return y

    def forward_batch(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # x is a dict with keys as view names and values as images. Concatenate the images along the batch dimension and reshape to (batch, patch_nums, embed_dim) after passing through the embedding layer.
        y = torch.cat(list(x.values()), dim=0)  # num_views*bs*img_cond_steps, c, h, w
        y = self.model(y)
        y = einops.rearrange(
            y,
            "(v b cs) p d -> b (cs v p) d",
            cs=self.img_cond_steps,
            v=self.num_views,
        )
        return y


class ResNetEncoder(TimmEncoder):
    def __init__(
        self,
        model_name="resnet18",
        pretrained=False,
        share_img_model=False,
        num_views=3,
        img_cond_steps=1,
        use_group_norm=True,
        frozen=False,
        use_lora=False,
        lora_rank=8,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            share_img_model=share_img_model,
            num_views=num_views,
            img_cond_steps=img_cond_steps,
            frozen=frozen,
            use_lora=use_lora,
            lora_rank=lora_rank,
        )

        print("Extra kwargs:", kwargs)

        if use_group_norm and not pretrained:
            # assert not pretrained
            if isinstance(self.model, nn.ModuleList):
                for i in range(num_views):
                    self.model[i] = replace_submodules(
                        root_module=self.model[i],
                        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(
                            num_groups=x.num_features // 16, num_channels=x.num_features
                        ),
                    )
            else:
                self.model = replace_submodules(
                    root_module=self.model,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=x.num_features // 16, num_channels=x.num_features
                    ),
                )

    def _build_model(
        self,
        model_name: str,
        pretrained: bool,
        # use_r3m: bool = True,
    ) -> nn.Module:
        # if use_r3m:
        #     import r3m

        #     model = r3m.load_r3m(model_name)
        #     model = model.module
        #     model = model.convnet.to("cpu")
        # else:
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool="",
            num_classes=0,
        )
        return nn.Sequential(*list(model.children())[:-2])  # Exclude FC layers


if __name__ == "__main__":
    model = timm.create_model("resnet18", global_pool="", num_classes=0)
    x = torch.randn(7, 3, 278, 256)
    y = model(x)
    breakpoint()
