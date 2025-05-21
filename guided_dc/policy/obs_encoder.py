import einops
import torch
import torch.nn as nn


class MultiviewObsEncoder(nn.Module):
    def __init__(
        self,
        img_model_name,
        feature_aggregation,
        obs_dim,
        aug=None,
        num_views=3,
        img_cond_steps=1,
        share_img_model=False,
        img_size=(3, 96, 96),
        obs_strat="concat",
        post_proc_dropout=0,
        feat_norm=None,
        token_dim=None,
        cond_mlp_dims=None,
        **kwargs,
    ):
        super().__init__()

        # 1. Augmentation
        if aug is not None:
            self.aug = torch.nn.Sequential(*aug)
            self.raw_img_size = img_size
            self.img_size = self.aug(torch.randint(0, 256, (1, *img_size))).shape[-3:]
        else:
            self.raw_img_size = img_size
            self.img_size = img_size

        # 2. Load image model
        self.img_model_name = img_model_name
        self.img_cond_steps = img_cond_steps
        self.num_views = num_views
        self.share_img_model = share_img_model
        self.img_model = self._load_img_model(**kwargs)

        # 3. Image feature aggregation
        self.img_feature_aggregation = feature_aggregation
        self.img_feature_aggregation.init_model(
            self.forward_img(self.example_img, aggregate=False)
        )
        self.img_feat_dim = self.forward_img(self.example_img).shape[-1]

        # 4. Obs encoder
        self.obs_strat = obs_strat
        self.cond_mlp_dims = cond_mlp_dims
        if obs_strat is None:
            obs_dim = 0
        if cond_mlp_dims is not None:
            from guided_dc.policy.common.mlp import ResidualMLP

            self.cond_mlp = ResidualMLP(
                dim_list=[obs_dim, *cond_mlp_dims],
                activation_type="Mish",
                out_activation_type="Identity",
            )
            obs_dim = cond_mlp_dims[-1]

        if obs_strat == "add_token":
            self._obs_proc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(obs_dim, self.img_feat_dim),
            )
            self.feat_dim = self.img_feat_dim
        elif obs_strat == "pad_img_tokens":
            self._obs_proc = nn.Dropout(p=0.2)
            self.feat_dim = obs_dim + self.img_feat_dim
        elif obs_strat == "concat":
            self.feat_dim = self.img_feat_dim + obs_dim
        elif obs_strat is None:
            self.feat_dim = self.img_feat_dim

        if obs_strat == "add_token" or obs_strat == "pad_img_tokens":
            # build (optional) token feature projection layer
            linear_proj = nn.Identity()
            if token_dim is not None and token_dim != self._token_dim:
                linear_proj = nn.Linear(self._token_dim, token_dim)
                self._token_dim = token_dim

            # build feature normalization layers
            if feat_norm == "batch_norm":
                norm = _BatchNorm1DHelper(self._token_dim)
            elif feat_norm == "layer_norm":
                norm = nn.LayerNorm(self._token_dim)
            else:
                assert feat_norm is None
                norm = nn.Identity()

            # final token post proc network
            self.post_proc = nn.Sequential(
                linear_proj, norm, nn.Dropout(post_proc_dropout)
            )
        else:
            self.post_proc = nn.Identity()

    def _load_img_model(self, **kwargs):
        if "dinov2" in self.img_model_name:
            from guided_dc.policy.vision.timm_models import DINOv2Encoder

            img_model = DINOv2Encoder
        elif "resnet" in self.img_model_name:
            from guided_dc.policy.vision.timm_models import ResNetEncoder

            img_model = ResNetEncoder
        elif "custom_vit" in self.img_model_name:
            from guided_dc.policy.vision.vit import VitEncoder2, validate_crop_size

            img_model = VitEncoder2
            validate_crop_size(self.aug, self.img_size, kwargs.get("patch_size"))
        else:
            raise ValueError(
                f"Model {self.img_model_name} not supported in {self.__class__.__name__}"
            )

        return img_model(
            img_size=self.img_size,
            num_views=self.num_views,
            img_cond_steps=self.img_cond_steps,
            share_img_model=self.share_img_model,
            **kwargs,
        )

    @property
    def example_img(self):
        return {
            f"{i}": torch.randint(0, 256, (1, self.img_cond_steps, *self.raw_img_size))
            for i in range(self.num_views)
        }

    def forward_img(self, img: dict, aggregate=True):
        for k in img.keys():
            img[k] = einops.rearrange(img[k], "b cs c h w -> (b cs) c h w")
        if self.aug:
            img = self.aug(img)
        # if 2 in img.keys():
        #     save_dir = "/n/fs/robot-data/guided-data-collection/training_images_3"
        #     os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        #     def get_next_index(save_path, prefix):
        #         existing_files = [
        #             f
        #             for f in os.listdir(save_path)
        #             if f.startswith(prefix) and f.endswith(".png")
        #         ]
        #         indices = [
        #             int(f[len(prefix) : -4])
        #             for f in existing_files
        #             if f[len(prefix) : -4].isdigit()
        #         ]
        #         return max(indices) + 1 if indices else 0

        #     for key in [0, 2]:
        #         save_path = os.path.join(save_dir, f"i{key}")
        #         os.makedirs(save_path, exist_ok=True)  # Ensure subdirectory exists
        #         index = get_next_index(
        #             save_path, "img_"
        #         )  # Get the next available index

        #         if index > 100:
        #             continue

        #         file_path = os.path.join(save_path, f"img_{index}.png")
        #         cv2.imwrite(
        #             file_path,
        #             cv2.cvtColor(
        #                 (img[key].cpu().numpy()[0].transpose(1, 2, 0) + 0.5) * 255,
        #                 cv2.COLOR_BGR2RGB,
        #             ),
        #         )
        #         print(f"Saved {file_path}")

        img_feat = self.img_model.forward(
            img
        )  # (bs, img_cond_steps * num_views * patch_nums, embed_dim)
        if aggregate:
            return self.img_feature_aggregation(img_feat)
        else:
            return img_feat

    def forward(self, img: dict, obs):
        img_feat = self.forward_img(img.copy())
        feat = self.combine(img_feat, obs.clone())
        return feat

    def combine(self, img_feat, obs, flatten=False):
        if self.obs_strat is not None:
            obs = obs.view(obs.shape[0], -1)
            if hasattr(self, "cond_mlp"):
                obs = self.cond_mlp(obs)
        else:
            return img_feat
        if self.obs_strat == "add_token":
            obs_token = self._obs_proc(obs)[:, None]
            feat = torch.cat((img_feat, obs_token), 1)
        elif self.obs_strat == "pad_img_tokens":
            obs_token = self._obs_proc(obs)
            obs_token = obs_token[:, None].repeat((1, img_feat.shape[1], 1))
            feat = torch.cat((img_feat, obs_token), 2)
        elif self.obs_strat == "concat":
            feat = torch.cat((img_feat, obs), dim=-1)
        feat = self.post_proc(feat)
        if flatten:
            return feat.reshape((feat.shape[0], -1))
        return feat


class _BatchNorm1DHelper(nn.BatchNorm1d):
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.transpose(1, 2)
            x = super().forward(x)
            return x.transpose(1, 2)
        return super().forward(x)
