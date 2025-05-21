"""
Parent validation agent class.

"""

import logging
import os
import random

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F

from guided_dc.utils.tensor_utils import batch_apply

log = logging.getLogger(__name__)


class BaseValAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        # set seed
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.model = hydra.utils.instantiate(cfg.policy)
        self.device = self.model.device
        self.normalization_stats_path = cfg.normalization_stats_path
        self.normalization_stats = np.load(
            self.normalization_stats_path, allow_pickle=True
        )
        self.obs_min = torch.tensor(
            self.normalization_stats["obs_min"], dtype=torch.float, device=self.device
        )
        self.obs_max = torch.tensor(
            self.normalization_stats["obs_max"], dtype=torch.float, device=self.device
        )
        self.action_min = torch.tensor(
            self.normalization_stats["action_min"],
            dtype=torch.float,
            device=self.device,
        )
        self.action_max = torch.tensor(
            self.normalization_stats["action_max"],
            dtype=torch.float,
            device=self.device,
        )

    def normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 1

    def normalize_action(self, action):
        return (
            2 * (action - self.action_min) / (self.action_max - self.action_min + 1e-6)
            - 1
        )

    def unnorm_obs(self, obs):
        return (obs + 1) * (self.obs_max - self.obs_min + 1e-6) / 2 + self.obs_min

    def unnorm_action(self, action):
        return (action + 1) * (
            self.action_max - self.action_min + 1e-6
        ) / 2 + self.action_min


class ValLossAgent(BaseValAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Build dataset
        self.dataset_val = hydra.utils.instantiate(cfg.task.val_dataset)
        self.dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.cfg.train.val_batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            sampler=None,
        )

    def run(self):
        loss_vals = []
        self.model.eval()

        for epoch, batch_val in enumerate(self.dataloader_val):
            batch_val = batch_apply(
                batch_val, lambda x: x.to(self.device, non_blocking=True)
            )
            batch_val = batch_apply(batch_val, lambda x: x.float())

            loss_val = self.calculate_validation_loss(batch_val)
            loss_vals.append(loss_val.item())
            print(f"Epoch {epoch}, Loss: {loss_val.item()}")
        np.savez(os.path.join(self.cfg.logdir, "val_loss.npz"), losses=loss_vals)

    def calculate_validation_loss(self, batch_val, loss_type="action_mse"):
        if loss_type == "action_mse":
            cond = batch_val[1]
            gt_actions = self.normalize_action(batch_val[0])
            cond["state"] = self.normalize_obs(cond["state"])
            actions = self.model.sample(cond, deterministic=True).trajectories
            return ((actions - gt_actions) ** 2).mean()
        else:
            raise NotImplementedError


def mean_knn_distance(a, bs, k):
    # Normalize the vectors for cosine similarity computation
    a_norm = F.normalize(a, p=2, dim=1)
    bs_norm = F.normalize(bs, p=2, dim=1)
    # Compute cosine similarity (higher means more similar)
    cosine_sim = torch.matmul(a_norm, bs_norm.T)
    print(cosine_sim.shape)
    # Convert similarity to distance (lower means more similar)
    cosine_dist = 1 - cosine_sim
    # Sort distances and take the mean of k-nearest neighbors for each input
    knn_distances = torch.topk(cosine_dist, k, largest=False).values
    print(knn_distances.shape)
    mean_knn_distances = knn_distances.mean(dim=1)
    print(mean_knn_distances.shape)

    return mean_knn_distances


def vis(batch, path, idx):
    for i in range(len(batch[0])):
        cv2.imwrite(
            f"{path}/side_{idx}_{i}.png",
            cv2.cvtColor(
                batch[0][i].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR
            ),
        )
        if 1 in batch:
            cv2.imwrite(
                f"{path}/wrist_{idx}_{i}.png",
                cv2.cvtColor(
                    batch[1][i].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR
                ),
            )


def process_batch(batch, device):
    batch = {k: v.to(device, non_blocking=True).float() for k, v in batch.items()}
    return batch


class AnomalyAgent(BaseValAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Build dataset
        self.cfg = cfg
        self.dataset_train = hydra.utils.instantiate(cfg.task.train_dataset)
        self.nominal_dataloader = torch.utils.data.DataLoader(
            hydra.utils.instantiate(cfg.task.nominal_dataset),
            batch_size=self.cfg.train.val_batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            sampler=None,
        )

        self.nominal_val_dataloader = torch.utils.data.DataLoader(
            hydra.utils.instantiate(cfg.task.nominal_val_dataset),
            batch_size=self.cfg.train.val_batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            sampler=None,
        )

        self.anomaly_dataloader = torch.utils.data.DataLoader(
            hydra.utils.instantiate(cfg.task.anomaly_dataset),
            batch_size=20,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            sampler=None,
        )
        self.model.eval()

    @torch.no_grad()
    def calculate_nominal_features(self):
        nominal_features = []
        os.makedirs("aaa/nominal", exist_ok=True)
        for idx, batch in enumerate(self.nominal_dataloader):
            batch = process_batch(batch, self.device)
            nominal_features.append(self.model.model.obs_encoder.forward_img(img=batch))
            vis(batch, "aaa/nominal", idx)
        return torch.cat(nominal_features, dim=0)

    @torch.no_grad()
    def calculate_nominal_distance(self, r_nom, k):
        distances = []
        os.makedirs("aaa/nominal_val", exist_ok=True)
        for idx, batch in enumerate(self.nominal_val_dataloader):
            batch = process_batch(batch, self.device)
            features = self.model.model.obs_encoder.forward_img(img=batch)
            dist = mean_knn_distance(features, self.nominal_features, k=k)
            distances.append(dist.cpu().numpy())
            vis(batch, "aaa/nominal_val", idx)
        distances = np.concatenate(distances)
        print(f"Min nominal distance: {distances.min()}")
        print(f"Max nominal distance: {distances.max()}")
        tau = self.get_quantile(
            distances, (len(distances) + 1) * r_nom / len(distances)
        )
        return distances, tau

    def normalize_distance(self, distance):
        all_dist = np.concatenate([self.nominal_distance, distance])
        norm_factor = 1 / all_dist.max()
        return distance * norm_factor

    def z_score(self, distance):
        return (distance - self.nominal_distance.mean()) / self.nominal_distance.std()

    @torch.no_grad()
    def run(self):
        k_range = [1, 5, 10]
        distances = {k: [] for k in k_range}
        normalized_distances = {k: [] for k in k_range}
        z_scores = {k: [] for k in k_range}
        anomaly_rate = {k: [] for k in k_range}
        tau = {k: [] for k in k_range}
        os.makedirs("aaa/anomaly", exist_ok=True)
        self.nominal_features = self.calculate_nominal_features()

        for k in k_range:
            self.nominal_distance, self.tau = self.calculate_nominal_distance(
                self.cfg.task.r_nom, k=k
            )

            for idx, batch_val in enumerate(self.anomaly_dataloader):
                # S_val: real demos with different variations compared to training
                # S_nominal: part of training set
                batch_val = process_batch(batch_val, self.device)
                with torch.no_grad():
                    # cv2.imwrite("img.png", cv2.cvtColor(batch_val[0][0][0].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
                    features = self.model.model.obs_encoder.forward_img(img=batch_val)
                    dist = mean_knn_distance(features, self.nominal_features, k=k)
                    distances[k].append(dist.cpu().numpy())
                    # normalized_distances.append(dist.cpu().numpy() / self.nominal_distance.mean())
                    normalized_distances[k].append(
                        self.normalize_distance(dist.cpu().numpy())
                    )
                    z_scores[k].append(self.z_score(dist.cpu().numpy()))
                vis(batch_val, "aaa/anomaly", idx)

            distances[k] = np.concatenate(distances[k])
            normalized_distances[k] = np.concatenate(normalized_distances[k])
            z_scores[k] = np.concatenate(z_scores[k])
            anomaly_rate[k] = (
                distances[k][distances[k] > self.tau].shape[0] / distances[k].shape[0]
            )
            tau[k] = self.tau
            print(f"Anomaly rate: {anomaly_rate[k]}")
            print(f"Tau: {self.tau}")
            print(f"Min distance: {distances[k].min()}")
            print(f"Max distance: {distances[k].max()}")
            print(f"Mean distance: {distances[k].mean()}")
            print(f"Mean normalized distance: {normalized_distances[k].mean()}")

        np.savez(
            os.path.join(self.cfg.logdir, "embedding_distance_.npz"),
            distances=distances,
            normalized_distances=normalized_distances,
            z_scores=z_scores,
            anomaly_rate=anomaly_rate,
            tau=tau,
        )

    def get_quantile(self, distances, quantile):
        return np.quantile(distances, quantile)
