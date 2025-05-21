"""
Parent pre-training agent class.

"""

import logging
import os
import random
import time
from copy import deepcopy

import GPUtil
import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

import wandb
from guided_dc.utils.scheduler import CosineAnnealingWarmupRestarts
from guided_dc.utils.tensor_utils import batch_apply

log = logging.getLogger(__name__)

DEVICE = "cuda"


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        log.error(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch, device="cuda"):
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)


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
    print(knn_distances.shape)

    return mean_knn_distances


class Timer:
    def __init__(self):
        self._start = time.time()

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff


class EMA:
    """
    Empirical moving average
    """

    def __init__(self, cfg):
        super().__init__()
        self.beta = cfg.decay

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class TrainAgent:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._setup_seed()
        self._setup_device()
        self._setup_logging()
        self._setup_model()
        self._setup_datasets()
        self._setup_dataloaders()
        self._setup_optimizer()
        self.reset_parameters()

        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.update_ema_freq = cfg.train.update_ema_freq
        self.epoch_start_ema = cfg.train.epoch_start_ema
        self.stop_criteria = cfg.train.stop_criteria
        self.save_model_freq = cfg.train.save_model_freq

        # Simulation eval params
        self.eval_epoch = cfg.train.eval_epoch
        self.additional_save_epochs = cfg.train.additional_save_epochs

    def _setup_seed(self):
        """Sets random seeds for reproducibility."""
        self.seed = self.cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _setup_device(self):
        """Configures device settings including GPU allocation."""
        self.num_gpus = torch.cuda.device_count()
        self.gpu_id = int(self.cfg.gpu_id)
        self.device = torch.device(
            f"cuda:{self.gpu_id}" if self.num_gpus > 0 else "cpu"
        )

        if self.cfg.get("debug", False) and self.gpu_id == 0:
            torch.cuda.memory._record_memory_history(max_entries=100000)

    def _setup_logging(self):
        # Wandb
        self.use_wandb = self.cfg.get("wandb", None)
        if self.use_wandb and self.gpu_id == 0:
            wandb.init(
                entity=self.cfg.wandb.entity,
                project=self.cfg.wandb.project,
                name=self.cfg.wandb.run,
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )
        self.log_freq = self.cfg.train.get("log_freq", 1)

        # Logging, checkpoints
        self.logdir = self.cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _setup_model(self):
        """Builds the model and wraps it with DDP if using multiple GPUs."""
        self.model = hydra.utils.instantiate(self.cfg.policy).to(self.device)

        if self.num_gpus > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP

            logging.info(f"Using {self.num_gpus} GPUs. Current GPU: {self.gpu_id}")
            self.model = DDP(
                self.model,
                device_ids=[self.gpu_id],
                gradient_as_bucket_view=True,
                static_graph=True,
            )

        self.ema = EMA(self.cfg.ema)
        self.ema_model = deepcopy(
            self.model.module if self.num_gpus > 1 else self.model
        )
        log.info(f"Device for EMA model: {self.ema_model.device}")
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated()
            log.info(
                f"Allocated GPU memory after loading model: {allocated_memory / 1024 / 1024 / 1024} GB"
            )
            GPUtil.showUtilization(all=True)

    def _setup_datasets(self):
        """Initializes datasets for training and validation."""
        self.dataset_train = hydra.utils.instantiate(self.cfg.task.train_dataset)
        self.dataset_val = (
            hydra.utils.instantiate(self.cfg.task.val_dataset)
            if self.cfg.task.get("val_dataset_name")
            else None
        )

        if self.cfg.task.do_anomaly:
            self.nominal_dataset = hydra.utils.instantiate(
                self.cfg.task.nominal_dataset
            )
            self.anomaly_dataset = hydra.utils.instantiate(
                self.cfg.task.anomaly_dataset
            )
            self.nominal_val_dataset = hydra.utils.instantiate(
                self.cfg.task.nominal_val_dataset
            )
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated()
            log.info(
                f"Allocated GPU memory after loading dataset: {allocated_memory / 1024 / 1024 / 1024} GB"
            )
            GPUtil.showUtilization(all=True, useOldCode=False)

    def _setup_dataloaders(self):
        """Creates dataloaders with appropriate samplers and settings."""
        store_gpu = self.cfg.train.store_gpu
        assert not store_gpu or self.dataset_train.device != "cpu", (
            self.dataset_train.device
        )

        sampler = (
            self._create_weighted_sampler()
            if self.cfg.train.get("use_weighted_sampler", False)
            else None
        )

        self.dataloader_train = self._create_dataloader(
            self.dataset_train, self.cfg.train.batch_size, store_gpu, sampler
        )
        self.dataloader_val = (
            self._create_dataloader(
                self.dataset_val, self.cfg.train.val_batch_size, store_gpu
            )
            if self.dataset_val
            else None
        )

        if self.cfg.task.do_anomaly:
            self.nominal_dataloader = self._create_dataloader(
                self.nominal_dataset, self.cfg.train.val_batch_size, store_gpu
            )
            self.nominal_val_dataloader = self._create_dataloader(
                self.nominal_val_dataset, self.cfg.train.val_batch_size, store_gpu
            )
            self.anomaly_dataloader = self._create_dataloader(
                self.anomaly_dataset, 20, store_gpu
            )

        log.info(f"Using {'GPU' if store_gpu else 'CPU'} memory for dataset")

        # if "train_split" in self.cfg.train and self.cfg.train.train_split < 1:
        #     val_indices = self.dataset_train.set_train_val_split(
        #         self.cfg.train.train_split
        #     )
        #     self.dataset_val = deepcopy(self.dataset_train)
        #     self.dataset_val.set_indices(val_indices)
        #     assert not store_gpu or self.dataset_val.device != "cpu", (
        #         self.dataset_val.device
        #     )

        #     self.dataloader_val = self._create_dataloader(
        #         self.dataset_val, self.val_batch_size, store_gpu
        #     )

    def _setup_optimizer(self):
        """Initializes optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=self.cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=self.cfg.train.learning_rate,
            min_lr=self.cfg.train.lr_scheduler.min_lr,
            warmup_steps=self.cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0,
        )

    def _create_dataloader(self, dataset, batch_size, store_gpu, sampler=None):
        num_workers = 0 if store_gpu else self.cfg.train.get("num_workers", 4)
        persistent_workers = (
            self.cfg.train.get("persistent_workers", False) if not store_gpu else False
        )
        use_distributed = self.num_gpus > 1
        shuffle = not use_distributed and not sampler
        sampler = DistributedSampler(dataset) if use_distributed else sampler

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=not store_gpu,
            persistent_workers=persistent_workers,
            sampler=sampler,
        )

    def _create_weighted_sampler(self):
        """Create a weighted sampler that samples sim trajectories always with probability 0.2, and real trajectories with probability 0.8."""
        sim_traj_flags = self.dataset_train.sim_traj_flags
        num_sim, num_real = (
            np.sum(sim_traj_flags),
            len(sim_traj_flags) - np.sum(sim_traj_flags),
        )
        weight_sim, weight_real = (
            (0.2 / num_sim if num_sim > 0 else 0),
            (0.8 / num_real if num_real > 0 else 0),
        )
        weights = torch.tensor(
            [weight_sim if flag else weight_real for flag in sim_traj_flags],
            device=self.device,
        )
        logging.info(
            f"Weighted Sampler - Sim: {weight_sim:.6f}, Real: {weight_real:.6f}, Total: {len(weights)}"
        )
        return WeightedRandomSampler(weights, len(weights), replacement=True)

    def reset_parameters(self):
        self.ema_model.load_state_dict(
            self.model.module.state_dict()
            if self.num_gpus > 1
            else self.model.state_dict()
        )

    def step_ema(self):
        if self.epoch < self.epoch_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(
            self.ema_model, self.model.module if self.num_gpus > 1 else self.model
        )

    def save_model(self, ckpt_name=None):
        """
        Save model and ema to disk;
        """
        data = {
            "epoch": self.epoch,
            "model": (
                self.model.module.state_dict()
                if self.num_gpus > 1
                else self.model.state_dict()
            ),
            "ema": self.ema_model.state_dict(),
            # "cfg": self.cfg,
        }

        if not ckpt_name:
            ckpt_name = f"state_{self.epoch}.pt"
        savepath = os.path.join(self.checkpoint_dir, ckpt_name)
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")

    def load(self, epoch):
        """
        loads model and ema from disk
        """
        loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")
        data = torch.load(loadpath, weights_only=True)
        self.epoch = data["epoch"]
        if self.num_gpus > 1:
            self.model.module.load_state_dict(data["model"])
        else:
            self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

    def _handle_distributed_training(self, epoch, sampler):
        if self.num_gpus > 1:
            dist.barrier()
            sampler.set_epoch(epoch)

    def _train_one_epoch(self, num_gradient_steps, stop_training):
        self.model.train()
        loss_train_epoch = []

        for batch in self.dataloader_train:
            batch = batch_apply(batch, lambda x: x.to(self.device, non_blocking=True))
            batch = batch_apply(batch, lambda x: x.float())

            loss = self.model(*batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            loss_train_epoch.append(loss.item())
            if len(loss_train_epoch) % self.update_ema_freq == 0:
                self.step_ema()

            num_gradient_steps += 1
            if self._should_stop_training(num_gradient_steps):
                stop_training = True
                break

        return np.mean(loss_train_epoch), num_gradient_steps, stop_training

    def _should_stop_training(self, num_gradient_steps, max_gradient_steps=50000):
        return (
            self.stop_criteria == "num_gradient_steps"
            and num_gradient_steps >= max_gradient_steps
        )

    def _validate_if_needed(self, epoch, stop_training):
        if self.stop_criteria == "epoch":
            val_flag = self.dataloader_val and epoch in self.eval_epoch
        elif self.stop_criteria == "num_gradient_steps":
            val_flag = self.dataloader_val and stop_training
        else:
            raise ValueError(f"Unknown stop criteria {self.stop_criteria}")

        if not val_flag:
            return None

        assert self.dataloader_val

        self.model.eval()
        loss_val_epoch = []
        with torch.no_grad():
            self._handle_distributed_training(epoch - 1, self.dataloader_val.sampler)
            for batch in self.dataloader_val:
                batch = batch_apply(
                    batch, lambda x: x.to(self.device, non_blocking=True)
                )
                batch = batch_apply(batch, lambda x: x.float())
                loss_val_epoch.append(self.calculate_validation_loss(batch).item())
        loss_val = np.mean(loss_val_epoch)
        np.savez(
            os.path.join(self.cfg.logdir, f"val_loss_{epoch}.npz"),
            losses=loss_val_epoch,
        )
        self.model.train()
        # if loss_val:
        #     if loss_val < best_val_loss:
        #         best_val_loss = loss_val
        #         self.save_model(ckpt_name=f"best_{best_val_loss:.4f}_{self.epoch}")
        return loss_val

    def _save_model_if_needed(self, epoch, stop_training):
        save_flag = (
            stop_training
            if self.stop_criteria == "num_gradient_steps"
            else (
                epoch % self.save_model_freq == 0
                or epoch == self.n_epochs
                or epoch in self.additional_save_epochs
            )
        ) and self.gpu_id == 0

        if save_flag:
            self.save_model()

    def _evaluate_if_needed(self, epoch, stop_training):
        eval_flag = (
            stop_training
            if self.stop_criteria == "num_gradient_steps"
            else epoch in self.eval_epoch
        ) and self.gpu_id == 0

        if eval_flag and self.cfg.task.do_anomaly:
            self.model.eval()
            self.calculate_emb_distance(epoch)
            self.model.train()

    def _log_metrics(self, epoch, loss_train, loss_val, timer):
        if epoch % self.log_freq == 0 and self.gpu_id == 0:
            log.info(f"{epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}")
            if self.use_wandb:
                wandb.log({"loss - train": loss_train}, step=epoch, commit=True)
                if loss_val is not None:
                    wandb.log({"loss - val": loss_val}, step=epoch, commit=True)

    def _debug_memory(self, epoch):
        if self.cfg.get("debug", False) and epoch == 2 and self.gpu_id == 0:
            try:
                torch.cuda.memory._dump_snapshot(
                    f"{self.cfg.train.batch_size}_mem_debug.pickle"
                )
            except Exception as e:
                logging.error(f"Failed to capture memory snapshot {e}")
            torch.cuda.memory._record_memory_history(enabled=None)

    def run(self):
        timer = Timer()
        self.epoch = 1
        # best_val_loss = np.inf
        num_gradient_steps = 0
        stop_training = False
        for epoch in range(self.n_epochs):
            # multi-gpu chore
            self._handle_distributed_training(epoch, self.dataloader_train.sampler)
            loss_train, num_gradient_steps, stop_training = self._train_one_epoch(
                num_gradient_steps, stop_training
            )
            loss_val = self._validate_if_needed(self.epoch, stop_training)
            self.lr_scheduler.step()
            self._save_model_if_needed(self.epoch, stop_training)
            self._evaluate_if_needed(self.epoch, stop_training)
            self._log_metrics(self.epoch, loss_train, loss_val, timer)
            self._debug_memory(self.epoch)

            self.epoch += 1
            if self.num_gpus > 1:
                dist.barrier()

            if stop_training:
                break

    def calculate_validation_loss(self, batch_val, loss_type="action_mse"):
        if loss_type == "action_mse":
            actions = self.model.sample(batch_val[1], deterministic=True).trajectories
            return ((actions - batch_val[0]) ** 2).mean()
        elif loss_type == "diffusion":
            return self.model(*batch_val)

    def normalize_distance(self, distance):
        all_dist = np.concatenate([self.nominal_distance, distance])
        norm_factor = 1 / all_dist.max()
        return distance * norm_factor

    # def z_score(self, distance):
    #     return (distance - self.nominal_distance.mean()) / self.nominal_distance.std()

    @torch.no_grad()
    def calculate_nominal_features(self):
        nominal_features = []
        for batch in self.nominal_dataloader:
            batch = self.process_batch(batch)
            nominal_features.append(self.model.model.obs_encoder.forward_img(img=batch))
        return torch.cat(nominal_features, dim=0)

    @torch.no_grad()
    def calculate_nominal_distance(self, r_nom, k):
        distances = []
        for batch in self.nominal_val_dataloader:
            batch = self.process_batch(batch)
            features = self.model.model.obs_encoder.forward_img(img=batch)
            dist = mean_knn_distance(features, self.nominal_features, k=k)
            distances.append(dist.cpu().numpy())
        distances = np.concatenate(distances)
        print(f"Min nominal distance: {distances.min()}")
        print(f"Max nominal distance: {distances.max()}")
        tau = self.get_quantile(
            distances, (len(distances) + 1) * r_nom / len(distances)
        )
        return distances, tau

    def process_batch(self, batch):
        batch = {
            k: v.to(self.device, non_blocking=True).float() for k, v in batch.items()
        }
        return batch

    @torch.no_grad()
    def calculate_emb_distance(self, epoch):
        k_range = [1, 5, 10]
        distances = {k: [] for k in k_range}
        normalized_distances = {k: [] for k in k_range}
        # z_scores = {k: [] for k in k_range}
        anomaly_rate = {k: [] for k in k_range}
        tau = {k: [] for k in k_range}
        self.nominal_features = self.calculate_nominal_features()
        for k in k_range:
            self.nominal_distance, self.tau = self.calculate_nominal_distance(
                self.cfg.task.r_nom, k=k
            )

            for batch_val in self.anomaly_dataloader:
                batch_val = self.process_batch(batch_val)
                with torch.no_grad():
                    # cv2.imwrite("img.png", cv2.cvtColor(batch_val[0][0][0].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
                    features = self.model.model.obs_encoder.forward_img(img=batch_val)
                    dist = mean_knn_distance(features, self.nominal_features, k=k)
                    distances[k].append(dist.cpu().numpy())
                    # normalized_distances.append(dist.cpu().numpy() / self.nominal_distance.mean())
                    normalized_distances[k].append(
                        self.normalize_distance(dist.cpu().numpy())
                    )
                    # z_scores[k].append(self.z_score(dist.cpu().numpy()))
                # vis(batch_val, "aaa/anomaly", idx)

            distances[k] = np.concatenate(distances[k])
            normalized_distances[k] = np.concatenate(normalized_distances[k])
            # z_scores[k] = np.concatenate(z_scores[k])
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
            os.path.join(self.cfg.logdir, f"embedding_distance_{epoch}.npz"),
            distances=distances,
            normalized_distances=normalized_distances,
            # z_scores=z_scores,
            anomaly_rate=anomaly_rate,
            tau=tau,
        )

    def get_quantile(self, distances, quantile):
        return np.quantile(distances, quantile)
