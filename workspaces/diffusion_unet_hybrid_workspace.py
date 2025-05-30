import os
import pathlib
import sys


# Set root path for module imports and working directory
def configure_paths():
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(root_dir))
    os.chdir(str(root_dir))


configure_paths()

# Standard imports
import copy
import pathlib
import random

# Third-party imports
import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Project-specific modules
from datasets.base import BaseDataset
from models.ema import EMAModel
from runners.base import BaseRunner
from utils.checkpoints import TopKCheckpointManager
from utils.logging import JsonLogger
from utils.lr_scheduler import get_scheduler
from utils.pytorch import dict_apply, optimizer_to
from workspaces.base import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)
        self._set_seed(cfg.training.seed)

        # Model and optimizer
        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            self._maybe_resume()

        train_loader, val_loader, normalizer = self._prepare_data(cfg)
        self.model.set_normalizer(normalizer)
        if self.ema_model:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = self._init_lr_scheduler(cfg, len(train_loader))
        ema = self._init_ema(cfg)
        env_runner = self._init_env(cfg)
        wandb_run = self._init_logging(cfg)

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        device = torch.device(cfg.training.device)
        self._move_to_device(device)

        if cfg.training.debug:
            self._enable_debug_mode(cfg)

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            self._train(
                cfg,
                train_loader,
                val_loader,
                ema,
                env_runner,
                topk_manager,
                wandb_run,
                json_logger,
                lr_scheduler,
                device,
            )

    def _maybe_resume(self):
        path = self.get_checkpoint_path()
        if path.is_file():
            print(f"Resuming from checkpoint: {path}")
            self.load_checkpoint(path=path)

    def _prepare_data(self, cfg):
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_loader = DataLoader(dataset, **cfg.dataloader)
        val_loader = DataLoader(dataset.get_validation_dataset(), **cfg.val_dataloader)
        return train_loader, val_loader, dataset.get_normalizer()

    def _init_lr_scheduler(self, cfg, num_batches):
        return get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(num_batches * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

    def _init_ema(self, cfg):
        return (
            hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            if cfg.training.use_ema
            else None
        )

    def _init_env(self, cfg):
        runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir
        )
        assert isinstance(runner, BaseImageRunner)
        return runner

    def _init_logging(self, cfg):
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})
        return wandb_run

    def _move_to_device(self, device):
        self.model.to(device)
        if self.ema_model:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

    def _enable_debug_mode(self, cfg):
        cfg.training.num_epochs = 2
        cfg.training.max_train_steps = 3
        cfg.training.max_val_steps = 3
        cfg.training.rollout_every = 1
        cfg.training.checkpoint_every = 1
        cfg.training.val_every = 1
        cfg.training.sample_every = 1

    def _train(
        self,
        cfg,
        train_loader,
        val_loader,
        ema,
        env_runner,
        topk_manager,
        wandb_run,
        json_logger,
        lr_scheduler,
        device,
    ):
        sampling_batch = None
        for _ in range(cfg.training.num_epochs):
            train_losses = []
            step_log = {}
            for batch_idx, batch in enumerate(
                tqdm.tqdm(
                    train_loader,
                    desc=f"Epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                )
            ):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                if sampling_batch is None:
                    sampling_batch = batch

                raw_loss = self.model.compute_loss(batch)
                loss = raw_loss / cfg.training.gradient_accumulate_every
                loss.backward()

                if self.global_step % cfg.training.gradient_accumulate_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                    if ema:
                        ema.step(self.model)

                train_losses.append(raw_loss.item())
                step_log.update(
                    {
                        "train_loss": raw_loss.item(),
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                )

                if batch_idx != (len(train_loader) - 1):
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1

                if cfg.training.max_train_steps and batch_idx >= (
                    cfg.training.max_train_steps - 1
                ):
                    break

            step_log["train_loss"] = np.mean(train_losses)
            self._evaluate(
                cfg, val_loader, ema, env_runner, sampling_batch, step_log, device
            )
            self._save_checkpoints(cfg, step_log, topk_manager)

            wandb_run.log(step_log, step=self.global_step)
            json_logger.log(step_log)
            self.global_step += 1
            self.epoch += 1

    def _evaluate(
        self, cfg, val_loader, ema, env_runner, sampling_batch, step_log, device
    ):
        policy = self.ema_model if ema else self.model
        policy.eval()

        if self.epoch % cfg.training.rollout_every == 0:
            step_log.update(env_runner.run(policy))

        if self.epoch % cfg.training.val_every == 0:
            with torch.no_grad():
                val_losses = []
                for batch_idx, batch in enumerate(
                    tqdm.tqdm(
                        val_loader,
                        desc=f"Validation {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    )
                ):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    loss = self.model.compute_loss(batch)
                    val_losses.append(loss)
                    if cfg.training.max_val_steps and batch_idx >= (
                        cfg.training.max_val_steps - 1
                    ):
                        break
                if val_losses:
                    step_log["val_loss"] = torch.mean(torch.tensor(val_losses)).item()

        if self.epoch % cfg.training.sample_every == 0:
            with torch.no_grad():
                batch = dict_apply(
                    sampling_batch, lambda x: x.to(device, non_blocking=True)
                )
                result = policy.predict_action(batch["obs"])
                mse = torch.nn.functional.mse_loss(
                    result["action_pred"], batch["action"]
                )
                step_log["train_action_mse_error"] = mse.item()

    def _save_checkpoints(self, cfg, step_log, topk_manager):
        if self.epoch % cfg.training.checkpoint_every != 0:
            return

        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint()
        if cfg.checkpoint.save_last_snapshot:
            self.save_snapshot()

        metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
        if topk_ckpt_path:
            self.save_checkpoint(path=topk_ckpt_path)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parents[1] / "config"),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
