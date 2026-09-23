import collections
import inspect
import math
import sys
import os
import re
import json
import shutil
import time
import warnings
from pathlib import Path
import importlib.util
from transformers import Trainer
from transformers.integrations import TensorBoardCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import WEIGHTS_NAME
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    reissue_pt_warnings,
)

from transformers.utils import logging
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers.optimization import Adafactor, AdamW, get_scheduler
import copy
# Set path to SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np
from datetime import datetime
from filelock import FileLock

logger = logging.get_logger(__name__)


class ScalarTensorBoardCallback(TensorBoardCallback):
    """Write numeric trainer logs without text or HParams summaries."""

    def on_train_begin(self, args, state, control, **kwargs):
        log_dir = None
        if state.is_hyper_param_search and state.trial_name is not None:
            log_dir = os.path.join(args.logging_dir, state.trial_name)
        self._init_summary_writer(args, log_dir)


class CLTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_callback(TensorBoardCallback)
        self.add_callback(ScalarTensorBoardCallback)

    def _set_matrix_capture(self, enabled: bool) -> None:
        loss_fct = getattr(self.model, "loss_fct", None)
        if loss_fct is None:
            return
        loss_fct.capture_matrices = enabled
        if not enabled:
            loss_fct.captured_matrices = None

    def _should_save_matrices(self, global_step: int) -> bool:
        save_steps = getattr(self.args, "matrix_save_steps", 0) or 0
        return save_steps > 0 and global_step % save_steps == 0

    def _save_captured_matrices(self) -> None:
        loss_fct = getattr(self.model, "loss_fct", None)
        matrices = getattr(loss_fct, "captured_matrices", None)
        if not matrices:
            return

        save_dir = getattr(self.args, "matrix_save_dir", None)
        if save_dir is None:
            save_dir = os.path.join(self.args.output_dir, "matrices")
        os.makedirs(save_dir, exist_ok=True)

        output_path = os.path.join(save_dir, "step-{:08d}.pt".format(self.state.global_step))
        temporary_path = output_path + ".tmp"
        payload = {"global_step": self.state.global_step, **matrices}
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
        loss_fct.captured_matrices = None
        loss_fct.capture_matrices = False
        logger.info("Saved full-precision intermediate matrices to %s", output_path)

    def log(self, logs: Dict[str, float]) -> None:
        """Add the latest gradient norm to regular training logs."""
        if "loss" in logs and getattr(self, "_last_grad_norm", None) is not None:
            grad_norm = self._last_grad_norm
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.detach().item()
            logs["grad_norm"] = float(grad_norm)

        target_diff_mse = getattr(getattr(self.model, "loss_fct", None), "last_target_diff_mse", None)
        if "loss" in logs and target_diff_mse is not None:
            if isinstance(target_diff_mse, torch.Tensor):
                target_diff_mse = target_diff_mse.detach().item()
            logs["target_diff_mse"] = float(target_diff_mse)
        super().log(logs)

    def _compute_grad_norm(self, model) -> Optional[torch.Tensor]:
        """Compute a read-only norm when gradient clipping is disabled."""
        parameters = model.parameters()
        gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
        if not gradients:
            return None

        grad_norm = torch.norm(
            torch.stack([torch.norm(gradient, 2.0) for gradient in gradients]),
            2.0,
        )
        if self.use_amp:
            grad_norm = grad_norm / self.scaler.get_scale()
        return grad_norm

    @staticmethod
    def _load_stsb_dev_pairs():
        path = os.path.join(
            PATH_TO_DATA,
            "downstream",
            "STS",
            "STSBenchmark",
            "sts-dev.csv",
        )
        pairs = []
        with open(path, encoding="utf-8") as reader:
            for line in reader:
                fields = line.rstrip("\n").split("\t")
                pairs.append((
                    tuple(fields[5].split()),
                    tuple(fields[6].split()),
                    float(fields[4]),
                ))
        return pairs

    @staticmethod
    def _effective_rank(embeddings):
        """Compute exp(entropy) of the normalized singular-value spectrum."""
        if embeddings.ndim != 2:
            raise ValueError("Effective rank expects a 2D embedding matrix")

        singular_values = torch.linalg.svdvals(embeddings.float())
        singular_value_sum = singular_values.sum()
        if singular_value_sum <= torch.finfo(singular_values.dtype).eps:
            return 0.0

        probabilities = singular_values / singular_value_sum
        probabilities = probabilities[probabilities > 0]
        entropy = -(probabilities * probabilities.log()).sum()
        return entropy.exp().item()

    @staticmethod
    def _alignment_and_uniformity(stsb_pairs, embeddings):
        missing = {
            sentence
            for sentence_a, sentence_b, _ in stsb_pairs
            for sentence in (sentence_a, sentence_b)
            if sentence not in embeddings
        }
        if missing:
            raise RuntimeError(
                "Missing {} STS-B dev embeddings for alignment/uniformity".format(len(missing))
            )

        embeddings_a = torch.stack([embeddings[sentence_a] for sentence_a, _, _ in stsb_pairs])
        embeddings_b = torch.stack([embeddings[sentence_b] for _, sentence_b, _ in stsb_pairs])
        embeddings_a = torch.nn.functional.normalize(embeddings_a.float(), dim=-1)
        embeddings_b = torch.nn.functional.normalize(embeddings_b.float(), dim=-1)

        positive_mask = torch.tensor(
            [score > 4.0 for _, _, score in stsb_pairs],
            dtype=torch.bool,
            device=embeddings_a.device,
        )
        alignment = (embeddings_a[positive_mask] - embeddings_b[positive_mask]).norm(dim=1).pow(2).mean()

        all_embeddings = torch.cat((embeddings_a, embeddings_b), dim=0)
        squared_distances = torch.pdist(all_embeddings, p=2).pow(2)
        uniformity = squared_distances.mul(-2.0).exp().mean().log()
        return alignment.item(), uniformity.item()

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:

        stsb_pairs = self._load_stsb_dev_pairs()
        stsb_sentences = {sentence for pair in stsb_pairs for sentence in pair[:2]}
        stsb_embeddings = {}

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentence_keys = [tuple(sentence) for sentence in batch]
            sentences = [' '.join(s) for s in batch]
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output
            cpu_output = pooler_output.cpu()
            if params.current_task == "STSBenchmark":
                for sentence, embedding in zip(sentence_keys, pooler_output):
                    if sentence in stsb_sentences:
                        stsb_embeddings[sentence] = embedding.detach().float()
            return cpu_output

        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark', 'SICKRelatedness']
        if eval_senteval_transfer or self.args.eval_transfer:
            tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        self.model.eval()
        results = se.eval(tasks)
        
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
        alignment, uniformity = self._alignment_and_uniformity(stsb_pairs, stsb_embeddings)
        embedding_matrix = torch.stack([
            stsb_embeddings[sentence]
            for sentence in sorted(stsb_sentences)
        ])
        embedding_matrix = torch.nn.functional.normalize(embedding_matrix.float(), dim=-1)
        erank = self._effective_rank(embedding_matrix)

        metrics = {
            "eval_stsb_spearman": stsb_spearman,
            "eval_sickr_spearman": sickr_spearman,
            "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2,
            "eval_alignment": alignment,
            "eval_uniformity": uniformity,
            "eval_erank": erank,
        }
        if eval_senteval_transfer or self.args.eval_transfer:
            avg_transfer = 0
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                avg_transfer += results[task]['devacc']
                metrics['eval_{}'.format(task)] = results[task]['devacc']
            avg_transfer /= 7
            metrics['eval_avg_transfer'] = avg_transfer

        self.log(metrics)
        return metrics
        
    def _save_checkpoint(self, model, trial, metrics=None):
        """Save only the best validation checkpoint when a best metric is configured."""
        assert model is self.model, "internal model should be a reference to self.model"

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]
            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is not None
                and self.state.best_model_checkpoint is not None
                and not operator(metric_value, self.state.best_metric)
            ):
                return

            output_dir = self.args.output_dir
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = output_dir
        else:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            if self.hp_search_backend is not None and trial is not None:
                if self.hp_search_backend == HPSearchBackend.OPTUNA:
                    run_id = trial.number
                else:
                    from ray import tune
                    run_id = tune.get_trial_id()
                run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
                output_dir = os.path.join(self.args.output_dir, run_name, checkpoint_folder)
            else:
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            self.store_flos()

        self.save_model(output_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        with warnings.catch_warnings(record=True) as caught_warnings:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        reissue_pt_warnings(caught_warnings)
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        if metrics is None or self.args.metric_for_best_model is None:
            self._rotate_checkpoints(use_mtime=True)

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        
        The main difference between ours and Huggingface's original implementation is that we 
        also load model_args when reloading best checkpoints for evaluation.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)

            model = self.call_model_init(trial)
            model = model.to(self.args.device)

            self.model = model
            self.model_wrapped = model

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        model = self.model_wrapped

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = True
        self.state.is_world_process_zero = True

        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        self._last_grad_norm = None
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(train_dataloader) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            assert train_dataset_is_sized, "currently we only support sized dataloader!"

            inputs = None
            last_inputs = None
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                should_optimizer_step = (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                )
                capture_matrices = should_optimizer_step and self._should_save_matrices(self.state.global_step + 1)
                self._set_matrix_capture(capture_matrices)

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                tr_loss += self.training_step(model, inputs)
                self._total_flos += self.floating_point_ops(inputs)

                if should_optimizer_step:
                    # Gradient clipping
                    grad_norm = None
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.args.max_grad_norm
                        )
                    else:
                        grad_norm = self._compute_grad_norm(model)

                    self._last_grad_norm = grad_norm

                    # Optimizer step
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()

                    model.zero_grad()

                    self.state.global_step += 1
                    self._save_captured_matrices()
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint, model_args=self.model_args)
                self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)