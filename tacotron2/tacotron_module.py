from typing import Tuple, Union
import argparse
import torch
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pathlib import Path
import copy
import shutil
from tacotron2.factory import Factory
from tacotron2.hparams import HParams
from transformers import AdamW
from tacotron2.utils import seed_everything, get_cur_time_str, dump_json, prepare_dataloaders
from pytorch_lightning import loggers
from torch import optim


class TacotronModule(pl.LightningModule):
    def __init__(self, hparams: Union[dict, argparse.Namespace]):
        hparams.n_symbols = 152
        super(TacotronModule, self).__init__()
        self.model = Factory.get_class(f'tacotron2.models.{hparams.model_class_name}')(hparams)
        self.hparams = hparams
        self._train_dataloader = None
        self._valid_dataloader = None

    def prepare_data(self):
        self._train_dataloader, self._valid_dataloader = prepare_dataloaders(self.hparams)

    def forward(self, batch):
        _, loss = self.model(batch)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        _, loss = self.model(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._valid_dataloader

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.hparams.lr_reduce_factor,
            patience=self.hparams.lr_reduce_patience
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': self.trainer.accumulate_grad_batches,
            'reduce_on_plateau': True,
            'monitor': 'loss'
        }
        return [optimizer], [scheduler]


def get_trainer(_args: argparse.Namespace, _hparams: HParams) -> pl.Trainer:

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=_hparams.models_dir,
        verbose=True,
        save_top_k=_hparams.save_top_k
    )

    tb_logger_callback = loggers.TensorBoardLogger(
        save_dir=_hparams.tb_logdir
    )

    trainer_args = copy.deepcopy(_args.__dict__)
    trainer_args.update(
        {
            'max_epochs': _hparams.epochs,
            'device': _hparams.device,
            'check_val_every_n_epoch ': _hparams.iters_per_checkpoint,
            'amp_level': _hparams.fp16_opt_level or False,
            'gradient_clip_val': _hparams.grad_clip_thresh,
            'logger': tb_logger_callback,
            'checkpoint_callback': False, #model_checkpoint_callback,
            'show_progress_bar': True,
            'accumulate_grad_batches': _hparams.accum_steps
        }
    )
    _trainer = pl.Trainer(**trainer_args)
    return _trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Run Tacotron experiment')
    parser.add_argument(
        '--experiments_dir', type=Path, required=True, help='Root directory of all your experiments'
    )
    parser.add_argument(
        '--hparams_file', type=Path, required=True, help='Path to the hparams yaml file'
    )
    parser.add_argument(
        '--tb_logdir', type=Path, required=True, help='Tensorboard logs directory'
    )
    args = parser.parse_args()
    return args


def prepare_hparams(args):
    hparams = HParams.from_yaml(args.hparams_file)
    experiments_dir = args.experiments_dir
    experiment_id = get_cur_time_str()
    tb_logdir = args.tb_logdir / experiment_id
    hparams['tb_logdir'] = tb_logdir

    experiment_dir: Path = experiments_dir / experiment_id
    experiment_dir.mkdir(exist_ok=False, parents=True)
    shutil.copy(str(args.hparams_file), str(experiment_dir / 'hparams.yaml'))
    dump_json(args.__dict__, experiment_dir / 'arguments.json')
    models_dir = experiment_dir / 'models'
    models_dir.mkdir(exist_ok=False, parents=True)
    hparams['models_dir'] = models_dir

    return hparams


if __name__ == '__main__':
    args = parse_args()
    hparams = prepare_hparams(args)

    model= TacotronModule(hparams)
    trainer = get_trainer(_args=args, _hparams=hparams)
    trainer.fit(model)
