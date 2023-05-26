import argparse
import sys
import time
import uuid
import traceback
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import wandb
from torchvision.transforms import functional

from trainer.FourBiTrainer import FourbiTrainingModule, set_seed
from trainer.Validator import Validator
from utils.WandbLog import WandbLog
from utils.htr_logging import get_logger, DEBUG

logger = get_logger('main')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    txt = '## CUDA is not available. Please use a GPU to run this code. ##'
    print('#' * len(txt))
    print(txt)
    print('#' * len(txt))


def train(config):
    wandb_log = None

    trainer = FourbiTrainingModule(config, device=device)

    if config['use_wandb']:  # Configure WandB
        tags = [Path(path).name for path in config['train_data_path']]
        wandb_id = wandb.util.generate_id()
        if trainer.checkpoint is not None and 'wandb_id' in trainer.checkpoint and not config['finetuning']:
            wandb_id = trainer.checkpoint['wandb_id']
        wandb_log = WandbLog(experiment_name=config['experiment_name'], tags=tags,
                             dir=config.wandb_dir, id=wandb_id)
        wandb_log.setup(config)

    trainer.model.to(device)

    if wandb_log:
        wandb_log.add_watch(trainer.model)

    threshold = config['threshold'] if config['threshold'] else 0.5
    train_validator = Validator(apply_threshold=config['apply_threshold_to_train'], threshold=threshold)

    try:
        start_time = time.time()
        patience = config['patience']

        for epoch in range(trainer.epoch, config['num_epochs']):
            wandb_logs = dict()
            wandb_logs['lr'] = trainer.optimizer.param_groups[0]['lr']
            trainer.epoch = epoch

            logger.info("Training has been started") if epoch == 1 else None
            logger.info(f"Epoch {trainer.epoch} of {trainer.num_epochs}")

            train_loss = 0.0

            trainer.model.train()
            train_validator.reset()
            data_times = []
            train_times = []
            start_data_time = time.time()
            start_epoch_time = time.time()

            for batch_idx, (train_in, train_out) in enumerate(trainer.train_data_loader):

                data_times.append(time.time() - start_data_time)
                start_train_time = time.time()
                inputs, outputs = train_in.to(device), train_out.to(device)

                trainer.optimizer.zero_grad()
                predictions = trainer.model(inputs)
                loss = trainer.criterion(predictions, outputs)
                loss.backward()
                trainer.optimizer.step()

                train_loss += loss.item()

                train_times.append(time.time() - start_train_time)

                with torch.no_grad():

                    metrics = train_validator.compute(predictions, outputs)

                    if batch_idx % config['train_log_every'] == 0:
                        size = batch_idx * len(inputs)
                        percentage = 100. * size / len(trainer.train_dataset)

                        elapsed_time = time.time() - start_time
                        time_per_iter = elapsed_time / (size + 1)
                        remaining_time = (len(trainer.train_dataset) - size - 1) * time_per_iter
                        eta = str(timedelta(seconds=remaining_time))

                        stdout = f"Train Loss: {loss.item():.6f} - PSNR: {metrics['psnr']:0.4f} -"
                        stdout += f" \t[{size} / {len(trainer.train_dataset)}]"
                        stdout += f" ({percentage:.2f}%)  Epoch eta: {eta}"
                        logger.info(stdout)

                start_data_time = time.time()

            avg_train_loss = train_loss / len(trainer.train_dataset)
            avg_train_metrics = train_validator.get_metrics()

            ##########################################
            #                  Train                 #
            ##########################################

            stdout = f"AVG training loss: {avg_train_loss:0.4f} - AVG training PSNR: {avg_train_metrics['psnr']:0.4f}"
            logger.info(stdout)

            wandb_logs['train/avg_loss'] = avg_train_loss
            wandb_logs['train/avg_psnr'] = avg_train_metrics['psnr']
            wandb_logs['train/data_time'] = np.array(data_times).mean()
            wandb_logs['train/time_per_iter'] = np.array(train_times).mean()

            original = inputs[0]
            pred = predictions[0].expand(3, -1, -1)
            output = outputs[0].expand(3, -1, -1)
            union = torch.cat((original, pred, output), 2)
            wandb_logs['Random Sample'] = wandb.Image(functional.to_pil_image(union), caption=f"Example")

            ##########################################
            #                  Test                  #
            ##########################################

            train_validator.reset()

            with torch.no_grad():
                start_test_time = time.time()
                test_metrics, test_loss, _ = trainer.test()

                wandb_logs['test/time'] = time.time() - start_test_time
                wandb_logs['test/avg_loss'] = test_loss
                wandb_logs['test/avg_psnr'] = test_metrics['psnr']

                ##########################################
                #               Validation               #
                ##########################################

                start_valid_time = time.time()
                valid_metrics, valid_loss, _ = trainer.validation()

                wandb_logs['valid/time'] = time.time() - start_valid_time
                wandb_logs['valid/avg_loss'] = valid_loss
                wandb_logs['valid/avg_psnr'] = valid_metrics['psnr']
                wandb_logs['valid/patience'] = patience

                trainer.psnr_list.append(valid_metrics['psnr'])
                psnr_running_mean = sum(trainer.psnr_list[-3:]) / len(trainer.psnr_list[-3:])
                reset_patience = False
                if valid_metrics['psnr'] > trainer.best_psnr:
                    trainer.best_psnr = valid_metrics['psnr']
                    reset_patience = True

                wandb_logs['Best PSNR'] = trainer.best_psnr

                if reset_patience:
                    patience = config['patience']

                    logger.info(f"Saving best model (valid) with valid_PSNR: {trainer.best_psnr:.02f}")

                    if epoch > 10:
                        trainer.save_checkpoints(
                            filename=config['experiment_name'] + f'{trainer.best_psnr:.02f}_best_psnr'
                        )

                else:
                    patience -= 1

            ##########################################
            #                 Generic                #
            ##########################################

            wandb_logs['epoch'] = trainer.epoch
            wandb_logs['epoch_time'] = time.time() - start_epoch_time

            stdout = f"Validation Loss: {valid_loss:.4f} - PSNR: {valid_metrics['psnr']:.4f}"
            stdout += f" Best Loss: {trainer.best_psnr:.3f}"
            logger.info(stdout)

            stdout = f"Test Loss: {test_loss:.4f} - PSNR: {test_metrics['psnr']:.4f}"
            stdout += f" Best Loss: {trainer.best_psnr:.3f}"
            logger.info(stdout)

            if config['lr_scheduler'] == 'plateau':
                trainer.lr_scheduler.step(metrics=psnr_running_mean)
            else:
                trainer.lr_scheduler.step()

            if wandb_log:
                wandb_log.on_log(wandb_logs)

            logger.info(f"Saving model...")
            trainer.save_checkpoints(filename=config['experiment_name'])
            logger.info('-' * 75)

            if patience == 0:
                stdout = f"There has been no update of Best PSNR value in the last {config['patience']} epochs."
                stdout += " Training will be stopped."
                logger.info(stdout)
                sys.exit()

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Training failed due to {e}")
    finally:
        logger.info("Training finished")
        sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiment_name', metavar='<name>', type=str,
                        help=f"The experiment name which will use on WandB")
    parser.add_argument('--path_checkpoint', type=str)
    parser.add_argument('-w', '--use_wandb', type=bool, default=not DEBUG)
    parser.add_argument('--n_blocks', type=int, default=9)
    parser.add_argument('--n_downsampling', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--operation', type=str, default='ffc', choices=['ffc', 'conv'])
    parser.add_argument('--skip', type=str, default='none', choices=['none', 'add', 'cat'])
    parser.add_argument('--resume', type=str, default='none')
    parser.add_argument('--wandb_dir', type=str, default='/tmp')
    parser.add_argument('--unet_layers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=742)
    parser.add_argument('--patience', type=int, default=60)
    parser.add_argument('--apply_threshold_to', type=str, default='test', choices=['none', 'val_test', 'test', 'all'])
    parser.add_argument('--loss', type=str, nargs='+', default=['CHAR'],
                        choices=['MSE', 'MAE', 'CHAR', 'BCE'])
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--lr_min', type=float, default=1.5e-5)
    parser.add_argument('--lr_scheduler', type=str, default='constant',
                        choices=['constant', 'exponential', 'multistep', 'linear', 'cosine', 'plateau', 'step'])
    parser.add_argument('--lr_scheduler_warmup', type=int, default=0)
    parser.add_argument('--lr_scheduler_kwargs', type=eval, default={})
    parser.add_argument('--load_data', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--train_transform_variant', type=str, default='latin',
                        choices=['threshold_mask', 'latin', 'none'])
    parser.add_argument('--merge_image', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--overlap_test', type=str, default='false', choices=['true', 'false'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--validation_dataset', type=str, required=True)
    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--exclude_datasets', type=str, nargs='+', default=[])
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--patch_size_raw', type=int)

    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=1)
    parser.add_argument('--kind_optimizer', type=str, default="Adam")

    parser.add_argument('--eps', type=float, default=1.0e-08)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--amsgrad', type=str, default='false', choices=['true', 'false'])

    parser.add_argument('--toggle_ffc', type=str, default='false', choices=['true', 'false'])

    args = parser.parse_args()

    logger.info("Start process ...")

    train_config = {'optimizer': {
        'eps': args.eps,
        'betas': [args.beta_1, args.beta_2],
        'weight_decay': args.weight_decay,
        'amsgrad': args.amsgrad == 'true'
    }, 'toggle_ffc': args.toggle_ffc == 'true', 'kind_optimizer': args.kind_optimizer,
        'input_channels': args.input_channels, 'output_channels': args.output_channels,
        'train_transform_variant': args.train_transform_variant, 'path_checkpoint': args.path_checkpoint,
        'init_conv_kwargs': {
            'ratio_gin': 0,
            'ratio_gout': 0
        },
        'down_sample_conv_kwargs': {
            'ratio_gin': 0,
            'ratio_gout': 0
        },
        'resnet_conv_kwargs': {
            'ratio_gin': 0.75,
            'ratio_gout': 0.75
        },
        'train_log_every': 100,
        'train_max_value': 500}

    if args.resume != 'none':
        checkpoint_path = Path(train_config['path_checkpoint'])
        checkpoints = sorted(checkpoint_path.glob(f"*_{args.resume}*.pth"))
        assert len(checkpoints) > 0, f"Found {len(checkpoints)} checkpoints with uuid {args.resume}"
        train_config['resume'] = checkpoints[0]
        args.experiment_name = checkpoints[0].stem.rstrip('_best_psnr')

    if args.experiment_name is None:
        exp_name = [
            str(uuid.uuid4())[:4]
        ]
        args.experiment_name = '_'.join(exp_name)

    train_config['experiment_name'] = args.experiment_name
    train_config['use_wandb'] = args.use_wandb
    train_config['wandb_dir'] = args.wandb_dir
    train_config['use_convolutions'] = args.operation == 'conv'
    train_config['skip_connections'] = args.skip
    train_config['unet_layers'] = args.unet_layers
    train_config['n_blocks'] = args.n_blocks
    train_config['n_downsampling'] = args.n_downsampling
    train_config['losses'] = args.loss
    train_config['lr_scheduler'] = args.lr_scheduler
    train_config['lr_scheduler_kwargs'] = args.lr_scheduler_kwargs
    train_config['lr_scheduler_warmup'] = args.lr_scheduler_warmup
    train_config['learning_rate'] = args.lr
    train_config['learning_rate_min'] = args.lr_min
    train_config['seed'] = args.seed

    args.datasets = {Path(dataset).name: dataset for dataset in args.datasets}
    args.datasets = {key: value for key, value in args.datasets.items() if key not in args.exclude_datasets}
    args.train_data_path = [dataset for key, dataset in args.datasets.items() if key != args.test_dataset]
    args.test_data_path = [args.datasets[args.test_dataset]]

    args.train_data_path = [dataset for key, dataset in args.datasets.items() if
                            key != args.test_dataset and key != args.validation_dataset]
    args.valid_data_path = [args.datasets[args.validation_dataset]]
    train_config['valid_data_path'] = args.valid_data_path
    train_config['train_data_path'] = args.train_data_path
    train_config['test_data_path'] = args.test_data_path
    assert len(train_config['test_data_path']) > 0, f"Test dataset {args.test_dataset} not found in {args.datasets}"
    train_config['merge_image'] = args.merge_image == 'true'

    train_config['train_kwargs'] = {
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'batch_size': args.batch_size
    }

    train_config['valid_kwargs'] = {
        'shuffle': False,
        'num_workers': args.num_workers,
        'batch_size': 1
    }

    train_config['test_kwargs'] = {
        'shuffle': False,
        'num_workers': args.num_workers,
        'batch_size': 1
    }

    train_config[
        'train_transform_variant'] = args.train_transform_variant if args.train_transform_variant != 'none' else None

    train_config['train_batch_size'] = train_config['train_kwargs']['batch_size']
    train_config['valid_batch_size'] = train_config['valid_kwargs']['batch_size']
    train_config['test_batch_size'] = train_config['test_kwargs']['batch_size']

    train_config['num_epochs'] = args.epochs
    train_config['patience'] = args.patience

    train_config['threshold'] = args.threshold
    train_config['load_data'] = args.load_data == 'true'

    train_config['apply_threshold_to_train'] = True
    train_config['apply_threshold_to_valid'] = True
    train_config['apply_threshold_to_test'] = True
    if args.apply_threshold_to == 'none':
        train_config['apply_threshold_to_train'] = False
        train_config['apply_threshold_to_valid'] = False
        train_config['apply_threshold_to_test'] = False
    elif args.apply_threshold_to == 'val_test':
        train_config['apply_threshold_to_train'] = False
    elif args.apply_threshold_to == 'test':
        train_config['apply_threshold_to_train'] = False
        train_config['apply_threshold_to_valid'] = False

    if args.overlap_test == 'true':
        train_config['test_stride'] = args.patch_size // 2
    else:
        train_config['test_stride'] = args.patch_size

    train_config['train_patch_size'] = args.patch_size
    train_config['train_patch_size_raw'] = args.patch_size_raw if args.patch_size_raw else args.patch_size + 128
    train_config['valid_patch_size'] = args.patch_size
    train_config['test_patch_size'] = args.patch_size

    set_seed(args.seed)

    train(train_config)
    sys.exit()
