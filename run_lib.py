import logging
import os
import torch
from torch import nn
from torch.nn import functional as F
from utils.data import load_data, data_prefetcher, get_mask_fn
from utils.time import time_calculator
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from utils.optim import get_optim
from utils.utils import seed_everything
from torch.utils.tensorboard import SummaryWriter
from models.uformer import MMUformer
from utils.utils import AverageMeter
from utils.PAT import DAS_operator
from models.losses import MixedLoss

def train(config, workdir, train_dir='train'):
    """Runs the training pipeline.

    Args:
    config: ml_collections.ConfigDict(), config of the project
    workdir: directory to store files.
    """

    assert config.distributed, "Distributed train is needed!"
    torch.backends.cudnn.benchmark = True
    workdir = os.path.join(workdir, train_dir)

    # -------------------
    # Initialize DDP
    # -------------------

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # -------------------
    # seeds
    # -------------------

    seed_everything(config.seed + rank)

    if config.use_deterministic_algorithms:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    # -----------------------------
    # Create directories for data
    # -----------------------------

    log_dir = os.path.join(workdir, 'tensorboard')
    ckpt_dir = os.path.join(workdir, 'ckpt')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # -------------------
    # Loggers
    # -------------------

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s: %(message)s')

    fh = logging.FileHandler(os.path.join(
        workdir, 'train_log.log'), encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # -------------------
    # Load data
    # -------------------

    if rank == 0:
        logger.info('Loading data...')

    train_loader, test_loader, train_sampler, test_sampler = load_data(config)
    dist.barrier()

    if rank == 0:
        logger.info('Data loaded.')
    
    # -------------------
    # Define DAS
    # -------------------

    DAS = DAS_operator(config)

    # -------------------
    # Initialize model
    # -------------------

    if rank == 0:
        logger.info('Begin model initialization...')

    model = MMUformer(
        img_size=config.data.resolution,
        embed_dim=config.model.embed_dim,
        win_size=8,
        token_projection='linear',
        token_mlp='leff',
        modulator=True
    )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # If model has BNs...

    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[rank])
    model_without_ddp = model.module

    if rank == 0:
        logger.info("Models initialized.")

    dist.barrier()

    # -------------------
    # define optimization
    # -------------------

    if rank == 0:
        logger.info('Handling optimizations...')

    optimizer, scheduler = get_optim(model, config)
    criterion = MixedLoss().cuda()

    if rank == 0:
        logger.info('Completed.')

    # -------------------
    # training loop
    # -------------------

    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    time_logger = time_calculator()

    best_loss = 999999999.
    iters_per_epoch = len(train_loader)

    mask_fn = get_mask_fn(config)

    dist.barrier()
    torch.cuda.empty_cache()

    for epoch in range(config.training.num_epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        model.train()
        train_loss_epoch = AverageMeter()

        if rank == 0:
            logger.info(f'Start training epoch {epoch + 1}.')

        # ----------------------------
        # initialize data prefetcher
        # ----------------------------

        train_prefetcher = data_prefetcher(train_loader, rank)
        img, sig = train_prefetcher.next()
        i = 0

        # ----------------------------
        # run the training process
        # ----------------------------

        while img is not None:
            mask = mask_fn(sig)
            masked_sig = mask * sig
            noisy_img = DAS.signal_to_image(masked_sig)
            with torch.cuda.amp.autocast(enabled=False):
                img_hat, sig_hat = model(noisy_img, masked_sig)
                loss = torch.mean(criterion(img_hat, img) + criterion(sig_hat, sig) + criterion(DAS.signal_to_image(sig_hat), img_hat) + criterion(DAS.image_to_signal(img_hat).unsqueeze(1), sig_hat))

            train_loss_epoch.update(loss.item(), img.shape[0])

            if rank == 0:
                writer.add_scalar("Loss", train_loss_epoch.val,
                                  epoch * iters_per_epoch + i)

            logger.info(
                f'Epoch: {epoch + 1}/{config.training.num_epochs}, Iter: {i + 1}/{iters_per_epoch}, Loss: {train_loss_epoch.val:.6f}, Device: {rank}')

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            if config.model.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.model.clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            img, sig = train_prefetcher.next()
            i += 1

        scheduler.step()

        dist.barrier()
        if rank == 0:
            logger.info(
                f'Epoch: {epoch + 1}/{config.training.num_epochs}, Avg loss: {train_loss_epoch.avg:.4f}, Time: {time_logger.time_length()}')

        # save snapshot periodically

        if (epoch + 1) % config.training.save_ckpt_freq == 0:
            if rank == 0:
                logger.info(f'Saving snapshot at epoch {epoch + 1}')
                snapshot = {
                    'epoch': epoch + 1,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict()
                }
                torch.save(snapshot, os.path.join(
                    ckpt_dir, f'{epoch+1}_loss_{train_loss_epoch.avg:.2f}.pth'))

        is_best = train_loss_epoch.avg < best_loss
        if is_best:
            best_loss = train_loss_epoch.avg
            if rank == 0:
                logger.info(
                    f'Saving best model state dict at epoch {epoch + 1}.')
                torch.save(model_without_ddp.state_dict, os.path.join(ckpt_dir, 'best.pth'))

        # Report loss on eval dataset periodically

        if (epoch + 1) % config.training.eval_freq == 0:
            if rank == 0:
                logger.info(f'Start evaluate at epoch {epoch + 1}.')

            eval_model = model_without_ddp
            # eval_model = model_without_ddp
            with torch.no_grad():
                eval_model.eval()
                iters_per_eval = len(test_loader)
                eval_loss_epoch = AverageMeter()

                # ----------------------------
                # initialize data prefetcher
                # ----------------------------

                test_prefetcher = data_prefetcher(test_loader, rank)
                img, sig = test_prefetcher.next()
                i = 0

                while img is not None:
                    mask = mask_fn(sig)
                    masked_sig = mask * sig
                    noisy_img = DAS.signal_to_image(masked_sig)
                    with torch.cuda.amp.autocast(enabled=False):
                        img_hat, sig_hat = model(noisy_img, masked_sig)
                        loss = torch.mean(criterion(img_hat, img) + criterion(sig_hat, sig) + criterion(DAS.signal_to_image(sig_hat), img_hat) + criterion(DAS.image_to_signal(img_hat).unsqueeze(1), sig_hat))

                    eval_loss_epoch.update(loss.item(), img.shape[0])
                    logger.info(
                        f'Epoch: {epoch + 1}/{config.training.num_epochs}, Iter: {i + 1}/{iters_per_eval}, Loss: {eval_loss_epoch.val:.6f}, Time: {time_logger.time_length()}, Device: {rank}')

                    img, sig = test_prefetcher.next()
                    i += 1

                if rank == 0:
                    writer.add_scalar('Eval loss', eval_loss_epoch.avg, epoch)

            if rank == 0:
                logger.info(
                    f'Epoch: {epoch + 1}/{config.training.num_epochs}, Avg eval loss: {eval_loss_epoch.avg:.4f}, Time: {time_logger.time_length()}')

        dist.barrier()

    if rank == 0:
        logger.info(
            f'Training complete.\nTotal time:, {time_logger.time_length()}')


def eval(config, workdir, eval_dir='eval'):
    pass
