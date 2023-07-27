from ml_collections.config_dict import ConfigDict


def get_config():

    cfg = ConfigDict()

    # ----------------
    # Train
    # ----------------

    training = cfg.training = ConfigDict()
    training.num_epochs = 200
    training.batch_size = 4
    training.save_ckpt_freq = 50
    training.eval_freq = 10

    # ----------------
    # Model
    # ----------------

    model = cfg.model = ConfigDict()
    model.clip_grad_norm = 1.
    model.embed_dim = 64

    # ----------------
    # Optimization
    # ----------------

    cfg.optim = optim = ConfigDict()
    optim.optimizer = 'AdamW'
    optim.schedule = 'CosineAnnealingLR'
    optim.grad_clip = 1.
    optim.initial_lr = 0.0001
    optim.weight_decay = 0.0001
    optim.min_lr = 0.001 * optim.initial_lr
    optim.warmup_epochs = 15

    # ----------------
    # Data
    # ----------------

    cfg.data = data = ConfigDict()
    data.num_workers = 2
    data.prefetch_factor = 1
    data.num_known = 32
    data.mask = 'random'
    data.resolution = 128
    data.path = '/storage/data/tongshq/dataset/mice/npy'
    data.len_sig = 1000
    data.num_sig = 128
    data.num_known = 32

    cfg.seed = 42
    cfg.distributed = True
    cfg.use_deterministic_algorithms = True
    cfg.debug = False

    return cfg
