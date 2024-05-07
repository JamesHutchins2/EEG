class Config():
    def __init__(self):
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 500
        self.warmup_epochs = 40
        self.batch_size = 32
        self.clip_grad = 0.8
        
        # Model Parameters
        self.mask_ratio = 0.75
        self.patch_size = 4
        self.embed_dim = 1024 
        self.decoder_embed_dim = 512 
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0

        #Other
        self.save_dir = 'checkpoints'
        self.plot_dir = 'plots'
        self.seed = 42
        self.accum_iter = 1
        self.log_steps = 20
        self.data_len = 512
        self.local_rank = 0


class Config_MBM_EEG():
    def __init__(self):
        self.lr = 2.5e-4
        