# python3.7
"""Configuration for training StyleGAN on FF-HQ (1024) dataset.

All settings are particularly used for one replica (GPU), such as `batch_size`
and `num_workers`.
"""

runner_type = 'StyleGAN2Runner'
gan_type = 'stylegan2'
resolution = 512
batch_size = 2
val_batch_size = 2
total_img = 25000_000
num_gpus = 2
total_iters = int(total_img / (batch_size * num_gpus))

# Training dataset is repeated at the beginning to avoid loading dataset
# repeatedly at the end of each epoch. This can save some I/O time.
DATA = '/home/data'
data = dict(
    num_workers=4,
    repeat=1,
    train=dict(root_dir=DATA, resolution=resolution, mirror=0.5, data_format='rdir'),
    val=dict(root_dir=DATA, resolution=resolution, data_format='rdir'),
)

controllers = dict(
    RunningLogger=dict(every_n_iters=10),
    Snapshoter=dict(every_n_iters=500, first_iter=True, num=200),
    FIDEvaluator=dict(every_n_iters=5000, first_iter=True, num=50000),
    Checkpointer=dict(every_n_iters=5000, first_iter=True),
)

modules = dict(
    discriminator=dict(
        model=dict(gan_type=gan_type, resolution=resolution),
        lr=dict(lr_type='FIXED'),
        opt=dict(opt_type='Adam', base_lr=1e-3, betas=(0.0, 0.99)),
        kwargs_train=dict(),
        kwargs_val=dict(),
    ),
    generator=dict(
        model=dict(gan_type=gan_type, resolution=resolution),
        lr=dict(lr_type='FIXED'),
        opt=dict(opt_type='Adam', base_lr=1e-3, betas=(0.0, 0.99)),
        kwargs_train=dict(w_moving_decay=0.995, style_mixing_prob=0.9,
                          trunc_psi=1.0, trunc_layers=0, randomize_noise=True),
        kwargs_val=dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False),
        g_smooth_img=10_000,
    )
)

loss = dict(
    type='LogisticGANLoss',
    d_loss_kwargs=dict(r1_gamma=10.0),
    g_loss_kwargs=dict(),
)