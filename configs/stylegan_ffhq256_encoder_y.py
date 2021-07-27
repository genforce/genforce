# python3.7
"""Configuration for training StyleGAN Encoder on FF-HQ (256) dataset.

All settings are particularly used for one replica (GPU), such as `batch_size`
and `num_workers`.
"""

gan_model_path = 'checkpoints/stylegan_ffhq256.pth'
perceptual_model_path = 'checkpoints/vgg16.pth'

runner_type = 'EncoderRunner'
gan_type = 'stylegan'
resolution = 256
batch_size = 12
val_batch_size = 25
total_img = 14000_000
space_of_latent = 'y'

# Training dataset is repeated at the beginning to avoid loading dataset
# repeatedly at the end of each epoch. This can save some I/O time.
data = dict(
    num_workers=4,
    repeat=500,
    # train=dict(root_dir='data/ffhq', resolution=resolution, mirror=0.5),
    # val=dict(root_dir='data/ffhq', resolution=resolution),
    train=dict(root_dir='data/', data_format='list',
               image_list_path='data/ffhq/ffhq_train_list.txt',
               resolution=resolution, mirror=0.5),
    val=dict(root_dir='data/', data_format='list',
             image_list_path='./data/ffhq/ffhq_val_list.txt',
             resolution=resolution),
)

controllers = dict(
    RunningLogger=dict(every_n_iters=50),
    Snapshoter=dict(every_n_iters=10000, first_iter=True, num=200),
    Checkpointer=dict(every_n_iters=10000, first_iter=False),
)

modules = dict(
    discriminator=dict(
        model=dict(gan_type=gan_type, resolution=resolution),
        lr=dict(lr_type='ExpSTEP', decay_factor=0.8, decay_step=36458 // 2),
        opt=dict(opt_type='Adam', base_lr=1e-4, betas=(0.9, 0.99)),
        kwargs_train=dict(),
        kwargs_val=dict(),
    ),
    generator=dict(
        model=dict(gan_type=gan_type, resolution=resolution, repeat_w=True),
        kwargs_val=dict(randomize_noise=False),
    ),
    encoder=dict(
        model=dict(gan_type=gan_type, resolution=resolution, network_depth=18,
                   latent_dim = [1024] * 8 + [512, 512, 256, 256, 128, 128],
                   num_latents_per_head=[4, 4, 6],
                   use_fpn=True,
                   fpn_channels=512,
                   use_sam=True,
                   sam_channels=512),
        lr=dict(lr_type='ExpSTEP', decay_factor=0.8, decay_step=36458 // 2),
        opt=dict(opt_type='Adam', base_lr=1e-4, betas=(0.9, 0.99)),
        kwargs_train=dict(),
        kwargs_val=dict(),
    ),
)

loss = dict(
    type='EncoderLoss',
    d_loss_kwargs=dict(r1_gamma=10.0),
    e_loss_kwargs=dict(adv_lw=0.08, perceptual_lw=5e-5),
    perceptual_kwargs=dict(output_layer_idx=23,
                           pretrained_weight_path=perceptual_model_path),
)
