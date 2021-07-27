# Data Preparation

## Data Format

Currently, our dataloader is able to load data from

- a directory that is full of images (support using [`turbojpeg`](https://pypi.org/project/PyTurboJPEG/) to speed up decoding images.)
- a `lmdb` file
- an image list
- a compressed file (i.e., `zip` package)

by modifying `data_format` in the configuration.

**NOTE:** For some computing clusters whose I/O speed may be slow, we recommend the `zip` format for two reasons. First, `zip` file is easy to create. Second, this can load a large file at one time instead of loading small files repeatedly.

## Data Sampling

Considering that most generative models are trained in the unit of iterations instead of epochs, we change the default data loader to an *iter-based* one. Besides, the original distributed data sampler is also modified to make the shuffling correspond to iteration instead of epoch.

**NOTE:** In order to reduce the data re-loading cost between epochs, we manually extend the length of sampled indices to make it much more efficient.

## Data Augmentation

To better align with the original implementation of PGGAN and StyleGAN (i.e., models that require progressive training), we support progressive resize in `transforms.py`, which downsamples images with the maximum resize factor of 2 at each time.
