# GenForce Lib for Generative Modeling

An efficient PyTorch library for deep generative modeling. May the Generative Force (GenForce) be with You.

![image](./teaser.gif)

## Updates

- **Encoder Training:** We support training encoders on top of pre-trained GANs for GAN inversion.
- **Model Converters:** You can easily migrate your already started projects to this repository. Please check [here](./converters/README.md) for more details.

## Highlights

- **Distributed** training framework.
- **Fast** training speed.
- **Modular** design for prototyping new models.
- **Model zoo** containing a rich set of pretrained GAN models, with [Colab live demo](https://colab.research.google.com/github/genforce/genforce/blob/master/docs/synthesize_demo.ipynb) to play.

## Installation

1. Create a virtual environment via `conda`.

   ```shell
   conda create -n genforce python=3.7
   conda activate genforce
   ```

2. Install `cuda` and `cudnn`. (We use `CUDA 10.0` in case you would like to use `TensorFlow 1.15` for model conversion.)

   ```shell
   conda install cudatoolkit=10.0 cudnn=7.6.5
   ```

3. Install `torch` and `torchvision`.

   ```shell
   pip install torch==1.7 torchvision==0.8
   ```

4. Install requirements

   ```shell
   pip install -r requirements.txt
   ```

## Quick Demo

We provide a quick training demo, `scripts/stylegan_training_demo.py`, which allows to train StyleGAN on a toy dataset (500 animeface images with 64 x 64 resolution). Try it via

```shell
./scripts/stylegan_training_demo.sh
```

We also provide an inference demo, `synthesize.py`, which allows to synthesize images with pre-trained models. Generated images can be found at `work_dirs/synthesis_results/`. Try it via

```shell
python synthesize.py stylegan_ffhq1024
```

You can also play the demo at [Colab](https://colab.research.google.com/github/genforce/genforce/blob/master/docs/synthesize_demo.ipynb).

## Play with GANs

### Test

Pre-trained models can be found at [model zoo](MODEL_ZOO.md).

- On local machine:

  ```shell
  GPUS=8
  CONFIG=configs/stylegan_ffhq256_val.py
  WORK_DIR=work_dirs/stylegan_ffhq256_val
  CHECKPOINT=checkpoints/stylegan_ffhq256.pth
  ./scripts/dist_test.sh ${GPUS} ${CONFIG} ${WORK_DIR} ${CHECKPOINT}
  ```

- Using `slurm`:

  ```shell
  CONFIG=configs/stylegan_ffhq256_val.py
  WORK_DIR=work_dirs/stylegan_ffhq256_val
  CHECKPOINT=checkpoints/stylegan_ffhq256.pth
  GPUS=8 ./scripts/slurm_test.sh ${PARTITION} ${JOB_NAME} \
      ${CONFIG} ${WORK_DIR} ${CHECKPOINT}
  ```

### Train

All log files in the training process, such as log message, checkpoints, synthesis snapshots, etc, will be saved to the work directory.

- On local machine:

  ```shell
  GPUS=8
  CONFIG=configs/stylegan_ffhq256.py
  WORK_DIR=work_dirs/stylegan_ffhq256_train
  ./scripts/dist_train.sh ${GPUS} ${CONFIG} ${WORK_DIR} \
      [--options additional_arguments]
  ```

- Using `slurm`:

  ```shell
  CONFIG=configs/stylegan_ffhq256.py
  WORK_DIR=work_dirs/stylegan_ffhq256_train
  GPUS=8 ./scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} \
      ${CONFIG} ${WORK_DIR} \
      [--options additional_arguments]
  ```

## Play with Encoders for GAN Inversion

### Train

- On local machine:

  ```shell
  GPUS=8
  CONFIG=configs/stylegan_ffhq256_encoder_y.py
  WORK_DIR=work_dirs/stylegan_ffhq256_encoder_y
  ./scripts/dist_train.sh ${GPUS} ${CONFIG} ${WORK_DIR} \
      [--options additional_arguments]
  ```


- Using `slurm`:

  ```shell
  CONFIG=configs/stylegan_ffhq256_encoder_y.py
  WORK_DIR=work_dirs/stylegan_ffhq256_encoder_y
  GPUS=8 ./scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} \
      ${CONFIG} ${WORK_DIR} \
      [--options additional_arguments]
  ```
## Contributors

| Member                                      | Module |
| :--                                         | :--    |
|[Yujun Shen](http://shenyujun.github.io/)    | models and running controllers
|[Yinghao Xu](https://justimyhxu.github.io/)  | runner and loss functions
|[Ceyuan Yang](http://ceyuan.me/)             | data loader
|[Jiapeng Zhu](https://zhujiapeng.github.io/) | evaluation metrics
|[Bolei Zhou](http://bzhou.ie.cuhk.edu.hk/)   | cheerleader

**NOTE:** The above form only lists the person in charge for each module. We help each other a lot and develop as a **TEAM**.

*We welcome external contributors to join us for improving this library.*

## License

The project is under the [MIT License](./LICENSE).

## Acknowledgement

We thank [PGGAN](https://github.com/tkarras/progressive_growing_of_gans), [StyleGAN](https://github.com/NVlabs/stylegan), [StyleGAN2](https://github.com/NVlabs/stylegan2), [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada) for their work on high-quality image synthesis. We thank [IDInvert](https://github.com/genforce/idinvert) and [GHFeat](https://github.com/genforce/ghfeat) for their contribution to GAN inversion. We also thank [MMCV](https://github.com/open-mmlab/mmcv) for the inspiration on the design of controllers.

## BibTex

We open source this library to the community to facilitate the research of generative modeling. If you do like our work and use the codebase or models for your research, please cite our work as follows.

```bibtex
@misc{genforce2020,
  title =        {GenForce},
  author =       {Shen, Yujun and Xu, Yinghao and Yang, Ceyuan and Zhu, Jiapeng and Zhou, Bolei},
  howpublished = {\url{https://github.com/genforce/genforce}},
  year =         {2020}
}
```
