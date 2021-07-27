# Model Converters

## Introduction

Besides training, we also support converting pre-trained model weights from officially released models and using them for inference. So, if you have already trained some models with the officially open-sourced codebase, don't worry, we have already made sure that they well match our codebase! We now support models trained with following repositories:

- [PGGAN](https://github.com/tkarras/progressive_growing_of_gans) (TensorFlow)
- [StyleGAN](https://github.com/NVlabs/stylegan) (TensorFlow)
- [StyleGAN2](https://github.com/NVlabs/stylegan2) (TensorFlow)
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada) (TensorFlow)
- [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) (PyTorch)

**NOTE:** Our codebase is completely built on PyTorch. But, if you want to convert the official TensorFlow model, you need to setup the TensorFlow environment. This can be easily done with `pip install tensorflow-gpu==1.15`.

We also mirror the officially open-sourced codes in this folder, which are relied on by `pickle.load()`. Specifically, we have

- `pggan_official/`: [PGGAN](https://github.com/tkarras/progressive_growing_of_gans)
- `stylegan_official/`: [StyleGAN](https://github.com/NVlabs/stylegan)
- `stylegan2_official/`: [StyleGAN2](https://github.com/NVlabs/stylegan2)
- `stylegan2ada_tf_official/`: [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada)
- `stylegan2ada_pth_official/`: [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

**NOTE:** These codes will ONLY be used for model conversion. After that, all codes within this folder will not be used anymore.

## Usage

The script to convert a model is provided as `../convert_model.py`. For example, to convert a pre-trained StyleGAN2 model (officially TensorFlow version), just run

```shell
cd ..
python convert_model.py stylegan2 \
       --source_model_path ${SOURCE_MODEL_PATH} \
       --test_num 10 \
       --save_test_image
```

The above command will execute the conversion and then test the conversion precision.
