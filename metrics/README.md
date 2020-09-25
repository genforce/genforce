# Evaluation Metrics

Frechet Inception Distance (FID) is commonly used to evaluate generative model. It employs an [Inception Model](https://arxiv.org/abs/1512.00567) (pretrained on ImageNet) to extract features from both real and synthesized images.

## Inception Model

For [PGGAN](https://github.com/tkarras/progressive_growing_of_gans), [StyleGAN](https://github.com/NVlabs/stylegan), etc, they use inception model from the [TensorFlow Models](https://github.com/tensorflow/models) repository, whose implementation is slightly different from that of `torchvision`. Hence, to make the evaluation metric comparable between different training frameworks (i.e., PyTorch and TensorFlow), we modify `torchvision/models/inception.py` as `inception.py`. The ported pre-trained weight is borrowed from [this repo](https://github.com/mseitzer/pytorch-fid).

**NOTE:** We also support using the model from `torchvision` to compute the FID. However, please be aware that the FID value from `torchvision` is usually ~1.5 smaller than that from the TensorFlow model.

Please use the following code to choose which model to use.

```python
from metrics.inception import build_inception_model

inception_model_tf = build_inception_model(align_tf=True)
inception_model_pth = build_inception_model(align_tf=False)
```
