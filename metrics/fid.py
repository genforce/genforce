# python3.7
"""Contains the functions to compute Frechet Inception Distance (FID).

FID metric is introduced in paper

GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash
Equilibrium. Heusel et al. NeurIPS 2017.

See details at https://arxiv.org/pdf/1706.08500.pdf
"""

import numpy as np
import scipy.linalg

__all__ = ['extract_feature', 'compute_fid']


def extract_feature(inception_model, images):
    """Extracts feature from input images with given model.

    NOTE: The input images are assumed to be with pixel range [-1, 1].

    Args:
        inception_model: The model used to extract features.
        images: The input image tensor to extract features from.

    Returns:
        A `numpy.ndarray`, containing the extracted features.
    """
    features = inception_model(images, output_logits=False)
    features = features.detach().cpu().numpy()
    assert features.ndim == 2 and features.shape[1] == 2048
    return features


def compute_fid(fake_features, real_features):
    """Computes FID based on the features extracted from fake and real data.

    Given the mean and covariance (m_f, C_f) of fake data and (m_r, C_r) of real
    data, the FID metric can be computed by

    d^2 = ||m_f - m_r||_2^2 + Tr(C_f + C_r - 2(C_f C_r)^0.5)

    Args:
        fake_features: The features extracted from fake data.
        real_features: The features extracted from real data.

    Returns:
        A real number, suggesting the FID value.
    """

    m_f = np.mean(fake_features, axis=0)
    C_f = np.cov(fake_features, rowvar=False)
    m_r = np.mean(real_features, axis=0)
    C_r = np.cov(real_features, rowvar=False)

    fid = np.sum((m_f - m_r) ** 2) + np.trace(
        C_f + C_r - 2 * scipy.linalg.sqrtm(np.dot(C_f, C_r)))
    return np.real(fid)
