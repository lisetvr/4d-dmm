# 4d-dmm: 4D deep motion model
This source code contains the implementation of the motion model presented in the article: "Probabilistic 4D predictive model from in-room surrogates using conditional generative networks for image-guided radiotherapy". 

This probabilistic model aims at addressing the problem of volumetric estimation with scalable predictive horizon from image-based surrogates, thus enabling out-of-plane tracking of targets. This problem is formulated as a conditional learning task, where the predictive variables are the 2D surrogate images and a pre-operative static 3D volume. The model employs a conditional variational autoencoder as backbone to establish correspondences between respiratory phases and dense motion fields. It learns a distribution of realistic motion fields over a population dataset. Simultaneously, a *seq-2-seq* inspired temporal mechanism acts over the surrogate images yielding extrapolated-in-time representations. Thus, it can generate multiple future volumes in one shot. The phase-specific motion distributions are associated with the predicted temporal representations, allowing the recovery of dense organ deformation in multiple times. In the test stage, it only requires a static 3D volume and cine 2D slices to predict future deformations.

<img src="https://github.com/lisetvr/4d-dmm/blob/main/model_figure.png" width="550" height="400">

## Cite
If you find this code useful for your research, please cite our [paper](https://doi.org/10.1016/j.media.2021.102250):
```
@article{romaguera2021probabilistic,
  title={Probabilistic 4D predictive model from in-room surrogates using conditional generative networks for image-guided radiotherapy},
  author={Romaguera, Liset V{\'a}zquez and Mezheritsky, Tal and Mansour, Rihab and Carrier, Jean-Fran{\c{c}}ois and Kadoury, Samuel},
  journal={Medical image analysis},
  volume={74},
  pages={102250},
  year={2021},
  publisher={Elsevier}
}

```
