# StateSpacesTorch

This repository contains the implementation of the state space models for Seq2Seq or Autoregressive modelling in pure
PyTorch.
This implementation is based on the original implementation of the [State-Spaces](https://github.com/state-spaces) and
was
guided by the [S4-Torch](https://github.com/TariqAHassan/S4Torch/tree/main)
and [Mamba-Minimal](https://github.com/johnma2006/mamba-minimal).
Implementations of S4nD are adapted from the original S4 and S4nD paper and adaptations are made to the original
implementation to make it more modular and less requirement-heavy to use. The implementations of S41D, S42D, S43D, S4nD,
and recurrent Mamba achieve
close to SoTA results on image and video tasks.

The purpose of this repository is to provide a simple module to use and extend common state space models in PyTorch.

# References

### S4 Paper

```bibtex
@article{DBLP:journals/corr/abs-2111-00396,
  author = {Albert Gu and Karan Goel and Christopher R{\'{e}}},
  title = {Efficiently Modeling Long Sequences with Structured State Spaces},
  journal = {CoRR},
  volume = {abs/2111.00396},
  year = {2021},
  url = {https://arxiv.org/abs/2111.00396},
  eprinttype = {arXiv},
  eprint = {2111.00396},
  timestamp = {Fri, 05 Nov 2021 15:25:54 +0100},
  biburl = {https://dblp.org/rec/journals/corr/abs-2111-00396.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### S4nD Paper

```bibtex
@inproceedings{10.5555/3600270.3600476,
    author = {Nguyen, Eric and Goel, Karan and Gu, Albert and Downs, Gordon W. and Shah, Preey and Dao, Tri and Baccus, Stephen A. and R\'{e}, Christopher},
    title = {S4ND: modeling images and videos as multidimensional signals using state spaces},
    year = {2022},
    isbn = {9781713871088},
    publisher = {Curran Associates Inc.},
    address = {Red Hook, NY, USA},
    abstract = {Visual data such as images and videos are typically modeled as discretizations of inherently continuous, multidimensional signals. Existing continuous-signal models attempt to exploit this fact by modeling the underlying signals of visual (e.g., image) data directly. However, these models have not yet been able to achieve competitive performance on practical vision tasks such as large-scale image and video classification. Building on a recent line of work on deep state space models (SSMs), we propose S4ND, a new multidimensional SSM layer that extends the continuous-signal modeling ability of SSMs to multidimensional data including images and videos. We show that S4ND can model large-scale visual data in 1D, 2D, and 3D as continuous multidimensional signals and demonstrates strong performance by simply swapping Conv2D and self-attention layers with S4ND layers in existing state-of-the-art models. On ImageNet-1k, S4ND exceeds the performance of a Vision Transformer baseline by 1.5\% when training with a 1D sequence of patches, and matches ConvNeXt when modeling images in 2D. For videos, S4ND improves on an inflated 3D ConvNeXt in activity classification on HMDB-51 by 4\%. S4ND implicitly learns global, continuous convolutional kernels that are resolution invariant by construction, providing an inductive bias that enables generalization across multiple resolutions. By developing a simple bandlimiting modification to S4 to overcome aliasing, S4ND achieves strong zero- shot (unseen at training time) resolution performance, outperforming a baseline Conv2D by 40\% on CIFAR-10 when trained on 8 \texttimes{} 8 and tested on 32 \texttimes{} 32 images. When trained with progressive resizing, S4ND comes within ~ 1\% of a high-resolution model while training 22\% faster.},
    booktitle = {Proceedings of the 36th International Conference on Neural Information Processing Systems},
    articleno = {206},
    numpages = {16},
    location = {New Orleans, LA, USA},
    series = {NIPS '22}
}
```

### Mamba Paper

```bibtex
@inproceedings{
    gu2024mamba,
    title = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
    author = {Albert Gu and Tri Dao},
    booktitle = {First Conference on Language Modeling},
    year = {2024},
    url = {https://openreview.net/forum?id=tEYskw1VY2}
}
```
