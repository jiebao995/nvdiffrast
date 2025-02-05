## Nvdiffrast &ndash; Modular Primitives for High-Performance Differentiable Rendering

### Description
This repo is forked from https://github.com/NVlabs/nvdiffrast. Used to replicate the same experimente and for future needs.

Summary of added files:

* [samples/](./samples)
  * [data/](./samples/data)
    * [suzanne.obj](./samples/data/suzanne.obj)
  * [torch/](./samples/torch)
    * [optimize_suzanne.py](./samples/torch/optimize_suzanne.py)
    * [test_viz_suzanne.py](./samples/torch/test_viz_suzanne.py)

The build setup is the same as the original work. Please refer to the official documentation.

To run the files in torch:
```
python optimize_suzanne.py
```

## Citation

```
@article{Laine2020diffrast,
  title   = {Modular Primitives for High-Performance Differentiable Rendering},
  author  = {Samuli Laine and Janne Hellsten and Tero Karras and Yeongho Seol and Jaakko Lehtinen and Timo Aila},
  journal = {ACM Transactions on Graphics},
  year    = {2020},
  volume  = {39},
  number  = {6}
}
```
