# Training data

The committed checkpoint was fine-tuned on a stratified subset of PlantVillage:

- Hugging Face dataset: [geraldmc/plantvillage-tiny](https://huggingface.co/datasets/geraldmc/plantvillage-tiny) revision `v0.1.0`
- About 50 images per class, 38 classes (crop + condition), with the publisher's train/test column
- License: **CC0 1.0** (public-domain dedication), inherited from the upstream PlantVillage release

`scripts/download_data.py` downloads that subset. Images are not committed; they land in `data/plantvillage/`, which is gitignored. One JPEG used by the API smoke test is stored at `tests/fixtures/sample.jpg` and is covered by the same CC0 dedication.

## Upstream sources

PlantVillage is an open collection of healthy and diseased leaf photographs:

- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*.
- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419.

The tiny release used here is a convenience subsample of `geraldmc/plantvillage-full`, which repackages the color images from the PlantVillage repository. It is a debug-scale subset: per-class test counts are small, so the accuracy in `artifacts/metrics.json` is a demo measurement, not a benchmark of the full dataset.

## Other datasets

`scripts/train.py` reads a generic ImageFolder layout (`train/<class>` and `test/<class>`). A PlantDoc export can be trained the same way if you arrange it that way yourself. PlantDoc is a separate collection with its own terms; this repository does not redistribute it. Check the dataset publisher's license before downloading or shipping a checkpoint trained on it.
