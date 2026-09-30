"""VirtualPaint-Tox: label-free virtual Cell Painting with pixel-wise uncertainty.

Sub-modules
-----------
manifest    build a field-level manifest (paths, well metadata, split) from Cell Painting Gallery load_data.csv + platemap
preprocess  stack the six TIFFs of a field into one uint16 .npy and compute normalisation statistics
data        PyTorch datasets (random crops for training, full fields for evaluation)
model       U-Net with an optional heteroscedastic (Laplace) uncertainty head
train       training loop (MPS / CUDA / CPU)
predict     inference with test-time augmentation, writes virtual stains and uncertainty maps
metrics     pixel-level metrics and uncertainty calibration
"""

CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]  # target order everywhere in the project
INPUT = "Brightfield"
