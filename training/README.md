# training/

Reserved for the model training phase. Training runs on **Kaggle** (PyTorch), so this
directory holds the code and notebooks that are pushed there rather than checkpoints.

Planned contents:

- dataset loader over `dataset/telemetry/*.jsonl` and `dataset/textures/manifest.json`
- feature extraction (camera pose/velocity history -> texture demand per mip level)
- model definition and the Kaggle training notebook/script
- export step that writes the trained model into `../exported_model/`

Nothing here yet - the neural network is not implemented in this phase.
