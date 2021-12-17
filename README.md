# Synthetic_Scans
A simple autoencoder for generating synthetic CT scans of skulls.

This work currently only contains a working framework for 2d slices and lacks practically useful performance. 

![plot](./plots/GUI.png)

It is optimized for the CERMEP dataset.

In order to improve the model, consider:
- Acquire more data
- Augmenting the data
  - Shifting
  - Rotating
- Increasing complexity
  - larger layers
  - more layers
- Replace the autoencoder by a Generative Adversarial Network
