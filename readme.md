# Multimodal Generation of Blendshape Weights
Simple yet effective generation of blendshape weights for facial animation. This repo follows the [ARKit API](https://developer.apple.com/augmented-reality/arkit/).

Supports and creates a shared space of:
- Video input
- Audio (talking) input
- Text input

### Training
#### Local Dockerfiler
- Clone the repo.
- Build and run the provided docker file; map the repo into the container under /app and your data under /data.
- Also provide a wanbd token if desired.
#### Cloud File
- The copy variant.

### Data generation
An iOS app is provided in the arkit directory, which is tested with iOS 18.5 and an iPhone 16 Pro Max.\
Record yourself for a few minutes reading and put the collected data into the mapped data directory.

### Acceleration
We use hf's accelerate for training, a default config (default_config.yaml) is provided, will be automatically used, and can be augmented.