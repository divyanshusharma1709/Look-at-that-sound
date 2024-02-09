# Computer Vision Fall 2023 Course Project: Audio2Image
In this project, we will build an architecture that can turn various input modalities into
images.

## Navigation
Classifier guidance (not used in the final model):
* `classifier_data_utils.py`: contains data loading utilities for classifier training
* `train_classifier.py`: contains training code for a classifier to try to use with classifier guidance
Audio2Image:
* `audio_embedder.py`: contains the audio embedder code that produces the conditioning state for the diffusion model
* `data_utils.py`: Dataset and data loading utilities
* `extract_best_frames.py`: A script that extracts a representative frame from a video by choosing a frame (from a
  randomly selected subset) that best matches the text label provided with the dataset
* `diffusion_model.py`: Combines the audio-embedder with the frozen diffusion model into audio2image
* `extract_data.sh`: A utility shell script for dataset curation
* `inference.py`: A script for generating images from audio samples using Audio2Image
* `train_model.py`: A training script for audio2image
