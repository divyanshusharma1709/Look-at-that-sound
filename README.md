# Look at that sound!
An attempt at using a deep generative diffusion-based model to generate images given audio as input. The model consists of an audio embedder based on Wav2vec 2.0 whose output is used to condition a frozen StableDiffusion latent diffusion image generation model. The audio embedder is trained on subsets of the VGGSound dataset.

## Sample output: 
![Sample output](generated.png)

## Project files: 
* `audio_embedder.py`: contains the audio embedder code that produces the conditioning state for the diffusion model
* `data_utils.py`: Dataset and data loading utilities
* `extract_best_frames.py`: A script that extracts a representative frame from a video by choosing a frame (from a
  randomly selected subset) that best matches the text label provided with the dataset
* `diffusion_model.py`: Combines the audio-embedder with the frozen diffusion model
* `extract_data.sh`: A utility shell script for dataset curation
* `inference.py`: A script for generating images from audio samples 
* `train_model.py`: Training script

## Future work:
Classifier guidance (not used in the final model):
* `classifier_data_utils.py`: contains data loading utilities for classifier training
* `train_classifier.py`: contains training code for a classifier to try to use with classifier guidance

