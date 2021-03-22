# WaveFrames
This is a reimplementation of [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) with a change to the input layer to process chunks of timesteps, with the hope of significant improvements to runtime. 

## Dataset
For this project, we used the [LibriSpeech](https://pytorch.org/audio/stable/_modules/torchaudio/datasets/librispeech.html) dataset, included in the PyTorch Datasets. 
The script https://github.com/timlai4/waveframes/blob/master/make_training.py was used to curate the dataset. 
We fixed a length that we would train on. Files that fell below this length were discarded. For all other files, we trimmed the first bit of the sound to this length which was then used in the training data. 

## Model
We used a simplified version of the original WaveNet architecture. We trimmed down many of the layers, but maintained the important feature of dilation. Note that the novel idea of using frames with multiple timesteps is incorporated. The simplified architecture allowed us to quickly assess the viability of this idea.
Note that our sound is encoded as a real vector, as opposed to the integer encoding in the original WaveNet implementation, and so we felt an MSE loss function was more appropriate. 

## Training
We first trained the model on the full training dataset, described in the previous section. After this initial training, we picked a speaker ID at random and fine-tuned the model to this person's sound files. 
Unfortunately, after training, it seems the model has not converged: we noticed the loss seemed to oscillate. Indeed, when attempting to generate new sound, the resulting vector seemed to be random small perturbations of 0. 
We are not sure at this time exactly why this failed and hope to return to this in the future.
