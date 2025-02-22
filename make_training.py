import torch
import torchaudio
import pickle

libri = torchaudio.datasets.LIBRISPEECH('.')
#vctk = torchaudio.datasets.VCTK('./')
dataloader = torch.utils.data.DataLoader(libri,
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=4)

waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = libri[0]

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))
print(waveform.dtype)
print(speaker_id)
test = [x[0] for x in libri if x[3] == 1624]
print(len(test))
test = [x for x in test if x.shape[-1] > 256*50 + 8000*16 + 184]
with open('train1','wb') as f:
    pickle.dump(test, f)
print(len(test))
