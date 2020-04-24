import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.optim as optim
import pickle

class WaveNet(nn.Module):
    # Expecting an input of size [1, 512*200]
    # Split that input up with torch.split
    def __init__(self, num_channels = 20):
        super(WaveNet, self).__init__()
        # 200 frames to construct next one
        # Does this mean use 512 * 200 samples from audio?
        self.input_dilate1 = nn.Conv1d(1, num_channels, 5)
        self.pool = nn.MaxPool2d(2)
        self.input_dilate2 = nn.Conv1d(num_channels // 2, num_channels // 2, 5, dilation = 2)
        self.input_dilate3 = nn.Conv1d(num_channels // 4, num_channels // 4, 5, dilation = 4)
        self.dilate1 = nn.Conv1d(106, num_channels*4, 9)
        self.dilate2 = nn.Conv1d(num_channels*2, num_channels*2, 9, dilation = 2)
        self.dilate3 = nn.Conv1d(num_channels, num_channels, 9, dilation = 4)        

    def forward(self, x):
        # Can only handle bs = 1 currently
        frames = torch.split(x, 512, dim = -1) # List containing the 200 frames
        # Each frame is a tensor of shape [1, 512]
        features = torch.empty(106, 200, device = device)
        
        for i_frame, frame in enumerate(frames):
            feature = self.pool(F.relu(self.input_dilate1(frame)))
            feature = self.pool(F.relu(self.input_dilate2(feature)))
            feature = self.pool(F.relu(self.input_dilate3(feature)))
            feature = torch.flatten(feature, start_dim = 1)
            feature = feature[:, None, :]
            feature = torch.transpose(torch.flatten(feature, start_dim = 1), 0, 1)

            features[:,i_frame] = feature[:,0]
        features = features[None, :, :] # Reshape to add a first axis to represent batch.
        features = self.pool(F.relu(self.dilate1(features)))
        features = self.pool(F.relu(self.dilate2(features)))
        features = self.dilate3(features) # Maybe consider adding a sigmoid/tanh function here?
        # Can add in a pooling to make this smaller if necessary
        x = torch.flatten(features, start_dim = 1)
        # Currently, the output is of size [1, 160]
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wave = WaveNet()
print(device)
wave.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(wave.parameters(), lr=0.0003)

libri = torchaudio.datasets.LIBRISPEECH('.')
#vctk = torchaudio.datasets.VCTK('./')
dataloader = torch.utils.data.DataLoader(libri,
                                         batch_size=1,
                                         pin_memory=False,
                                         shuffle=True)
num_epochs = 5
bs = 1
for epoch in range(num_epochs):
    print(epoch)
    running_loss = 0.0
    for i,data in enumerate(dataloader):
        sound = data[0]
        if sound.shape[-1] <= 512*200 + 160:
            continue
        x, y = sound[:,:,:512*200], sound[:,:,512*200:512*200 + 160].view(1, -1)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = wave(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i/bs % 20000 == 19999:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20000))
            running_loss = 0.0
print('Finished Training')
PATH = 'wavegeneral.pth'
torch.save(wave.state_dict(), PATH)
