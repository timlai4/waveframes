import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchaudio
import torch.optim as optim
import pickle

class WaveNet(nn.Module):
    # Expecting an input of size [bs, 1, 512*100]
    # Split that input up with torch.split
    def __init__(self, num_channels = 20):
        super(WaveNet, self).__init__()
        self.input_dilate1 = nn.Conv1d(1, num_channels, 5)
        self.pool = nn.MaxPool2d(2)
        self.input_dilate2 = nn.Conv1d(num_channels // 2, num_channels // 2, 5, dilation = 2)
        self.input_dilate3 = nn.Conv1d(num_channels // 4, num_channels // 4, 5, dilation = 4)
        self.dilate1 = nn.Conv1d(106, num_channels*8, 9)
        self.dilate2 = nn.Conv1d(num_channels*4, num_channels*4, 9, dilation = 1)
        self.dilate3 = nn.Conv1d(num_channels*2, num_channels*2, 9, dilation = 2)        

    def forward(self, x):
        frames = torch.split(x, 512, dim = -1) # List containing the 200 frames
        # Each frame is a tensor of shape [1, 512]
        for i_frame, frame in enumerate(frames):
            feature = self.pool(F.relu(self.input_dilate1(frame)))
            feature = self.pool(F.relu(self.input_dilate2(feature)))
            feature = self.pool(F.relu(self.input_dilate3(feature)))
            feature = torch.flatten(feature, start_dim = 1)
            feature = feature[:, None, :]
            if i_frame == 0: 
                features = feature
            else: 
                features = torch.cat((features, feature), 1)
        features = torch.transpose(features, 1, 2)
        features = self.pool(F.relu(self.dilate1(features)))
        features = self.pool(F.relu(self.dilate2(features)))
        features = torch.tanh(self.dilate3(features)) 
        # Can add in a pooling to make this smaller if necessary
        x = torch.flatten(features, start_dim = 1)
        # Currently, the output is of size [1, 120]
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wave = WaveNet()
PATH = 'wavegeneral.pth'
wave.load_state_dict(torch.load(PATH))
wave.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(wave.parameters(), lr=0.0003)

with open('trainId1624', 'rb') as f:
    ds = pickle.load(f)
num_epochs = 40
bs = 16
for epoch in range(num_epochs):
    print(epoch)
    running_loss = 0.0
    for i in range(len(ds)):
        sound = ds[i].to(device)
        x, y = sound[:,:512*100], sound[:,512*100:512*100 + 120]
        x, y = x[None, :, :], y[None, :, :]
        for j in range(1, bs):
            shift_x = sound[:, 8000*j:512*100 + 8000*j]
            shift_y = sound[:,512*100 + 8000*j:512*100 + 120 + 8000*j]
            shift_x, shift_y = shift_x[None, :, :], shift_y[None, :, :]
            x = torch.cat((x, shift_x), 0)
            y = torch.cat((y, shift_y), 0)
        x.to(device)
        y.to(device)
        optimizer.zero_grad()
        output = wave(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i/bs % 20 == 19:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
print('Finished Training')
PATH = 'wave1624.pth'
torch.save(wave.state_dict(), PATH)
