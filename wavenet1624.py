import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

class InputFrames(nn.Module):
    # Expecting an input of size [bs*50, 1, 256]
    # Split that input up with torch.split
    def __init__(self):
        super(InputFrames, self).__init__()
        self.input_dilate1 = nn.Conv1d(1, 20, 5, stride = 2)
        self.input_dilate2 = nn.Conv1d(20, 20, 5, dilation = 2, stride = 2)
        self.input_dilate3 = nn.Conv1d(20, 20, 5, dilation = 4, stride = 2)
    def forward(self, x):
        x = F.relu(self.input_dilate1(x))
        x = F.relu(self.input_dilate2(x))
        x = F.relu(self.input_dilate3(x))
        return x # shape [bs*50, 20, 22]
# Reshape to [bs, 20*22, 50]
class WaveNet(nn.Module):
    def __init__(self, feature_dim = 20*22):
        super(WaveNet, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, (feature_dim - 32) // 2, 8, stride = 2)
        self.conv2 = nn.Conv1d((feature_dim - 32) // 2, (feature_dim - 128) // 4, 8)
        self.conv3 = nn.Conv1d((feature_dim - 128) // 4, (feature_dim - 256) // 8, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.flatten(x, start_dim = 1)
        # Currently, the output is of size [bs, 23, 8]
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wave = WaveNet()
frames = InputFrames()
PATH = 'wavegeneral.pth'
PATH2 = 'framegeneral.pth'
frames.load_state_dict(torch.load(PATH2, map_location = device))
wave.load_state_dict(torch.load(PATH, map_location = device))
wave.to(device)
frames.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(wave.parameters(), lr=0.0003)

with open('trainId1624', 'rb') as f:
    ds = pickle.load(f)
    
num_epochs = 100
bs = 16
pred_size = 184
for epoch in range(num_epochs):
    print(epoch)
    running_loss = 0.0
    for i in range(len(ds)):
        sound = ds[i]
        x, y = sound[:,:256*50], sound[:,256*50:256*50 + pred_size]
        x = x[None, :, :]
        for j in range(1, bs):
            shift_x = sound[:, 8000*j:256*50 + 8000*j]
            shift_y = sound[:,256*50 + 8000*j:256*50 + pred_size + 8000*j]
            shift_x = shift_x[None, :, :]
            x = torch.cat((x, shift_x), 0)
            y = torch.cat((y, shift_y), 0)
        x = torch.reshape(x, (bs*50, 1, 256))
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        x = frames(x)
        x = torch.reshape(x, (bs, x.shape[1]*x.shape[2], 50))
        output = wave(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
print('Finished Training')
PATH = 'wave1624.pth'
torch.save(wave.state_dict(), PATH)
PATH2 = 'frame1624.pth'
torch.save(frames.state_dict(), PATH2)
