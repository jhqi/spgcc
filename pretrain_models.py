import torch.nn as nn
import torch


class ConvBNRelu3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=24, kernel_size=(51, 3, 3), padding=0, stride=1):
        super(ConvBNRelu3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.conv = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn = nn.BatchNorm3d(num_features=self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBNRelu2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=24, kernel_size=(51, 3, 3), stride=1, padding=0):
        super(ConvBNRelu2D, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Enc_VAE(nn.Module):
    def __init__(self, channel, output_dim, windowSize):
        super(Enc_VAE, self).__init__()
        self.channel = channel
        self.output_dim = output_dim
        self.windowSize = windowSize
        self.conv1 = ConvBNRelu3D(
            in_channels=1, out_channels=8, kernel_size=(7, 3, 3), stride=1, padding=0)
        self.conv2 = ConvBNRelu3D(
            in_channels=8, out_channels=16, kernel_size=(5, 3, 3), stride=1, padding=0)
        self.conv3 = ConvBNRelu3D(
            in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.conv4 = ConvBNRelu2D(in_channels=32 * (self.channel - 12),
                                  out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.projector = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=False))
        self.mu = nn.Linear(512, output_dim)
        self.log_sigma = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape([x.shape[0], -1, x.shape[3], x.shape[4]])
        x = self.conv4(x)
        map = self.pool(x)
        h = map.reshape([map.shape[0], -1])
        x = self.projector(h)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        sigma = torch.exp(log_sigma)
        return h, mu, sigma


class Dec_VAE(nn.Module):  # input (-1,128)
    def __init__(self, channel=30, windowSize=25, input_dim=64):
        super(Dec_VAE, self).__init__()
        self.channel = channel
        self.windowSize = windowSize
        self.fc1 = nn.Linear(in_features=input_dim, out_features=256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=64 *
                             (self.windowSize - 8) * (self.windowSize - 8))
        self.relu2 = nn.ReLU()
        # reshape to (-1,64,17,17) and then deconv.
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32 * (self.channel - 12), kernel_size=(3, 3), stride=(1, 1),
                                          padding=0)  # H_{out}=(H_{in}-1)stride-2padding+kernel_size+output_padding
        self.bn3 = nn.BatchNorm2d(num_features=32 * (self.channel - 12))
        self.relu3 = nn.ReLU()
        # reshape to (-1,32,18,19,19)
        self.deconv4 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=0)
        self.bn4 = nn.BatchNorm3d(num_features=16)
        self.relu4 = nn.ReLU()
        # (-1,16,20,21,21)
        self.deconv5 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(
            5, 3, 3), stride=(1, 1, 1), padding=0)
        self.bn5 = nn.BatchNorm3d(num_features=8)
        self.relu5 = nn.ReLU()
        # [-1, 8, 24, 23, 23]
        self.deconv6 = nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=(
            7, 3, 3), stride=(1, 1, 1), padding=0)
        self.bn6 = nn.BatchNorm3d(num_features=1)
        self.relu6 = nn.ReLU()
        # [-1,1,30,25,25]
        self._initialize_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = x.view(-1, 64, self.windowSize - 8, self.windowSize - 8)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = x.view(-1, 32, self.channel - 12,
                   self.windowSize - 6, self.windowSize - 6)
        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.deconv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.deconv6(x)
        x = self.bn6(x)
        # x = self.relu6(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class CNN_Enc(nn.Module):
    def __init__(self, channel, output_dim, windowSize):
        super(CNN_Enc, self).__init__()
        self.channel = channel
        self.output_dim = output_dim
        self.windowSize = windowSize
        self.conv1 = ConvBNRelu3D(
            in_channels=1, out_channels=8, kernel_size=(7, 3, 3), stride=1, padding=0)
        self.conv2 = ConvBNRelu3D(
            in_channels=8, out_channels=16, kernel_size=(5, 3, 3), stride=1, padding=0)
        self.conv3 = ConvBNRelu3D(
            in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.conv4 = ConvBNRelu2D(in_channels=32 * (self.channel - 12),
                                  out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape([x.shape[0], -1, x.shape[3], x.shape[4]])
        x = self.conv4(x)
        x = self.pool(x)
        x = x.reshape([x.shape[0], -1])
        return x