import torch
import torch.nn as nn
import configs

class WeatherCompression(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(WeatherCompression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1) # (b, out_channels, ~ts_len) -> (b, out_channels, ts_len/2)
        )

    def forward(self, x):
        h = self.conv(x).squeeze(-1) #(b, out_channels, 1) -> (b, out_channels)
        return h

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale_size = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(scale_size),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels = 0, scale_size = 2):
        super(Decoder, self).__init__()

        self.conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scale_size),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.conv(x)
        x = torch.concat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class FinalOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(
            self, 
            lidar_channels = configs.LIDAR_IN_CHANNELS, 
            sentinel_channels = configs.SEN_IN_CHANNELS, 
            weather_channels = configs.WEATHER_IN_CHANNELS, 
            output_channels=1
            ):

        super(Unet, self).__init__()

        self.lidar_channels = lidar_channels
        self.sentinel_channels = sentinel_channels
        self.weather_channels = weather_channels
        self.output_channels = output_channels

        self.in_weather_in_season = WeatherCompression(weather_channels, configs.W1, kernel_size=configs.IN_SEASON_KERNEL_SIZE)
        self.in_weather_pre_season = WeatherCompression(weather_channels, configs.W2, kernel_size=configs.PRE_SEASON_KERNEL_SIZE)

        self.enc_1 = Encoder(lidar_channels, configs.C1)
        self.enc_2 = Encoder(configs.C1, configs.C2)
        self.enc_3 = Encoder(configs.C2, configs.C3, scale_size=5)
        self.enc_4 = Encoder(configs.C3 + configs.S1, configs.C4)
        self.enc_5 = Encoder(configs.C4, configs.C5)
        self.enc_6 = Encoder(configs.C5, configs.C6)
        self.enc_7 = Encoder(configs.C6, configs.C7)
        self.enc_8 = Encoder(configs.C7, configs.C8)

        self.dec_8 = Decoder(configs.C8 + configs.W1 + configs.W2, configs.C7, skip_channels=configs.C7)
        self.dec_7 = Decoder(configs.C7, configs.C6, skip_channels=configs.C6)
        self.dec_6 = Decoder(configs.C6, configs.C5, skip_channels=configs.C5)
        self.dec_5 = Decoder(configs.C5, configs.C4, skip_channels=configs.C4)
        self.dec_4 = Decoder(configs.C4, configs.C3 + configs.S1, skip_channels=configs.C3 + configs.S1)

        self.final_output = FinalOutput(configs.C3, output_channels)

    def forward(self, lidar_data, sentinel_data, weather_in_season_data, weather_out_season_data):
        i1 = x1 = lidar_data
        i2 = sentinel_data
        i3 = self.in_weather_in_season(weather_in_season_data)
        i4 = self.in_weather_pre_season(weather_out_season_data)



        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)

        x4 = torch.cat([x4, i2], dim=1)
        x5 = self.enc_4(x4)
        x6 = self.enc_5(x5)
        x7 = self.enc_6(x6)
        x8 = self.enc_7(x7)
        x9 = self.enc_8(x8)

        i3 = i3.unsqueeze(-1).unsqueeze(-1)
        i4 = i4.unsqueeze(-1).unsqueeze(-1)
        i3 = i3.expand(-1, -1, x9.shape[2], x9.shape[3])
        i4 = i4.expand(-1, -1, x9.shape[2], x9.shape[3])
        x9 = torch.cat([x9, i3, i4], dim=1)

        x = self.dec_8(x9, x8)
        x = self.dec_7(x, x7)
        x = self.dec_6(x, x6)
        x = self.dec_5(x, x5)
        x = self.dec_4(x, x4)

        x = self.final_output(x)

        return x