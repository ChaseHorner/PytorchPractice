import datetime

C1 = 32
C2 = 64
C3 = 128
C4 = 256
C5 = 512

S1 = 10
W1 = 10
W2 = 1

#Lidar Info
_LIDAR_SIZE = [5120, 5120]
_LIDAR_IN_CHANNELS = 1
LIDAR_OUT_CHANNELS = C1

#Sentinel Info
_SEN_SIZE = [256, 256]
SEN_IN_CHANNELS = 10
SEN_OUT_CHANNELS = S1

#Weather Info
time_series_start_date = "03-01"
prediction_date = "10-1"

WEATHER_IN_CHANNELS = 2

#This is the "rolling temporal window" the model uses to process weather data
IN_SEASON_KERNEL_SIZE = 3
OUT_SEASON_KERNEL_SIZE = 14

IN_SEASON_OUT_CHANNELS = W1


BOTTLENECK_SIZE = [64, 64]
BATCH_SIZE = 16