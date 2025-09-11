import datetime

C1 = 32
C2 = 64
C3 = 128
C4 = 256
C5 = 512
C6 = 1024
C7 = 2048
C8 = 4096



#Lidar Info
_LIDAR_SIZE = [5120, 5120]
LIDAR_IN_CHANNELS = L1 = 1
LIDAR_OUT_CHANNELS = C1

#Sentinel Info
_SEN_SIZE = [256, 256]
SEN_IN_CHANNELS = 11
SEN_PERIODS = 21

#Number of channels after sentinel data is compressed
S1 = SEN_IN_CHANNELS * SEN_PERIODS


#Weather Info
pre_season_start_date = "10-01"
time_series_start_date = "03-01"
prediction_date = "09-01"

IN_SEASON_DAYS = (datetime.datetime.strptime(prediction_date, "%m-%d") - datetime.datetime.strptime(time_series_start_date, "%m-%d")).days
PRE_SEASON_DAYS = (
	datetime.datetime.strptime(time_series_start_date, "%m-%d").replace(year=2001) -
	datetime.datetime.strptime(pre_season_start_date, "%m-%d").replace(year=2000)
).days

WEATHER_IN_CHANNELS = 2

#This is the "rolling temporal window" the model uses to process weather data
IN_SEASON_KERNEL_SIZE = 3
PRE_SEASON_KERNEL_SIZE = 14

#Number of channels after weather data is compressed
W1 = 10
W2 = 1

BOTTLENECK_SIZE = [8, 8]
BATCH_SIZE = 1
ACCUMULATION_STEPS = 8