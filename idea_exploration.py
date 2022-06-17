import pandas as pd
import numpy as np

# start_month = 7
# end_month = 7
#
# start_year = 2010
# end_year = 2011
#
# df_collector = []
# for year in range(start_year, end_year):
#     date_range = pd.date_range(start=f"{start_month}/1/{year}",
#                                end=f"{end_month + 1}/1/{year}",
#                                freq="D")
#     for date in date_range[:-1]:
#
#         url = f"https://mesowest.utah.edu/cgi-bin/droman/download_api2_handler.cgi?output=csv&product=&stn=KMSO&unit=1&daycalendar=1&hours=1&day1=" \
#               f"{date.day}&month1={date.month:02}&year1={date.year}&time=LOCAL&hour1={0}&var_0=air_temp&var_2=relative_humidity"
#         df = pd.read_csv(url, header=6, skiprows=[7])
#         df.index = pd.to_datetime(df["Date_Time"])
#         df = df.resample("1H").mean()
#         df_collector.append(df)
#
# df = pd.concat(df_collector)
# df.interpolate(inplace=True)
# df.to_csv("./weather_data/sample_data.csv")



data = pd.read_csv("./weather_data/sample_data.csv", index_col=0)
data.index = pd.to_datetime(data.index)


sample_range = pd.date_range(data.index[0], data.index[-72], freq="D")

for i in range(10):
    end = len(sample_range) - 3
    index = np.random.randint(0, len(sample_range) - 3)
    sample = data.loc[sample_range[index]: sample_range[index + 3]]
    temp = sample["air_temp_set_1"].array
    humidity = sample["relative_humidity_set_1"].array

    print("check")



print()
