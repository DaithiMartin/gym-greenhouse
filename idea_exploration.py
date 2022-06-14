import pandas as pd

start_month = 3
end_month = 3

start_year = 2010
end_year = 2012

df_collector = []
for year in range(start_year, end_year):
    date_range = pd.date_range(start=f"{start_month}/1/{year}",
                               end=f"{end_month + 1}/1/{year}",
                               freq="D")
    for date in date_range[:-1]:

        url = f"https://mesowest.utah.edu/cgi-bin/droman/download_api2_handler.cgi?output=csv&product=&stn=KMSO&unit=0&daycalendar=1&hours=1&day1=" \
              f"{date.day}&month1={date.month:02}&year1={date.year}&time=LOCAL&hour1={0}&var_0=air_temp&var_2=relative_humidity"
        df = pd.read_csv(url, header=6, skiprows=[7])
        df.index = pd.to_datetime(df["Date_Time"])
        df = df.resample("1H").mean()
        df_collector.append(df)

        print("check")

df = pd.concat(df_collector)
test = df.groupby([df.index.dt.day]).sum()
print()
