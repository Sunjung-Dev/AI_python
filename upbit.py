import time
from datetime import datetime, timedelta
import requests
import pandas as pd

MAX_BATCH_COUNT = 4000
 

class Upbit():
    def get_ohlcv(self, symbol, to_time, count = 200):
        """ request candles in a minute from upbit """
        # to_time = '2023-06-07 05:00:00'
        url = "https://api.upbit.com/v1/candles/minutes/60"
        params = {
            'market': symbol,
            'to': to_time,
            'count': count
        }
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers, params=params)
        return response.json()
    

    def fetch_as_df(self, ticker:str):
        """ fetch ohlcv from exchange, then change it to pandas.df """
        utc_now = datetime.now() - timedelta(hours=9) - timedelta(minutes=1)
        utc_now_oclock_str = utc_now.strftime("%Y-%m-%d %H:%M:00")
        print('utc_now_oclock_str')
        print(utc_now_oclock_str)

        ohlcv_list = self.get_ohlcv(ticker, utc_now_oclock_str)
        count = 0
        try:
            # ex) 216
            repeat_count = 17
            for _ in range(1, repeat_count):
                time.sleep(0.1)
                to_time = ohlcv_list[-1]['candle_date_time_kst']
                candle_datetime = datetime.strptime(to_time.replace("T", " "), "%Y-%m-%d %H:%M:%S")
                to = str(candle_datetime - timedelta(hours=9))
                current_ohlcv = self.get_ohlcv(ticker, to, 200)
                ohlcv_list += current_ohlcv
                count += 200
                if count >= 4000:
                    break
        except Exception as ex:
            print(ex)

        ohlcv_list_df = pd.DataFrame(ohlcv_list, columns =['candle_date_time_kst',
                                    'opening_price',
                                    'high_price',
                                    'low_price',
                                    'trade_price'
                                    ])
        ohlcv_list_df.columns = ['time', 'open', 'high', 'low', 'close']
        utc_finish_now = datetime.now() - timedelta(hours=9) - timedelta(minutes=1)
        utc_finish_now_oclock_str = utc_finish_now.strftime("%Y-%m-%d %H:%M:00")
        print('utc_finish_now_oclock_str')
        print(utc_finish_now_oclock_str)
        return ohlcv_list_df

upbit_b = Upbit()
print(upbit_b.fetch_as_df("KRW-ETH"))