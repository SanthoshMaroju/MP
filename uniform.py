import random
import pandas as pd

df = pd.read_csv('D:\Major_Project\MP\IoT_Modbus.csv')

sample_size = 100

df['rand_num'] = [random.random() for _ in range(len(df))]

df = df.sort_values(by='rand_num')

sample = df.head(sample_size)

