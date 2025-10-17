import numpy as np
import download_stock
import wavelet_func



stock_input = input('ENTER stock TICKER and press ENTER >> ')
stock = stock_input.upper()

df = download_stock.download(stock)
print(df)


wavelet_func.wavelet_full_analysis(df, stock, wavelet='cmor1.5-1.0', scales=np.arange(1, 128), sampling_rate=1.0)