"""
Compare HDF5 files with different liquidity filters.
"""

import h5py
import os

files = {
    'ALL': '../data/prices_all.h5',
    'LIQUID': '../data/prices_liquid.h5',
    'HIGHLY_LIQUID': '../data/prices_highly_liquid.h5'
}

print()
print('=' * 90)
print('HDF5 FILE COMPARISON - LIQUIDITY FILTERING RESULTS')
print('=' * 90)
print()

# Header
header = f"{'Metric':<30} {'ALL':>18} {'LIQUID':>18} {'HIGHLY_LIQUID':>18}"
print(header)
print('-' * 90)

data = {}
for name, path in files.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        continue
    with h5py.File(path, 'r') as f:
        data[name] = {
            'n_tickers': f.attrs['n_tickers'],
            'n_rows': f.attrs['n_rows'],
            'total_volume': f.attrs.get('total_volume', 0),
            'total_value': f.attrs.get('total_value', 0),
            'file_size_mb': os.path.getsize(path) / (1024 * 1024),
            'tickers': [t.decode('utf-8') if isinstance(t, bytes) else t for t in f['ticker_names'][:]]
        }

# Print comparison
row1 = f"{'Number of Tickers':<30} {data['ALL']['n_tickers']:>18,} {data['LIQUID']['n_tickers']:>18,} {data['HIGHLY_LIQUID']['n_tickers']:>18,}"
row2 = f"{'Number of Data Points':<30} {data['ALL']['n_rows']:>18,} {data['LIQUID']['n_rows']:>18,} {data['HIGHLY_LIQUID']['n_rows']:>18,}"
row3 = f"{'Total Volume':<30} {data['ALL']['total_volume']:>18,.0f} {data['LIQUID']['total_volume']:>18,.0f} {data['HIGHLY_LIQUID']['total_volume']:>18,.0f}"
row4 = f"{'Total Value ($)':<30} {data['ALL']['total_value']:>18,.0f} {data['LIQUID']['total_value']:>18,.0f} {data['HIGHLY_LIQUID']['total_value']:>18,.0f}"
row5 = f"{'File Size (MB)':<30} {data['ALL']['file_size_mb']:>18,.2f} {data['LIQUID']['file_size_mb']:>18,.2f} {data['HIGHLY_LIQUID']['file_size_mb']:>18,.2f}"

print(row1)
print(row2)
print(row3)
print(row4)
print(row5)
print()

# Percentages relative to ALL
print('-' * 90)
print('PERCENTAGES RELATIVE TO ALL:')
print('-' * 90)

pct_tickers_liquid = data['LIQUID']['n_tickers']/data['ALL']['n_tickers']*100
pct_tickers_highly = data['HIGHLY_LIQUID']['n_tickers']/data['ALL']['n_tickers']*100
pct_rows_liquid = data['LIQUID']['n_rows']/data['ALL']['n_rows']*100
pct_rows_highly = data['HIGHLY_LIQUID']['n_rows']/data['ALL']['n_rows']*100
pct_vol_liquid = data['LIQUID']['total_volume']/data['ALL']['total_volume']*100
pct_vol_highly = data['HIGHLY_LIQUID']['total_volume']/data['ALL']['total_volume']*100
pct_size_liquid = data['LIQUID']['file_size_mb']/data['ALL']['file_size_mb']*100
pct_size_highly = data['HIGHLY_LIQUID']['file_size_mb']/data['ALL']['file_size_mb']*100

print(f"{'Tickers Retained (%)':<30} {'100.00%':>18} {pct_tickers_liquid:>17.2f}% {pct_tickers_highly:>17.2f}%")
print(f"{'Data Points Retained (%)':<30} {'100.00%':>18} {pct_rows_liquid:>17.2f}% {pct_rows_highly:>17.2f}%")
print(f"{'Volume Retained (%)':<30} {'100.00%':>18} {pct_vol_liquid:>17.2f}% {pct_vol_highly:>17.2f}%")
print(f"{'File Size (%)':<30} {'100.00%':>18} {pct_size_liquid:>17.2f}% {pct_size_highly:>17.2f}%")
print()

# Data quality metrics
print('-' * 90)
print('DATA QUALITY METRICS (avg per ticker):')
print('-' * 90)
avg_rows_all = data['ALL']['n_rows'] / data['ALL']['n_tickers']
avg_rows_liquid = data['LIQUID']['n_rows'] / data['LIQUID']['n_tickers']
avg_rows_highly = data['HIGHLY_LIQUID']['n_rows'] / data['HIGHLY_LIQUID']['n_tickers']
print(f"{'Avg Data Points per Ticker':<30} {avg_rows_all:>18,.0f} {avg_rows_liquid:>18,.0f} {avg_rows_highly:>18,.0f}")

avg_vol_all = data['ALL']['total_volume'] / data['ALL']['n_tickers']
avg_vol_liquid = data['LIQUID']['total_volume'] / data['LIQUID']['n_tickers']
avg_vol_highly = data['HIGHLY_LIQUID']['total_volume'] / data['HIGHLY_LIQUID']['n_tickers']
print(f"{'Avg Total Volume per Ticker':<30} {avg_vol_all:>18,.0f} {avg_vol_liquid:>18,.0f} {avg_vol_highly:>18,.0f}")
print()

# Show highly liquid tickers
print('-' * 90)
print(f"HIGHLY LIQUID TICKERS ({data['HIGHLY_LIQUID']['n_tickers']}):")
print('-' * 90)
for i, ticker in enumerate(sorted(data['HIGHLY_LIQUID']['tickers'])):
    print(f'  {i+1:2}. {ticker}')
print()

# Show sample of liquid tickers
print('-' * 90)
print(f"SAMPLE OF LIQUID TICKERS (first 20 of {data['LIQUID']['n_tickers']}):")
print('-' * 90)
for i, ticker in enumerate(sorted(data['LIQUID']['tickers'])[:20]):
    print(f'  {i+1:2}. {ticker}')
print('  ...')
print()

print('=' * 90)
print('RECOMMENDATION:')
print('-' * 90)
print('  LIQUID (342 tickers):       Good balance of data quality and diversity')
print('  HIGHLY_LIQUID (17 tickers): Best for initial testing and fast iteration')
print('=' * 90)

