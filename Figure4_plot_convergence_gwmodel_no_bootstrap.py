# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 12:12:00 2021

@author: Jing
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Load data
N = 1000
ret_brute_force = np.load('ret_brute_force_gwmodel_no_bootstrap_' + str(N) + '.npy')
ret_fast_estimator = np.load('ret_fast_estimator_gwmodel_2000000_no_bootstrape.npy')

# Convert to percent value
ret_brute_force = ret_brute_force * 100
ret_fast_estimator = ret_fast_estimator * 100


def start_valid_island(a, thresh=1, window_size=100):
    m = a<thresh
    me = np.r_[False,m,False]
    idx = np.flatnonzero(me[:-1]!=me[1:])
    lens = idx[1::2]-idx[::2]
    return idx[::2][(lens >= window_size).argmax()]


#%% Absolute relative error
plt.figure(figsize=(12, 8))

# PSI of Rechagre 
plt.subplot(4, 3, 1)
abs_rel_err = abs(ret_brute_force [-1, 0] - ret_brute_force [:, 0]) / ret_brute_force [-1, 0] * 100
plt.plot(np.arange(2, N, 1, dtype='int64')**3 * 8 / 1e9, abs_rel_err, 
         color='b', alpha=0.5, lw=1.75, linestyle='-', label='(a1) Recharge\nBrute force MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=200)
plt.axvline(x=np.arange(2, N, 1)[xid]**3 * 8 / 1e9, color='k', lw=2, linestyle=':')
#plt.text(xid**3 * 8 / 1e9, 2, '  ' + str(xid**3 * 8 / 1e9)[:4], ha='left', va='center')
plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 8 * N**3 / 1e9])
plt.ylim([0, 5])
plt.xticks([np.arange(2, N, 1)[xid]**3 * 8 / 1e9, 2, 4, 6, 8], 
           [str(xid**3 * 8 / 1e9)[:4], '2', '4', '6', '8'])
plt.ylabel('Abs. relative err.\n' + '$PS_{K}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^9)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)
plt.title('', fontsize=12)


plt.subplot(4, 3, 4)
abs_rel_err = abs(ret_brute_force [-1, 0] - ret_fast_estimator[:, 0]) / ret_brute_force [-1, 0] * 100
plt.plot(np.arange(10, 2000000, 1000) * 5 * 8 / 1e6, abs_rel_err,
         color='r', alpha=0.5, lw=2, label='(b1) Recharge\nQuasi-MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=100)
plt.axvline(x=np.arange(10, 2000000, 1000)[xid] * 5 * 8 / 1e6, color='k', lw=2, linestyle=':')
#plt.text(np.arange(10, 2000000, 1000)[xid] * 5 * 8 / 1e6, 2, '  ' + str(np.arange(10, 2000000, 1000)[xid] * 5 * 8 / 1e6)[:4], ha='left', va='center')
plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 200000 * 5 * 8 / 1e6])
plt.ylim([0, 5])
plt.xticks([np.arange(10, 2000000, 1000)[xid] * 5 * 8 / 1e6, 2, 4, 6, 8], 
           [str(np.arange(10, 2000000, 1000)[xid] * 5 * 8 / 1e6)[:4], '2', '4', '6', '8'])
plt.ylabel('Abs. relative err.\n' + '$PS_{K}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^6)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)


# PSI of Geology 
plt.subplot(4, 3, 2)
abs_rel_err = abs(ret_brute_force [-1, 2] - ret_brute_force [:, 2]) / ret_brute_force [-1, 2] * 100
plt.plot(np.arange(2, N, 1, dtype='int64')**3 * 8 / 1e9, abs_rel_err, 
         color='b', alpha=0.5, lw=1.5, linestyle='-', label='(a2) Geology\nBrute force MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=100)
plt.axvline(x=np.arange(2, N, 1)[xid]**3 * 8 / 1e9, color='k', lw=2, linestyle=':')
#plt.text(xid**3 * 8 / 1e9, 2, '  ' + str(xid**3 * 8 / 1e9)[:4], ha='left', va='center')
plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 8 * N**3 / 1e9])
plt.ylim([0, 5])
plt.xticks([0, np.arange(2, N, 1)[xid]**3 * 8 / 1e9, 4, 6, 8], 
           ['0', str(xid**3 * 8 / 1e9)[:4],  '4', '6', '8'])
plt.ylabel('Abs. relative err.\n' + '$PS_{K}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^9)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)
plt.title('', fontsize=12)

plt.subplot(4, 3, 5)
abs_rel_err = abs(ret_brute_force [-1, 2] - ret_fast_estimator[:, 2]) / ret_brute_force [-1, 2] * 100
plt.plot(np.arange(10, 2000000, 1000)* 5 * 8 / 1e6, abs_rel_err, 
         color='r', alpha=0.5, lw=2, label='(b2) Geology\nQuasi-MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=100)
plt.axvline(x=np.arange(10, 2000000, 1000)[xid] * 5 * 8 / 1e6, color='k', lw=2, linestyle=':')
#plt.text(np.arange(10, 2000000, 1000) * 5 * 8 / 1e6, 2, str(xid**3 * 8 / 1e6)[:4], ha='left', va='center')
plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 200000 * 5 * 8 / 1e6])
plt.ylim([0, 5])
plt.xticks([np.arange(10, 2000000, 1000)[xid] * 5 * 8 / 1e6, 2, 4, 6, 8], 
           [str(np.arange(10, 2000000, 1000)[xid] * 5 * 8 / 1e6)[:4],  '2', '4', '6', '8'])
plt.ylabel('Abs. relative err.\n' + '$PS_{K}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^6)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)

# PSI of Snowmelt 
plt.subplot(4, 3, 3)
abs_rel_err = abs(ret_brute_force [-1, 4] - ret_brute_force [:, 4]) / ret_brute_force [-1, 4] * 100
plt.plot(np.arange(2, N, 1, dtype='int64')**3 * 8 / 1e8, abs_rel_err, 
         color='b', alpha=0.5, lw=1.5, linestyle='-', label='(a3) Snowmelt\nBrute force MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=100)
plt.axvline(x=np.arange(2, N, 1)[xid]**3 * 8 / 1e8, color='k', lw=2, linestyle=':')
#plt.text(xid**3 * 8 / 1e8, 2, '  ' + str(xid**3 * 8 / 1e8)[:4], ha='left', va='center')
plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 8 * N**3 / 1e8 * 0.1])
plt.ylim([0, 5])
plt.xticks([0, np.arange(2, N, 1)[xid]**3 * 8 / 1e8, 4, 6, 8], 
           ['0', str(xid**3 * 8 / 1e8)[:4],  '4', '6', '8'])
plt.ylabel('Abs. relative err.\n' + '$PS_{K}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^8)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)
plt.title('', fontsize=12)

plt.subplot(4, 3, 6)
abs_rel_err = abs(ret_brute_force [-1, 4] - ret_fast_estimator[:, 4]) / ret_brute_force [-1, 4] * 100
plt.plot(np.arange(10, 2000000, 1000)* 5 * 8 / 1e5, abs_rel_err, 
         color='r', alpha=0.5, lw=2, label='(b3) Snowmelt\nQuasi-MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=100)
plt.axvline(x=np.arange(10, 2000000, 1000)[xid] * 5 * 8 / 1e5, color='k', lw=2, linestyle=':')
#plt.text(np.arange(10, 2000000, 1000) * 5 * 8 / 1e6, 2, str(xid**3 * 8 / 1e6)[:4], ha='left', va='center')
plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 200000 * 5 * 8 / 1e5 * 0.1])
plt.ylim([0, 5])
plt.xticks([0, np.arange(10, 2000000, 1000)[xid] * 5 * 8 / 1e5, 4, 6, 8], 
           ['0', str(np.arange(10, 2000000, 1000)[xid] * 5 * 8 / 1e5)[:4],  '4', '6', '8'])
plt.ylabel('Abs. relative err.\n' + '$PS_{K}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^5)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)


# PST of Rechagre 
plt.subplot(4, 3, 7)
abs_rel_err = abs(ret_brute_force [-1, 6] - ret_brute_force [:, 6]) / ret_brute_force [-1, 6] * 100
plt.plot(np.arange(2, N, 1, dtype='int64')**3 * 8 / 1e9, abs_rel_err, 
         color='b', alpha=0.5, lw=1.75, linestyle='-', label='(c1) Recharge\nBrute force MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=100)
plt.axvline(x=np.arange(2, N, 1)[xid]**3 * 8 / 1e9, color='k', lw=2, linestyle=':')
#plt.text(xid**3 * 8 / 1e9, 2, '  ' + str(xid**3 * 8 / 1e9)[:4], ha='left', va='center')
plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 8 * N**3 / 1e9])
plt.ylim([0, 5])
plt.xticks([np.arange(2, N, 1)[xid]**3 * 8 / 1e9, 2, 4, 6, 8], 
           [str(xid**3 * 8 / 1e9)[:4], '2',  '4', '6', '8'])
plt.ylabel('Abs. relative err.\n' + '$PS_{TK}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^9)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)
plt.title('', fontsize=12)

plt.subplot(4, 3, 10)
abs_rel_err =  abs(ret_brute_force [-1, 6] - ret_fast_estimator[:, 6]) / ret_brute_force [-1, 6] * 100
plt.plot(np.arange(10, 2000000, 1000)* 5 * 8 / 1e6, abs_rel_err,
         color='r', alpha=0.5, lw=2, label='(d1) Recharge\nQuasi-MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=100)
plt.axvline(x=np.arange(10, 2000000, 1000)[xid]* 5 * 8 / 1e6, color='k', lw=2, linestyle=':')
plt.xticks([np.arange(10, 2000000, 1000)[xid]* 5 * 8 / 1e6, 2, 4, 6, 8], 
           [str(np.arange(10, 2000000, 1000)[xid]* 5 * 8 / 1e6)[:4], '2',  '4', '6', '8'])

plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 200000 * 5 * 8 / 1e6])
plt.ylim([0, 5])
plt.ylabel('Abs. relative err.\n' + '$PS_{TK}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^6)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)


# PST of Geology 
abs_rel_err = abs(ret_brute_force [-1, 8] - ret_brute_force [:, 8]) / ret_brute_force [-1, 8] * 100
plt.subplot(4, 3, 8)
plt.plot(np.arange(2, N, 1, dtype='int64')**3 * 8 / 1e9, abs_rel_err, 
         color='b', alpha=0.5, lw=1.5, linestyle='-', label='(c2) Geology\nBrute force MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=100)
#plt.axvline(x=np.arange(2, N, 1)[xid]**3 * 8 / 1e9, color='k', lw=2, linestyle=':')
plt.axvline(x=5.07, color='k', lw=2, linestyle=':')
#plt.text(xid**3 * 8 / 1e9, 2, '  ' + str(xid**3 * 8 / 1e9)[:4], ha='left', va='center')
plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 8 * N**3 / 1e9])
plt.ylim([0, 5])
plt.xticks([0, 2, 4, 5.07, 6, 8], 
           ['0', '2', '4', 5.07, '6', '8'])
plt.ylabel('Abs. relative err.\n' + '$PS_{TK}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^9)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)
plt.title('', fontsize=12)

plt.subplot(4, 3, 11)
abs_rel_err = abs(ret_brute_force [-1, 8] - ret_fast_estimator[:, 8]) / ret_brute_force [-1, 8] * 100 
plt.plot(np.arange(10, 2000000, 1000)* 5 * 8 / 1e6, abs_rel_err,
         color='r', alpha=0.5, lw=2, label='(d2) Geology\nQuasi-MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=100)
plt.axvline(x=np.arange(10, 2000000, 1000)[xid]* 5 * 8 / 1e6, color='k', lw=2, linestyle=':')
plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 200000 * 5 * 8 / 1e6])
plt.ylim([0, 5])
plt.xticks([np.arange(10, 2000000, 1000)[xid]* 5 * 8 / 1e6, 2, 4, 6, 8], 
           [str(np.arange(10, 2000000, 1000)[xid]* 5 * 8 / 1e6)[:4], '2',  '4', '6', '8'])

plt.ylabel('Abs. relative err.\n' + '$PS_{TK}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^6)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)

# PST of Snowmelt 
plt.subplot(4, 3, 9)
abs_rel_err = abs(ret_brute_force [-1, 10] - ret_brute_force [:, 10]) / ret_brute_force [-1, 10] * 100
plt.plot(np.arange(2, N, 1, dtype='int64')**3 * 8 / 1e7, abs_rel_err,
         color='b', alpha=0.5, lw=1.5, linestyle='-', label='(c3) Snowmelt\nBrute force MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=100)
plt.axvline(x=np.arange(2, N, 1)[xid]**3 * 8 / 1e7, color='k', lw=2, linestyle=':')
#plt.text(xid**3 * 8 / 1e7, 2, '  ' + str(xid**3 * 8 / 1e7)[:4], ha='left', va='center')
plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 8 * N**3 / 1e7 * 0.01])
plt.ylim([0, 5])
plt.xticks([np.arange(2, N, 1)[xid]**3 * 8 / 1e7, 2, 4, 6, 8], 
           [str(xid**3 * 8 / 1e7)[:4], '2',  '4', '6', '8'])
plt.ylabel('Abs. relative err.\n' + '$PS_{TK}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^7)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)
plt.title('', fontsize=12)

plt.subplot(4, 3, 12)
abs_rel_err = abs(ret_brute_force [-1, 10] - ret_fast_estimator[:, 10]) / ret_brute_force [-1, 10] * 100
plt.plot(np.arange(10, 2000000, 1000)* 5 * 8 / 1e5, abs_rel_err, 
         color='r', alpha=0.5, lw=2, label='(d3) Snowmelt\nQuasi-MC')
plt.axhline(y=1.0, color='k', lw=2, linestyle=':')
xid = start_valid_island(abs_rel_err, thresh=1, window_size=100)
plt.axvline(x=np.arange(10, 2000000, 1000)[xid]* 5 * 8 / 1e5, color='k', lw=2, linestyle=':')
plt.text(7.0, 0.4, '1%', fontsize=12, c='k')
plt.xlim([0, 200000 * 5 * 8 / 1e5 * 0.1])
plt.ylim([0, 5])
plt.xticks([np.arange(10, 2000000, 1000)[xid]* 5 * 8 / 1e5, 2, 4, 6, 8], 
           [str(np.arange(10, 2000000, 1000)[xid]* 5 * 8 / 1e5)[:4], '2',  '4', '6', '8'])
plt.ylabel('Abs. relative err.\n' + '$PS_{TK}$' + ' (%)', fontsize=12)
plt.xlabel('Number of simulations ' + '$(x10^5)$', fontsize=12)
plt.legend(loc='upper right', fontsize=12, frameon=False)


plt.subplots_adjust(hspace=0.45, wspace=0.3)


