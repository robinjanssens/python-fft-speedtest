import numpy as np
import scipy.io
import pyfftw
import time
import matplotlib.pyplot as plt

# ------------------------------
# Config
# ------------------------------
# prime 8191
# 401407, 73727
mode = 'complex' # ['real', 'complex']
length_max  = 32*1024
length_step = 1
iterations = 1

# ------------------------------
# Test
# ------------------------------
binary_lengths = np.arange(length_step, length_max, length_step)
# arbitrary_lengths = np.arange(length_step, length_max, length_step)+8191
# lengths = np.sort(np.concatenate([binary_lengths,arbitrary_lengths]))
lengths = binary_lengths
print('lengths: '+str(lengths))
result_numpy = []
result_fftw = []
time_total_start = time.time()

for l in lengths:
    time_numpy = []
    time_fftw = []
    # print('calculating length: '+str(l))
    for i in range(iterations):
        if mode == 'complex':
            data = np.random.rand(l) + np.random.rand(l) * 1j
        else: # real
            data = np.random.rand(l)

        time_start = time.time()
        result = np.fft.fft(data)
        time_end = time.time()
        time_numpy.append(time_end - time_start)

        time_start = time.time()
        result = pyfftw.interfaces.numpy_fft.fft(data)
        time_end = time.time()
        time_fftw.append(time_end - time_start)

        scipy.io.savemat('output.mat', mdict={'lengths': lengths, 'result_numpy': result_numpy, 'result_fftw': result_fftw})

    result_numpy.append(np.average(time_numpy))
    result_fftw.append(np.average(time_fftw))

time_total_end = time.time()
print('Total time: '+str(time_total_end - time_total_start)+'s')

# ------------------------------
# Plot
# ------------------------------
fig, ax = plt.subplots()
ax.plot(lengths, result_numpy)
ax.plot(lengths, result_fftw)

ax.set(xlabel='length [Sa]', ylabel='time [s]',
       title='Time of fft calculation in function of sample length')
ax.grid()

# fig.savefig("test.png")
plt.show()
