import matplotlib.pyplot as plt
import numpy as np

import geosss as gs

# autoregressive process
mu = 0.
tau = 0.99
sigma = 1.

# generate samples
x = [0.]
while len(x) < 100_000:
    x.append(np.random.randn() * sigma + tau * x[-1] + (1-tau) * mu)
x = np.array(x)

# two ways to compute the autocorrelation
with gs.take_time("real space"):
    a = gs.acf(x)
with gs.take_time("fft"):
    b = gs.acf_fft(x)

plt.plot(a,  lw=5, label='real')
plt.plot(b, label='fft')
plt.xlim(0, 2000)
plt.legend()
plt.show()
