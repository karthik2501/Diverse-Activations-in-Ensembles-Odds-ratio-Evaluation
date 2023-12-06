import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy

# Generate data
x = np.linspace(-10, 10, 1000)

# Sigmoid function
sigmoid = 1 / (1 + np.exp(-x))

# CDF of normal distribution with variance=40
normal_cdf_var_40 = norm.cdf(x, scale=np.sqrt(40))

# CDF of normal distribution with variance=20
normal_cdf_var_20 = norm.cdf(x, scale=np.sqrt(20))

# CDF of Cauchy distribution with gamma=1
cauchy_cdf_gamma_1 = cauchy.cdf(x, scale=1)

# CDF of Cauchy distribution with gamma=2
cauchy_cdf_gamma_2 = cauchy.cdf(x, scale=2)

# Plot the graphs
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid, label='Sigmoid')
plt.plot(x, normal_cdf_var_40, label='Normal ($\mu=0,\sigma$=40)')
plt.plot(x, normal_cdf_var_20, label='Normal ($\mu=0,\sigma$=20)')
plt.plot(x, cauchy_cdf_gamma_1, label='Cauchy ($x_{0}=0,\gamma$=1)')
plt.plot(x, cauchy_cdf_gamma_2, label='Cauchy ($x_{0}=0,\gamma$=2)')

# Add labels and title
plt.xlabel('x',fontsize=35)
plt.ylabel('A(x)',fontsize=35)
#plt.title('Activations',fontsize=35)
plt.legend(fontsize=24)

# Show the plot
plt.show()

