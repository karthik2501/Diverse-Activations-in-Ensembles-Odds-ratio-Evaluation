import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy

# Generate data
x = np.linspace(-10, 10, 1000)

# Derivative of the sigmoid function
sigmoid_derivative = np.exp(-x) / (1 + np.exp(-x))**2

# PDF of normal distribution with variance=40
normal_pdf_var_40 = norm.pdf(x, scale=np.sqrt(40))

# PDF of normal distribution with variance=20
normal_pdf_var_20 = norm.pdf(x, scale=np.sqrt(20))

# PDF of Cauchy distribution with gamma=1
cauchy_pdf_gamma_1 = cauchy.pdf(x, scale=1) / np.pi / (1 + x**2)

# PDF of Cauchy distribution with gamma=2
cauchy_pdf_gamma_2 = cauchy.pdf(x, scale=2) / np.pi / (4 + x**2)

# Plot the derivatives
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid_derivative, label='Sigmoid')
plt.plot(x, normal_pdf_var_40, label='Normal ($\mu=0,\sigma$=40)')
plt.plot(x, normal_pdf_var_20, label='Normal ($\mu=0,\sigma$=20)')
plt.plot(x, cauchy_pdf_gamma_1, label='Cauchy ($x_{0}=0,\gamma$=1)')
plt.plot(x, cauchy_pdf_gamma_2, label='Cauchy ($x_{0}=0,\gamma$=2)')

# Add labels and title
plt.xlabel('x',fontsize=35)
plt.ylabel("A'(x)",fontsize=35)
plt.legend(fontsize=24)

# Show the plot
plt.show()

