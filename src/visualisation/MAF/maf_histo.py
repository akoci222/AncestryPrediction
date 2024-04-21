import matplotlib.pyplot as plt
import numpy as np


maf_val = np.loadtxt('src/visualisation/MAF/final_data_freq.txt', skiprows=1)

plt.figure(figsize=(8, 6))
plt.hist(maf_val, bins=50, edgecolor='black')
plt.xlabel('Minor Allele Frequency Distribution')
plt.ylabel('Frequency')
plt.savefig('maf.png')
plt.show()