import pandas as pd
import matplotlib.pyplot as plt

fst_data = pd.read_csv("src/visualisation/FST/final_fst.fst", sep='\t')

plt.figure(figsize=(8, 6))
plt.hist(fst_data["FST"], bins=100, edgecolor='black')
plt.xlabel("Fst Values")
plt.ylabel("Frequency")
plt.savefig('fst1.png')
plt.show()

