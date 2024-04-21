import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('src/visualisation/PCA/pca.csv')

pc1 = data['PC1'].values
pc2 = data['PC2'].values
pc3 = data['PC3'].values
label = data['Superpopulation code'].values

populations = data['Superpopulation code'].unique()

leg = ['o', 's', '^', 'D', 'v', '*']
key_dict = dict(zip(populations, leg[:len(populations)]))

plt.figure(figsize=(12, 10))
for pop, leg in key_dict.items():
    plt.scatter(pc1[label == pop], pc2[label == pop], marker=leg, label=pop, alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PC1 vs PC2')
plt.legend(loc='upper left')
plt.savefig('pca12.png')
plt.show()


