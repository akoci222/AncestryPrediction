import seaborn as sns
import matplotlib.pyplot as plt

confusion_matrix = [
    [80, 1, 0, 0, 0],
    [0, 16, 1, 18, 4],
    [0, 0, 64, 0, 0],
    [0, 0, 0, 64, 0],
    [0, 1, 0, 6, 53]
]

class_labels = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('randomforest Confusion Matrix')
plt.show()