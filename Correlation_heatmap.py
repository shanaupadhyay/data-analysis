import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset/mahindra.csv")

corr = df.corr(numeric_only=True).round(2)
print(corr.to_string())

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Heatmap_Shana-12303228")
plt.show()
