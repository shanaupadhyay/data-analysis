import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset/mahindra.csv")

df.columns = df.columns.str.strip()

# Automatically find FixedAssets column
fa = [c for c in df.columns if "fixed" in c.lower() and "asset" in c.lower()][0]

# Dependent variable
y = df[fa]

# Independent variables (numeric only)
X = df.select_dtypes("number").drop(columns=[fa], errors="ignore")

# Remove columns containing 'other' and 'liab'
X = X.drop(columns=[c for c in X.columns if "other" in c.lower() and "liab" in c.lower()],
           errors="ignore")

# Add constant
model = sm.OLS(y, sm.add_constant(X), missing="drop").fit()

print(model.summary())

# Bar chart of coefficients
coef = model.params.drop("const", errors="ignore")

plt.bar(coef.index, coef.values)
plt.xticks(rotation=45)
plt.title("Regression Coefficients_Shana-12303228")
plt.tight_layout()
plt.show()
