
# Fraud Detection & Risk Prediction System


# Step 1: Data loading & Exploratory Data Analysis (EDA)

# Dataset: Credit Card Fraud Detection (ULB) - 284,807 transactions, 0.17% fraud rate.

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

print(pd.__version__)  # confirm pandas version for reproducibility

df = pd.read_csv('creditcard.csv')

print(df.shape)     # (rows, columns) -> confirms full dataset loaded (284807, 31)
print(df.head())    # quick visual sanity check on the first few rows


# ---------------------------------------------------------
# 2. STRUCTURE CHECK
# ---------------------------------------------------------

# .info() -> column dtypes + non-null counts (31 cols, all float64 except
#            Class which is int64, zero missing values anywhere)
# .describe() -> numeric summary stats (mean, std, min/max, quartiles)
# .isnull().sum() -> explicit missing-value count per column (confirms 0)

df.info()
df.describe()
df.isnull().sum()


# ---------------------------------------------------------
# 3. CLASS BALANCE
# ---------------------------------------------------------

# The single most important check in the whole project: how rare is
# fraud? This number drives almost every modeling decision later on
# (why accuracy won't work, why we need SMOTE/class weights, why we'll
# use PR-AUC instead of ROC-AUC).

print(df['Class'].value_counts())                      # raw counts
print(df['Class'].value_counts(normalize=True) * 100)  # as percentages

# Result: 284,315 legit (99.83%) vs 492 fraud (0.17%) -> severe imbalance


# ---------------------------------------------------------
# 4. DUPLICATE & SCALE CHECKS
# ---------------------------------------------------------
print(df.duplicated().sum())              # 1081 duplicate rows found
print(df[['Time','Amount']].describe())   # Time & Amount aren't PCA'd -> worth
                                           # eyeballing scale/outliers separately
                                           # from the anonymized V1-V28 features

# Before dropping duplicates, check whether they're concentrated in one
# class -- fraud cases are already extremely rare, so we don't want to
# accidentally lose a big chunk of them without noticing.

print(df[df.duplicated()]['Class'].value_counts())

# Result: 1062 legit duplicates / 19 fraud duplicates (~3.9% of all fraud)
# Decision: drop them (likely real data-collection artifacts), but note
# the minority-class impact check explicitly in the README.


# ---------------------------------------------------------
# 5. FIRST LOOK: Amount distribution, Legit vs Fraud (raw counts)
# ---------------------------------------------------------
# Note: different y-axis scales and no shared x-limit here, so this
# first version is a rough look only -- see the corrected version below.

fig, axes = plt.subplots(1, 2, figsize=(12,4))
df[df['Class']==0]['Amount'].hist(bins=50, ax=axes[0])
axes[0].set_title('Amount - Legit')
df[df['Class']==1]['Amount'].hist(bins=50, ax=axes[1])
axes[1].set_title('Amount - Fraud')
plt.show()

# ---------------------------------------------------------
# 6. DROP DUPLICATES
# ---------------------------------------------------------

df = df.drop_duplicates()
print(df.shape)                     # confirm new row count after dropping
print(df['Class'].value_counts())   # confirm class counts after dropping

# ---------------------------------------------------------
# 7. CORRECTED VISUALIZATION: Amount distribution, Legit vs Fraud
# ---------------------------------------------------------
# Fixed version of step 5: same x-axis range (0-2500) and density=True
# on both plots, so we're comparing distribution SHAPE fairly rather
# than being misled by different axis scales and raw counts.

fig, axes = plt.subplots(1, 2, figsize=(12,4))
bins = 50
xlim = (0, 2500)  # same range on both plots for a fair comparison

df[df['Class']==0]['Amount'].hist(bins=bins, range=xlim, density=True, ax=axes[0])
axes[0].set_title('Amount - Legit (density)')
axes[0].set_xlim(xlim)

df[df['Class']==1]['Amount'].hist(bins=bins, range=xlim, density=True, ax=axes[1])
axes[1].set_title('Amount - Fraud (density)')
axes[1].set_xlim(xlim)

plt.show()

# ---------------------------------------------------------
# NEXT STEP (not yet implemented): train/test split
# ---------------------------------------------------------

# IMPORTANT: split the data BEFORE any scaling or SMOTE is applied,
# to avoid data leakage (information from the test set leaking into
# training via transformations fit on the full dataset).