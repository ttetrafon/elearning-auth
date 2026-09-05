from typing import Any, Tuple, cast

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split  # type: ignore

# (1) data creation
rng = np.random.default_rng(42)
n: int = 600

label_age: str = "age"
label_city: str = "city"
label_income: str = "income"
label_job: str = "job"

city_athens: str = "Athens"
city_berlin: str = "Berlin"
city_paris: str = "Paris"
city_rome: str = "Rome"

job_student: str = "student"
job_engineer: str = "engineer"
job_teacher: str = "teacher"
job_sales: str = "sales"

unknown_value: str = "Unknown"

# build initial data
df: pd.DataFrame = pd.DataFrame({
    label_age: rng.integers(18, 70, size=n).astype(float),
    label_income: rng.normal(45000, 15000, size=n).clip(5000, 120000),
    label_city: rng.choice([city_athens, city_paris, city_berlin, city_rome], size=n, p=[0.35, 0.25, 0.25, 0.15]),
    label_job: rng.choice([job_student, job_engineer, job_teacher, job_sales], size=n),
})

# introduce missing values
df.loc[rng.choice(n, size=int(0.06 * n), replace=False), label_age] = np.nan
df.loc[rng.choice(n, size=int(0.08 * n), replace=False), label_income] = np.nan
df.loc[rng.choice(n, size=int(0.04 * n), replace=False), label_city] = np.nan

# generate binary target from features + noise
age_f: pd.Series = df[label_age].fillna(df[label_age].median())
income_f: pd.Series = df[label_income].fillna(df[label_income].median())
city_f: pd.Series = df[label_city].fillna(unknown_value)
job_f: pd.Series = df[label_job].fillna(unknown_value)

score: pd.Series = (
    -6.0
    + 0.06 * age_f
    + 0.00006 * income_f
    + (city_f == city_paris).astype(float) * 0.7
    + (city_f == city_berlin).astype(float) * 0.3
    + (job_f == job_engineer).astype(float) * 0.6
    + rng.normal(0, 0.8, size=n)
)

proba = 1 / (1 + np.exp(-score))
y: np.ndarray[Tuple[Any, ...], np.dtype[Any]] = (proba > 0.5).astype(int)
X: pd.DataFrame = df.copy()

print("Dataset shape:", X.shape)
print("Positive rate (label 1):", y.mean().round(3))
print()
print(X.head())
print()
print("Missing values per column:")
print(X.isnull().sum())

# (2) train/test split
X_train, X_test, y_train, y_test = cast(
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)
print()
print("Positive rate-train:", y_train.mean().__round__(3))
print("Positive rate-test:", y_test.mean().__round__(3))

# (3) preprocessing

# (4) pipeline

# (5) evaluation

# (6) cross-validation
