# Notes

## (2) Introduction to Machine Learning and Basic Python Libraries

- AL vs ML vs Deep Learning
  - AI: machines perform tasks that look intelligent
  - ML: subset of AI, learn from data
  - Deep Learning: subset of ML, multi-layer neural networks
- Applications of ML
  - Vision: object recognition
  - Text: translation, sentiment analysis, summarisation
  - Recommendations: movies, products, content feeds, etc
  - Finance: fraud detection, risk scoring
  - Operations: demand forecasting, predictive maintenance
- ML typical pipeline
  - engineering loop: `build -> evaluate -> improve data/features/model`
  - check performance on unseen/new data

### Learning Strategies

- Supervised Learning
  - `features & labels -> train model`
  - used for **classification (predict a category)** and **regression (predict a value)**
- Unsupervised learning
  - `input -> discover structure`
  - used for **clustering (group similar items)** and **dimensionality reduction (compress/visualise high-dimensional data)**
- Semi-Supervised Learning
  - a mixture of the previous two
- Reinforced Learning
  - agent interacts with an environment
  - receives rewards and learns by trial-and-error
  - used in **gameplay systems**, **decision control**, and **robotics**

### Training

- **Loss** measures how wrong predictions are and is used to optimise model parameters
  - Training is the process of minimising the loss.
- **Generalisation** is the goal - to work with accuracy on unseen data
  - Good accuracy on training data is not enough.
  - Underfitting: model is too simple
  - Overfitting: model memorises noise
- Data are split into _training_, _validation_, and _test_ sets to evaluate the model properly and avoid bias.

### Python Libraries

#### NumPy (Numeric Computing in Python)

##### Arrays

- N-dimensional structures of numbers.
- Arrays in NumPy operate vectors, and as such mathematical operations (`+`, `-`, `*`, `/`, `**`) are applied to their elements individually.
  - this includes internal methods: `np.sqrt()`, `np.log()`, `np.exp()`, `np.abs()`, ...
- Arrays support:
  - _broadcasting_: arrays can affect others during operations if their dimensions are compatible.
  - _aggregations_: an array is reduced to a summary value (`sum`, `mean`, `max`, `std`, etc)
    - an `axis=#` can be provided for matrices so the aggregation happens only on that dimension

```python
import NumPy as np

heights_list = np.array([166, 175, 179, 182, 185])
zeros = np.zeros(5) # creates an array of length 5 of 0s
ones = np.ones(10) # creates an array of length 10 of 1s
a = np.arange(10) # creates an array of length 10 with elements 0, 1, 2, ..., 9
c = np.full((2, 3), 7) # fills a 2x3 table with 7s
e = np.linspace(0, 1, 5) # creates a linear space in the (0, 1) domain containing 5 values (i.e.: [0, 0.25, 0.5, 0.75, 1])

e_double = e * 2 # -> [0, 0.5, 1, 1.5, 2]
```

- Useful methods/properties:
  - `array.shape`: returns the array's dimensions
  - `array.ndim`: returns the array's number of dimensions
  - `array[m, n]`: accesses the element at position (m, n) by _reference_
    - `:` selects the full line/column
    - `#1:#2`: selects a slice between #1 and #2
      - if `#1` is not defined, selects from start to #2
      - if `#1` is not defined, selects from #1 to end
  - `array.dtype`: returns the array's data type (**int64**, **float64**, etc)
  - `array.astype(#)`: converts the array to the specified type (e.g.: `np.float32`, `np.float64`, etc)
  - `array.reshape(dims)`: rearranges the data without modifying them; total number of elements must remain the same
  - `array @ condition`: creates a mask (e.g.: `a > 6` returns `[False, False, False, False, False, False, False, True, True, True]`)
    - `a[mask]`: filters the array based on the provided mask

##### Random Numbers

- `rng = np.random.default_rng(seed)`: use a specific seed to create a random number generator
- A generator can be used to then produce random values:
  - `rng.random()`
  - `rng.integers()`
  - `rng.normal()`

#### Pandas

- Library to work with structured, tabular data.
- Two core structures: **Series (1D)** and **DataFrame (2D)**.
  - Both structures feature _indices_ to improve access efficiency.
- Pandas can import from csv (`pd.read_csv()`), either locally or online.
- Useful methods:
  - `df.head()`:
  - `df.info()`: column types and non-null counts
  - `df.describeO()`: summary statistics for numeric columns
  - `df.dtypes()`: data types for each column
  - `df.columns()`: list of column names
  - `df.copy()`: returns a copy of the full DataFrame
  - selection and slicing done through `[]`
    - `df["column_name"]`: returns the specified column as a Series
    - `df[["column_name_1", "column_name_1"]]`: returns the specified columns as a DataFrame
    - `df.loc[]`: selects by label
    - `df.iloc[]`: selects by position
  - masking can be done in pandas line in NumPy
    - in a DataFrame masking will select rows based on the condition(s)
  - `df.isna().sum()`: returns the number of NaN values per column
  - `df.dropna()`: drop rows with NaN at any column
    - can define an axis (`dropna(axis=1)`) to drop whole columns with NaN values
  - `df["col_name"] = df["col_name"].filna(default_value)`: fill all NaN in the specified column with some specific value
  - `df.rename()`
  - `df.drop()`
  - `df.groupby()`: splits the DataFrame into groups based on a column
    - `.reset_index()` flattens the result into a regular DataFrame
  - `df.sort_values(["col_names"], ascending=True/False)`
  - `df["col_name"].value_counts()`: returns occurrences of unique values
    - `normalize=True` changes the counts to relative frequencies
  - `df.to_csv(path)`: stores the DataFrame into csv format
    - `index=False` to not save the indices
  - `df.to_json(path)`: stores the DataFrame into json format
- mathematical operations are applied to columns at once
- Pandas integrates with matplotlib for plotting data, but does not support the full matplotlib stuff

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=["a", "b", "c"])

df = pd.DataFrame({
  "name": ["Alice", "Bob"],
  "age": [25, 30],
  "score": [88, 92]
})

url = ("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")
df = pd.read_csv(url)

df["Age"] = df["Age"].filna(df["Age"].mean())

high = df[df["Fare"] > 85]

res = df[(df["Fare"] >= 85) & (df["Age"] < 30)]
# or
res = df.query("Fare >= 85 % Age < 30")

df.groupby("Pclass")["Fare"].mean()
df.groupby("Pclass")["Fare"].agg(["mean", "max", "count"])
df.groupby(["Pclass", "Sex"])["Fare"].mean().reset_index()
```

#### Matplotlib

- Data visualisation library
- **Plots**
  - `plt.plot(x: list, y: list)`: line plot
  - `plt.subplots(x, y)`: create an (x,y) matrix of plots
  - `plt.scatter(x, y, s=#)`: scatter (points) plot
  - `plt.barplot(categories, values)`
  - `plt.hist(data, bins=#, rwidth=#)`: histogram
  - `plt.show`: actually draws the plots on the screen
- **Figure options**
  - A _figure_ is the canvas that contains one or more plots.
  - `figsize=(x, y)`: defines the figure's size
  - `sharex=True`/`sharey=True`: aligns the plots in the figure on the appropriate axis to facilitate visual comparisons
- **Plot options**
  - `ax.set_title()`
  - `ax.set_xlabel()`
  - `ax.set_ylabel()`
  - `label=#`: defines the plotted data label
  - `ax.legend()`: displays the legend
  - `ax.annotate()`: creates annotations within a plot
  - `ax.set_yscale()`: changes the scale of the y axis

```python
import matplotlib.pyplot as plt
import numpy as np

a = [1, 2, 3, 4]
b = [2, 4, 4, 5]

fig_s, ax_s = plt.subplots(figsize=(4, 2))
ax.plot(a, b)
ax.set_title("My Plot")

fig_m, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(2, 3)
ax1.plot(a, b)
# ...

x = np.linspace(0, 5, 200)
fig_d, ax_d = plt.subplots(figsize(4, 2))
ax_d.plot(x, x, label="y = x")
ax_d.plot(x, x**2, label="y = x^2")
ax_d.set_title("Two Functions")
ax_d.set_xlabel("x")
ax_d.set_ylabel("y")
ax_d.legend()

x = np.linspace(2, 2*np.pi, 200)
y = np.sin(x)
imax = np.argmax(y)
fig, ax = plt.subplots(figsize(4, 2))
ax.plot(x, y)
ax.scatter([x[imax], y[imax]])
ax.annotate(
  "peak",
  xy=(x[imax], y[imax]),
  xytext=(x[imax] + 0.9, y[imax] - 0.3)
  arrowprops={"arrowstyle": "->"}
)

plt.show()
```

#### Seaborn

- Data visualisation library
  - extends **MatPlotLib**
  - integrates with **Pandas**
- Plot options:
  - `scatterplot`:
    - `hue=#`: applies colour based on some data variable
    - `scatter_kws={"s": #}`: visualises the correlation between the axis alongside its uncertainty
  - `histogram`:
    - `kde=True`: displays the Kernel Destiny Estimation
  - `boxplot`: summarises distribution by median and quantiles
  - `barplot`
  - `relplot`: creates one plot per subset to showcase relations
    - `kind=#`: determines the type of the internal plots
  - `heatmap`: correlation matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

a = [1, 2, 3, 4]
b = [2, 4, 3, 5]
sns.lineplot(x=a, y=b)

data = sns.load_dataset("tips")
# data table: { "total_bill": number, "day": string, "smoker": yes/no, "tip": number }
sns.scatterplot(data=data, x="total_bill", y="tip")

sns.scatterplot(data=data, x="total_bill", y="tip", hue="smoker", ax=ax)
ax.set_title("Tip vs total bill (coloured by smoker)")

sns.scatterplot(data=data, x="total_bill", y="tip", scatter_kws={"s": 8})

fig, ax = plt.subplots(figsize=(4, 2))
sns.histplot(data=data, x="total_bill", kde=True, ax=ax)
ax.set_title("Distribution of total_bill")

sns.boxplot(data=data, x="day", y="total_bill")

sns.barplot(data=data, x="day", y="tip")

g = sns.relplot(data=data, x="total_bill", y="tip", col="day", hue="smoker", kind="scatter", col_wrap=2, height=1.2, aspect=1.5)
g.fig.suptitle("Tip vs bill faceted by day", y=1.02)

df_num = data[["total_bill", "tip"]].copy()
df_num["tip_pct"] = data["tip"] / data["total_bill"]
corr = df_num.corr(numeric_only=True)
fig, ax = plt.subplots (figsize=(2, 2))
sns.heatmap(corr, annot=True, ax=ax)
ax.set_title("Correlation heatmap")

plt.show()
```

## (3) Supervised Learning
