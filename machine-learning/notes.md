# Notes

## (2) Introduction to Machine Learning

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

### Useful Python Libraries

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

#### Matplotlib

#### Seaborn

## (3) Introduction to Machine Learning
