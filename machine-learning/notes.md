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

- ...

#### Pandas

#### Matplotlib

#### Seaborn

## (3) Introduction to Machine Learning
