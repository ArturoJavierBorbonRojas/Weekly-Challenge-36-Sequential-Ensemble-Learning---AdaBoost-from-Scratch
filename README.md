# Weekly-Challenge-36-Sequential-Ensemble-Learning---AdaBoost-from-Scratch
This week marks a major transition in my Machine Learning algorithms portfolio. While previous ensemble methods like the Random Forest operate on **Bagging** (training multiple deep trees in parallel independently), Week 36 focuses on **Boosting**.

I programmed an **AdaBoost (Adaptive Boosting)** classifier entirely from scratch using pure Python and NumPy. AdaBoost builds a strong predictive model by sequentially training very simple "weak learners" (Decision Stumps - trees with a maximum depth of 1). The core philosophy is to learn from mistakes: each new stump in the sequence pays mathematically more attention to the data points that the previous stumps got wrong.

## How it works (The Math)
1. **Initial Weights:** Every student in the dataset starts with an equal weight: $w_i = \frac{1}{N}$.
2. **Train Weak Learner:** The algorithm finds the single best feature and threshold that minimizes the *weighted* classification error ($\epsilon_m$).
3. **Calculate "Amount of Say" ($\alpha$):** The lower the error, the more voting power the stump gets in the final prediction.
   $$\alpha_m = \frac{1}{2} \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right)$$
4. **Update Weights:** The weights of the students who were incorrectly classified are increased exponentially, forcing the next stump to focus on them.
   $$w_i \leftarrow w_i \exp(-\alpha_m y_i h_m(x_i))$$
5. **Final Prediction:** The overall risk is calculated via a weighted majority vote of all stumps:
   $$H(x) = \text{sign} \left( \sum_{m=1}^{M} \alpha_m h_m(x) \right)$$

## Technical Highlights
* **Pure NumPy Implementation:** No `scikit-learn` dependencies. The entire exponential weight update mechanism, polarity flipping, and alpha calculations are executed mathematically.
* **Label Optimization:** Adapted the typical `(0, 1)` binary labels to `(-1, 1)` to allow the mathematical elegance of AdaBoost's sign-based updates to function natively.

## Dependencies
* Python 3.14.3
* NumPy
