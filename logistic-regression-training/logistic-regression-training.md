What Logistic Regression Does

Logistic regression is a binary classification algorithm. Given a set of input features, it predicts the probability that an example belongs to the positive class (class 1).

Despite its name, logistic regression is used for classification, not regression. The term "regression" comes from the fact that the model learns a set of coefficients, similar to linear regression. However, instead of predicting an unrestricted real-valued output, logistic regression transforms the model's linear output into a probability between 0 and 1.

The model can answer questions such as:

Is this email spam or not spam?

Will this customer churn or stay?

Is this tumor malignant or benign?



---

The Model Structure

Logistic regression has two main steps.

Part 1: Linear Combination

First, the model computes a linear combination of the input features:

z = Xw + b

where:

X is the input matrix with shape (n_samples, n_features)

w is the weight vector with shape (n_features,)

b is the bias, a scalar broadcast across all samples

z is the logit (linear output) with shape (n_samples,)


For a single example with feature vector x:

z = xᵀw + b

The value z can be any real number: positive, negative, or zero.

Part 2: Sigmoid Function

The logit is then passed through the sigmoid function:

p = σ(z) = 1 / (1 + e⁻ᶻ)

The sigmoid maps any real-valued input to the range (0, 1), which we interpret as the model's predicted probability of class 1.

Therefore, the complete model is:

p = σ(Xw + b)

or equivalently:

p = 1 / (1 + e⁻⁽ˣʷ⁺ᵇ⁾)

For each example:

pᵢ = P(yᵢ = 1 | xᵢ)

So pᵢ represents the model's estimated probability that example i belongs to class 1.


---

Understanding the Sigmoid Function

The sigmoid function is:

σ(z) = 1 / (1 + e⁻ᶻ)

It has several important properties:

Its output is always strictly between 0 and 1.

σ(0) = 0.5.

σ(z) → 1 as z → +∞.

σ(z) → 0 as z → -∞.

It satisfies the symmetry property:


σ(-z) = 1 - σ(z)

Some example values:

σ(-5) ≈ 0.0067

σ(-2) ≈ 0.1192

σ(0) = 0.5

σ(2) ≈ 0.8808

σ(5) ≈ 0.9933


The sigmoid also has a particularly convenient derivative:

dσ/dz = σ(z)(1 - σ(z))

If we write p = σ(z), this becomes:

dp/dz = p(1 - p)

This derivative is important when deriving the gradient used to train logistic regression.


---

Binary Cross-Entropy Loss

The model's predictions need to be compared with the true labels. For binary classification, logistic regression commonly uses binary cross-entropy, also called log loss.

For n training examples:

L = -(1/n) Σᵢ [yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ)]

where:

n is the number of training examples

yᵢ ∈ {0, 1} is the true label

pᵢ is the predicted probability of class 1


Why Does This Formula Work?

When yᵢ = 1

The loss for that example becomes:

Lᵢ = -log(pᵢ)

If the model predicts a high probability for class 1:

pᵢ ≈ 1

then:

-log(pᵢ) ≈ 0

So the loss is small.

If the model predicts a very low probability:

pᵢ ≈ 0

then:

-log(pᵢ) → ∞

So confidently incorrect predictions receive a very large penalty.

When yᵢ = 0

The loss becomes:

Lᵢ = -log(1 - pᵢ)

If the model predicts:

pᵢ ≈ 0

then the loss is close to zero.

If the model predicts:

pᵢ ≈ 1

then the loss becomes very large.

Thus, binary cross-entropy rewards confident correct predictions and strongly penalizes confident incorrect predictions.


---

Gradient Descent for Logistic Regression

Training logistic regression means finding values of w and b that minimize the loss.

One way to do this is gradient descent.

The basic training process is:

1. Initialize w and b.


2. Compute the logits: z = Xw + b


3. Compute the probabilities: p = σ(z)


4. Compute the loss L.


5. Compute the gradients with respect to w and b.


6. Update the parameters in the opposite direction of the gradient.


7. Repeat until the loss converges or another stopping criterion is reached.



For binary cross-entropy combined with the sigmoid function, the gradients simplify nicely.

The gradient with respect to the weights is:

∂L/∂w = (1/n) Xᵀ(p - y)

The gradient with respect to the bias is:

∂L/∂b = (1/n) Σᵢ(pᵢ - yᵢ)

Here, (p - y) represents the prediction error for each example.

Parameter Updates

Using learning rate α, gradient descent updates the parameters as:

w ← w - α(∂L/∂w)

and:

b ← b - α(∂L/∂b)

The negative sign is important because we move in the direction that decreases the loss.


---

Deriving the Gradients

The elegant gradient formulas come from combining the chain rule, the sigmoid derivative, and the binary cross-entropy loss.

For a single example, the loss is:

L = -[y log(p) + (1 - y) log(1 - p)]

Since:

p = σ(z)

we can use the chain rule:

∂L/∂z = (∂L/∂p)(∂p/∂z)

First:

∂L/∂p = -y/p + (1 - y)/(1 - p)

And because:

∂p/∂z = p(1 - p)

we get:

∂L/∂z = [-y/p + (1 - y)/(1 - p)]p(1 - p)

which simplifies to:

∂L/∂z = p - y

This is one of the most useful results in logistic regression: the derivative of binary cross-entropy with respect to the logit is simply the prediction error.

Now consider the logit:

z = xᵀw + b

Its derivatives are:

∂z/∂w = x

and:

∂z/∂b = 1

Therefore, by the chain rule:

∂L/∂w = (∂L/∂z)(∂z/∂w) = (p - y)x

and:

∂L/∂b = (∂L/∂z)(∂z/∂b) = p - y

Averaging these gradients over all training examples gives:

∂L/∂w = (1/n) Xᵀ(p - y)

and:

∂L/∂b = (1/n) Σᵢ(pᵢ - yᵢ)


---

A Training Example

Consider a simple dataset with two features and four examples.

Features

X = [[1, 2], [2, 1], [-1, -1], [-2, -2]]

Labels

y = [1, 1, 0, 0]

Initialize:

w = [0, 0]

b = 0

First Forward Pass

The logits are:

z = Xw + b

Since w and b are both zero:

z = [0, 0, 0, 0]

Applying the sigmoid:

p = [0.5, 0.5, 0.5, 0.5]

So initially, the model assigns a probability of 0.5 to every example.

This makes sense: before learning anything, the model has no information that favors either class.

Compute the Gradient

The error vector is:

p - y = [-0.5, -0.5, 0.5, 0.5]

The weight gradient is:

∂L/∂w = (1/4) Xᵀ(p - y)

We have:

Xᵀ = [[1, 2, -1, -2], [2, 1, -1, -2]]

Therefore:

Xᵀ(p - y) = [-2.5, -2.5]

and hence:

∂L/∂w = [-0.625, -0.625]

For the bias:

∂L/∂b = (1/4)(-0.5 - 0.5 + 0.5 + 0.5) = 0

So:

∂L/∂b = 0

First Parameter Update

With learning rate:

α = 0.1

the new weights are:

w ← w - 0.1[-0.625, -0.625]

Therefore:

w = [0.0625, 0.0625]

The bias remains:

b = 0

Notice what happened: the weights became positive. Since the positive examples have positive feature values and the negative examples have negative feature values, positive weights increase the logits for the positive examples and decrease them for the negative examples.

This is exactly the direction we want.

After many iterations, the model learns parameters that separate the two classes.


---

Convergence and Learning Rate

The learning rate α controls the size of each gradient-descent update.

Too large: updates can overshoot the minimum, causing oscillation or divergence.

Too small: training can be unnecessarily slow.

Appropriate: the loss generally decreases toward a minimum.


There is no universal "correct" learning rate. Values such as 0.001, 0.01, or 0.1 can work depending on the scale of the features, the dataset, and the optimization method.

Feature scaling can make optimization substantially easier when features have very different scales.

Signs of Good Convergence

Typical signs include:

The training loss decreases.

The loss eventually levels off.

Parameter updates become smaller.

The optimization reaches a stable solution.


Signs of Problems

Possible warning signs include:

Loss oscillates or increases → the learning rate may be too large.

Loss decreases extremely slowly → the learning rate may be too small or the features may be poorly scaled.

Loss becomes NaN or infinite → numerical instability may be occurring.


In practice, implementations should compute the loss and sigmoid carefully to avoid numerical overflow or underflow. Many libraries instead optimize the log-loss directly from the logits using a numerically stable formulation.


---

Making Predictions

After training, the model has learned w and b.

For a new example:

Step 1: Compute the logit

z = xᵀw + b

Step 2: Convert the logit to a probability

p = σ(z)

Step 3: Convert the probability into a class prediction

Using the default threshold of 0.5:

ŷ = 1 if p ≥ 0.5

ŷ = 0 if p < 0.5

Because:

σ(0) = 0.5

the threshold p = 0.5 corresponds exactly to:

z = 0

Therefore, the decision boundary is:

xᵀw + b = 0

This is an important property of logistic regression: the probability changes nonlinearly through the sigmoid, but the decision boundary in feature space is linear.

The threshold does not have to be 0.5. In practice, it can be adjusted depending on the relative costs of false positives and false negatives.


---

The Big Picture

The entire logistic regression pipeline can be summarized as:

X → linear model → z = Xw + b → sigmoid → p = σ(z) → threshold → ŷ

During training:

X, y → z → p → L → gradients → update w, b

The key ideas to remember are:

1. Linear layer: z = Xw + b


2. Sigmoid: p = σ(z)


3. Loss: binary cross-entropy


4. Gradient: ∇wL = (1/n)Xᵀ(p - y)


5. Update: move parameters opposite the gradient


6. Decision boundary: xᵀw + b = 0



The most important intuition is that logistic regression learns a linear decision boundary, while the sigmoid converts the linear score into a value that can be interpreted as a probability.