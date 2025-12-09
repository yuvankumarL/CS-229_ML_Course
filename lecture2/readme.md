

Flow of Supervised Learing

    Training set
        |
        | (data feeded to)
        |
    Learing algorithms
        |
        | (job of l_a is to )
        |
    hypothesis(h)
    output the function to make prediction, lets call this as hypothesis(h)

House prediction data:

| Size(feet^2) | price ($1000s) |
| :----------- | -------------: |
|     2104     |     400        |
|     1416     |      232       |
|     1534     |      315       |
|     852      |      178       |
|     .        |      .         |
|     .        |      .         |
|     .        |      .         |

how to represent the hypothesis?

$$ h(x) = \theta + \theta_1*X $$
h(x) --> input size x

$\theta (output) $

the equation explains,
for the given input of size X, the output number as linear function of the size X.


House prediction data after adding no_of_bedrooms:

| Size(feet^2) | no_of_bedrooms | price ($1000s) |
| :----------- | :------------: | -------------: |
|     2104     |       3        |      400       |
|     1416     |       2        |      232       |
|     1534     |       3        |      315       |
|     852      |       2        |      178       |
|     .        |       .        |      .         |
|     .        |       .        |      .         |
|     .        |       .        |      .         |

now the hypothesis will be,

$$ h(x) = \theta + \theta_1 * X_1 + \theta_2 * X_2 $$
$ X_1 -> size $

$ X_2 -> noofbedrooms $


In order to simplify the notation, we can write the hypothesis as 


$$ \sum_{j=0}^n (Oj * Xj) $$

where $X\theta$ = 1, n= no of features

$\theta = [ \theta_0
        \theta_1
        \theta_2]$

X = [ X0   -> always 1
        X1   -> size
        X2]  -> no_of_bedrooms

O = "parameters" (of the learning algorithmns, job of the learing algorithm is to choose parameters theta(O) that allows you to make a good predictions about the prices of houses)
m = "no of training examples" (no. of rows in table above)
X = "inputs" (features)
y = "output" (target variable)
(x, y) = one training example
(xi, yi) = ith training example
n = no of features(2)

how to choose parameters data?
    
lets, choose $\theta $ such that h(x) ≈ y for the training examples.
we can denote $h_\theta$(x) as h(x) 

$$ \tfrac{1}{2} \sum_{i=1}^m (h_\theta(x^i) - y^i)^2 $$

In linear regression we gonna define cost function J of Theta to be equal to that.

$$ 
J = \tfrac{1}{2} \sum_{i=1}^m (h_\theta(x^i) - y^i)^2 
$$

    J -> cost function of theta

we will find parameters data that minimizes the cost function J of theta

$ minimize_\theta J(\theta) $


## Gradient Descent

start with some value of $\theta (say \theta = \vec{O}) $

keep changing $\theta$ to reduce $J(\theta)$ 


