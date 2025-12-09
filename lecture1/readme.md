
### AI is the new electricity

---

A computer program is said to learn from experience E with repect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

---

## Supervised Learning

eg: House prediction

X --> size(feet^2)

y --> price

In supervised learning, you will be given a dataset of (x, y) and your goal is to learn mapping X --> y

This called a regression problem

Different type of problem:

    1. Breast cancer

        X --> tumor size
        y --> malignant or not? (1 or 0)

        This is an example of classification problem, because y is a discerete numbers
        
        Discrete numbers are distinct, countable values (like integers: 1, 2, 3), with gaps in between, used for counting

    2. House prediction

        y is a real number

        Real numbers are continuous, including all rational and irrational numbers (like pi, sqrt{2}, 1.5), forming a complete number line with no gaps, used for measuring.


for most of the ml agorithms the input(X) size will be multi dimensional

eg., for breast cancer prediction you are given with 2 features X (tumor_size, age)

SVM allows you to use infinite-dimensional vector

Its difficult to store infinite amount of data, to do this we use SVM specifically the techinical methods called Kernels, how to build learning algorithms so infinite long list of features.

the heart of supervised learning is the idea that during training you are given inputs X together with labels y and you give it both at the same time, and the job of your learning algorithmn is to find a mapping, so that given an new X you can map it to the most appropriate output y.
    

## Machine learning statergy (Learning Theory)

Machine Learning (ML) Strategy based on Learning Theory, involves mathematical frameworks to understand how and why algorithms learn, focusing on data distribution, model complexity (hypothesis spaces), and performance guarantees, moving beyond just what to do (like supervised vs. unsupervised learning) to analyze the inherent difficulty of problems, proving algorithm success conditions, and ensuring models generalize well from limited training data to unseen data, using concepts like PAC learning and VC dimension to analyze feasibility and bounds


## unsupervised learning

here you will be given with only the input X and no y. from this you need to give me something interesting structure in this data.

here we use clustering

eg., 

    1. orgainize computing cluster
    2. social network analysis
    3. market segmentation
    4. Astronomical data analysis

Is unsupervised learning same as clustering?
    
    No, its not. UnSup learning is the concept of using unlabelled data to find some interesting thing from it.

The cocktail party problem is
the human ability to focus on one conversation amidst many other sounds, and the difficulty machines have in replicating this feat. It's a challenge in signal processing and machine learning that involves separating a desired audio signal (like a single voice) from a mixture of other noises and voices. The problem has applications in fields such as artificial intelligence, telecommunications, and bioacoustics

Independent compound analysis



## reinforcement learning

    eg., Flying an helicopter

    no one knows the optimal way to fly a helicopter, if you fly a helicopter, you have two contro sticks that you are moving, but no one knows the optimal way to move the control sticks. so the way you can get a helicopter fly itself is, let the helicopter do whatever think of this as training a dog. You can't teach the dog optimal way to behave. so how can you train a dog? you let the dog whatever it wants and then whenever it behaves well, you go "Oh, good dog" and when it misbehaves you go "bad dog". And Moreover time, dog learns to do the good dog things and some bad dog things.

