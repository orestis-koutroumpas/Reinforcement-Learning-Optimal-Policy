# Reinforcement Learning Optimal Policy Estimation

This repository implements the numerical and data-driven approaches to solve the Short-sighted Reinforcement Learning problem. Specifically, it computes the optimal functions 𝜈1(𝑆) and 𝜈2(𝑆) using numerical methods, and then applies a data-driven approach to obtain approximations 𝜔(𝑢(𝑋,𝜃)) using neural networks. The data-driven approach utilizes two sets of pairs, with the functions [A1] and [C1] applied for approximation. For [C1], we demonstrate that the conditional expectations take values in the interval [0,2].


## Problem Overview

The problem involves a Markov decision process with two possible transition densities depending on the action 𝑎

1. For 𝑎𝑡=1: 𝒮𝑡+1=0.8𝒮𝑡+1.0+𝑊𝑡

2. For 𝑎𝑡=2: 𝒮𝑡+1=−2.0+𝑊𝑡
   
Where 𝑊𝑡∼𝑁(0,1).

At each new state, a reward 𝑅(𝒮)=𝑚𝑖𝑛{2,𝒮^2} is received. The goal is to solve for the optimal action policy that maximizes the expected reward.

The problem is defined as:
```𝜈𝑗(𝑋)=𝔼_𝑆𝑡+1^𝑗[ℛ(𝑆𝑡+1)|𝑆𝑡=𝑋],𝑗=1,2```

The optimal action policy is:

𝑎𝑡=𝑎𝑟𝑔𝑚𝑎𝑥 {𝜈1(𝑆𝑡),𝜈2(𝑆𝑡)}

That is, at time 𝑡, choose 𝑎𝑡=1 if \𝜈1(𝑆𝑡)>𝜈2(𝑆𝑡), otherwise choose 𝑎𝑡=2.

## Methodology

### Numerical Approach

Numerical solutions are computed for the expected rewards using the transition functions and the reward function. For a sampling interval {𝑠0,…,𝑠𝑛},, the vectors are defined as:

𝑉𝑗=[𝜈𝑗(𝑠0)⋮𝜈𝑗(𝑠𝑛)],𝑅=[ℛ(𝑠0)⋮ℛ(𝑠𝑛)]

The matrices ℱ𝑗, for j = 1, 2 are calculated based on the difference in transition probabilities.

ℱ𝑗1𝑘=0.5(𝐻𝑘(𝑠1|𝑠𝑗)−𝐻𝑘(𝑠0|𝑠𝑗)),𝑘=1,2 

ℱ𝑗𝑛𝑘=0.5(𝐻𝑘(𝑠𝑛|𝑠𝑗)−𝐻𝑘(𝑠𝑛−1|𝑠𝑗)),𝑘=1,2 

ℱ𝑗𝑖𝑘=0.5(𝐻𝑘(𝑠𝑖|𝑠𝑗)−𝐻𝑘(𝑠𝑖−2|𝑠𝑗)),𝑘=1,2,

𝑖=2,…,𝑛−1

### Data-Driven Approach

The data-driven approach uses neural networks to approximate 𝑢(𝑋,𝜃𝜊𝑗). The networks are trained to approximate the expected rewards for each state. The cost function for the neural network is:

𝐽̂(𝜃𝑗)=1𝑛𝑗Σ{𝜑(𝑢(𝑋𝑖𝑗,𝜃𝑗))+ℛ(𝑌𝑖𝑗)𝜓(𝑢(𝑋𝑖𝑗,𝜃𝑗))}

where 𝑢(𝑋𝑖𝑗,𝜃𝑗) is the neural network approximation.

The optimization algorithm used is Gradient Descent:

𝜃𝑡𝑗=𝜃𝑡−1𝑗−𝜇Σ[ℛ(𝑌𝑖𝑗)−𝜔(𝑢(𝑋𝑖𝑗,𝜃𝑡−1𝑗))]𝑛𝑗𝑖=1𝜌(𝑢(𝑋𝑖𝑗,𝜃𝑡−1𝑗))∇𝜃𝑢(𝑋𝑖𝑗,𝜃𝑡−1𝑗)

### Conditional Expectation Functions

For the data-driven approach, we apply the function families [A1] and [C1]:

- **[A1]**: 
  - 𝜔(𝓏) = 𝓏, 
  - 𝜌(𝓏) = −1,
  - 𝜑(𝓏) = z^2 / 2,
  - 𝜓(𝓏) = −𝓏 

- **[C1]**:
  - 𝜔(𝓏) = 𝑎 / (1+𝑒^𝓏) + 𝑏 / (1+𝑒^𝓏),
  - 𝜌(𝓏) = − 𝑒^𝓏 / (1+𝑒^𝓏),
  - 𝜑(𝓏) = (𝑏−𝑎) / (1 + 𝑒^𝓏)+ 𝑏 log(1+𝑒^𝓏),
  - 𝜓(𝓏) = −log(1+𝑒^𝓏)

### Comparison and Results

After generating 1000 random actions with a 50% chance for each action 𝑎𝑡=1 or 𝑎𝑡=2, the states {𝑆1,…,𝑆1001} are created. These sets are used to train the neural networks for approximating 𝜈1(𝑋) and 𝜈2(𝑋) using the respective data sets.

The neural networks used are of a single hidden layer with 100 neurons and ReLU activation functions. The Gradient Descent learning rate is set to 0.001.

The results demonstrate that the neural network approximations converge well to the numerical solution, with the optimal policy derived from the comparison of 𝜈1(𝑆𝑡) and 𝜈2(𝑆𝑡).

### Conclusion

The data-driven approach using neural networks provides good approximations to the optimal functions and action policies, showing that the use of functions [A1] and [C1] successfully estimates the conditional expectations in this reinforcement learning problem.

## Installation

To clone the repository:

        git clone https://github.com/orestis-koutroumpas/Reinforcement-Learning-Optimal-Policy.git

### Dependencies

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib