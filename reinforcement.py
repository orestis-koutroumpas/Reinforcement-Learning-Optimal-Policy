import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def H1(y, x):
    return norm.cdf(y - 0.8 * x - 1, loc=0, scale=1)

def H2(y, x):
    return norm.cdf(y + 2, loc=0, scale=1)

def reward(S):
    return np.minimum(2, S**2)

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# ReLU Derivative
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

### [A1]
# ω(z)
def omega_A1(z):
    return z
def rho_A1(z):
    return -1
def phi_A1(z):
    return z**2 / 2
def psi_A1(z):
    return -z

### [C1]
def omega_C1(z, a=0, b=2):
    return a / (1 + np.exp(z)) + b * np.exp(z) / (1 + np.exp(z))
def rho_C1(z):
    return - np.exp(z) / (1 + np.exp(z))
def phi_C1(z, a=0, b=2):
    return (b-a) / (1 + np.exp(z)) + b * np.log(1 + np.exp(z))
def psi_C1(z):
    return -np.log(1 + np.exp(z))

# Cost function
def J(u, Y, d_func, phi_func, psi_func):
    return np.mean(phi_func(u) + d_func(Y) * psi_func(u))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.A1 = np.random.normal(0, np.sqrt(1 / input_size), (hidden_size, input_size))
        self.B1 = np.zeros((hidden_size, 1))
        self.A2 = np.random.normal(0, np.sqrt(1 / hidden_size), (output_size, hidden_size))
        self.B2 = np.zeros((output_size, 1))
        self.learning_rate = learning_rate

        # Initialize ADAM parameters
        self.m_A1 = np.zeros_like(self.A1)
        self.v_A1 = np.zeros_like(self.A1)
        self.m_B1 = np.zeros_like(self.B1)
        self.v_B1 = np.zeros_like(self.B1)

        self.m_A2 = np.zeros_like(self.A2)
        self.v_A2 = np.zeros_like(self.A2)
        self.m_B2 = np.zeros_like(self.B2)
        self.v_B2 = np.zeros_like(self.B2)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step for ADAM

    def forward(self, X):
        self.Z1 = relu(np.dot(self.A1, X) + self.B1)
        self.Y = np.dot(self.A2, self.Z1) + self.B2
        return self.Y

    def backward(self, X, Y, d_func, omega_func, rho_func):
        # Compute forward pass
        u = self.forward(X)

        # Compute gradient components
        d_loss = d_func(Y) - omega_func(u)
        v2 = d_loss * rho_func(u)
        
        # Compute gradients for output layer
        d_A2 = np.dot(v2, self.Z1.T) / X.shape[1]
        d_B2 = np.mean(v2, axis=1, keepdims=True)

        # Compute gradients for hidden layer
        v1 = np.dot(self.A2.T, v2) * relu_derivative(np.dot(self.A1, X) + self.B1)
        d_A1 = np.dot(v1, X.T) / X.shape[1]
        d_B1 = np.mean(v1, axis=1, keepdims=True)

        # Update time step
        self.t += 1

        # Update weights and biases using ADAM
        for param, grad, m, v in [
            (self.A1, d_A1, self.m_A1, self.v_A1),
            (self.B1, d_B1, self.m_B1, self.v_B1),
            (self.A2, d_A2, self.m_A2, self.v_A2),
            (self.B2, d_B2, self.m_B2, self.v_B2)
        ]:
            m[:] = self.beta1 * m + (1 - self.beta1) * grad  
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # Correct bias for moments
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def train(self, X, Y, epochs, d_func, omega_func, rho_func, phi_func, psi_func, g_name, functions_name):
        losses = []
        print(f"\nTraining Neural Network with {functions_name}.")
        print("=" * 50)

        for epoch in range(epochs):
            self.backward(X, Y, d_func, omega_func, rho_func)
            loss = J(self.forward(X), Y, d_func, phi_func, psi_func)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        # Plot learning curve
        plt.plot(losses)
        plt.xlabel("Number of Iterations")
        plt.title(f"Learning Curve for {g_name} with {functions_name}")
        plt.show()

if __name__ == "__main__":
    ### Numerical solution
    N=1000
    St = np.linspace(-10, 10, N)
    Wt = np.random.normal(0, 1, (1, N))

    F1 = np.zeros((N, N))
    F2 = np.zeros((N, N))
    for j in range(N):
        F1[j][0] = 0.5 * (H1(St[1], St[j]) - H1(St[0], St[j]))
        F1[j][-1] = 0.5 * (H1(St[-1], St[j]) - H1(St[-2], St[j]))

        F2[j][0] = 0.5 * (H2(St[1], St[j]) - H2(St[0], St[j]))
        F2[j][-1] = 0.5 * (H2(St[-1], St[j]) - H2(St[-2], St[j]))

        for i in range(2, N):
            F1[j][i - 1] = 0.5 * (H1(St[i], St[j]) - H1(St[i - 2], St[j]))
            F2[j][i - 1] = 0.5 * (H2(St[i], St[j]) - H2(St[i - 2], St[j]))

    R = np.array([reward(s) for s in St])
    V1 = np.dot(F1, R)
    V2 = np.dot(F2, R)

    plt.figure(figsize=(10, 6))
    plt.plot(St, R, label="Reward", color="grey")
    plt.xlim(St.min(), St.max())
    plt.ylim(-1, 3)
    plt.title(r"$Reward\ R(S)=\min \{2,S^2\}$")
    plt.xlabel(r'$\mathcal{S}$', fontsize=16)
    plt.ylabel(r'$R(S)$', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Generate actions
    actions = np.zeros(N)
    for i in range(N):
        actions[i] = 1 if np.random.uniform(0, 1) > 0.5 else 2
    
    # Generate States
    States = np.zeros(N+1)
    States[0] = np.random.normal(0, 1)

    for t in range(N):
        Wt = np.random.normal(0, 1)
        if actions[t] == 1:
            States[t+1] = 0.8 * States[t] + 1.0 + Wt
        else:
            States[t+1] = -2.0 + Wt

    # Separate the data into two sets based on the action
    set_action_1 = [(States[t], States[t + 1]) for t in range(N) if actions[t] == 1]
    set_action_2 = [(States[t], States[t + 1]) for t in range(N) if actions[t] == 2]

    # Convert to numpy arrays
    set_action_1 = np.array(set_action_1)
    set_action_2 = np.array(set_action_2)
    
    # Training data for v1(S)
    X1 = set_action_1[:, 0].reshape(1, -1)
    Y1 = set_action_1[:, 1].reshape(1, -1)
    # Training data for v2(S)
    X2 = set_action_2[:, 0].reshape(1, -1)
    Y2 = set_action_2[:, 1].reshape(1, -1)

    # Neural Network parameters
    input_size = 1
    hidden_size = 100
    output_size = 1
    learning_rate = 0.001
    epochs = 2000

    # Create and train the neural networks

    ### V1(S)
    nn_V1_A1 = NeuralNetwork(input_size, hidden_size, output_size, learning_rate) 
    nn_V1_A1.train(X1, Y1, epochs, d_func=reward, omega_func=omega_A1, rho_func=rho_A1, phi_func=phi_A1, psi_func=psi_A1, g_name=r"$u(\mathcal{X},θ^1_o)$", functions_name="[A1]")
    nn_V1_A1_pred = np.array(omega_A1([nn_V1_A1.forward(np.array([[x]])) for x in St]))

    nn_V1_C1 = NeuralNetwork(input_size, hidden_size, output_size, learning_rate) 
    nn_V1_C1.train(X1, Y1, epochs, d_func=reward, omega_func=omega_C1, rho_func=rho_C1, phi_func=phi_C1, psi_func=psi_C1, g_name=r"$u(\mathcal{X},θ^1_o)$", functions_name="[C1]")
    nn_V1_C1_pred = np.array(omega_C1([nn_V1_C1.forward(np.array([[x]])) for x in St]))

    plt.figure(figsize=(10, 6))
    plt.plot(St, V1, label="Numerical", color="black")
    plt.plot(St, nn_V1_A1_pred.flatten(), label="[A1]", color="blue")
    plt.plot(St, nn_V1_C1_pred.flatten(), label= "[C1]", color="red")
    plt.xlim(-6, 4)
    plt.ylim(0, 3)
    plt.title(r"$Conditional\ Expectation\ v_1(X)$")
    plt.xlabel(r'$\mathcal{X}$', fontsize=16)
    plt.ylabel(r'$\mathbb{E}^1_{S_{t+1}} \left[R(S_{t+1}) \mid S_t = X \right]$', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

    ### V2(S)
    nn_V2_A1 = NeuralNetwork(input_size, hidden_size, output_size, learning_rate) 
    nn_V2_A1.train(X2, Y2, epochs, d_func=reward, omega_func=omega_A1, rho_func=rho_A1, phi_func=phi_A1, psi_func=psi_A1, g_name=r"$u(\mathcal{X},θ^2_o)$", functions_name="[A1]")
    nn_V2_A1_pred = np.array(omega_A1([nn_V2_A1.forward(np.array([[x]])) for x in St]))

    nn_V2_C1 = NeuralNetwork(input_size, hidden_size, output_size, learning_rate) 
    nn_V2_C1.train(X2, Y2, epochs, d_func=reward, omega_func=omega_C1, rho_func=rho_C1, phi_func=phi_C1, psi_func=psi_C1, g_name=r"$u(\mathcal{X},θ^2_o)$", functions_name="[C1]")
    nn_V2_C1_pred = np.array(omega_C1([nn_V2_C1.forward(np.array([[x]])) for x in St]))

    plt.figure(figsize=(10, 6))
    plt.plot(St, V2, label="Numerical", color="black")
    plt.plot(St, nn_V2_A1_pred.flatten(), label="[A1]", color="blue")
    plt.plot(St, nn_V2_C1_pred.flatten(), label= "[C1]", color="red")
    plt.xlim(-5, 5)
    plt.ylim(0, 3)
    plt.title(r"$Conditional\ Expectation\ v_2(X)$")
    plt.xlabel(r'$\mathcal{X}$', fontsize=16)
    plt.ylabel(r'$\mathbb{E}^2_{S_{t+1}} \left[R(S_{t+1}) \mid S_t = X \right]$', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot V1(S) and V2(S)
    plt.figure(figsize=(10, 6))
    plt.plot(St, V1, label="Numerical-1", color="black")
    plt.plot(St, V2, label="Numerical-2", color="grey")
    plt.plot(St, nn_V1_A1_pred.flatten(), label="[A1]-1", color="blue")
    plt.plot(St, nn_V1_C1_pred.flatten(), label="[C1]-1", color="green")
    plt.plot(St, nn_V2_A1_pred.flatten(), label="[A1]-2", color="red")
    plt.plot(St, nn_V2_C1_pred.flatten(), label="[C1]-2", color="orange")
    plt.xlim(-6, 4)
    plt.ylim(0, 3)
    plt.title(r"$Conditional\ Expectations\ v_1(X),\ v_2(X)$")
    plt.xlabel(r'$\mathcal{S}$', fontsize=16)
    plt.ylabel(r'$\mathbb{E}^j_{S_{t+1}} \left[ R(S_{t+1}) \mid S_t = X \right]$', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optimal Actions
    opt_actions = np.zeros(N)
    A1_actions = np.zeros(N)
    C1_actions = np.zeros(N)

    for t in range(N):
        opt_actions[t] = 1 if V1[t] > V2[t] else 2
        A1_actions[t] = 1 if omega_A1(nn_V1_A1.forward(St[t])) > omega_A1(nn_V2_A1.forward(St[t])) else 2
        C1_actions[t] = 1 if omega_C1(nn_V1_C1.forward(St[t])) > omega_C1(nn_V2_C1.forward(St[t])) else 2

    plt.figure(figsize=(10, 6))
    plt.plot(St, opt_actions, label="Optimal")
    plt.plot(St, A1_actions, label="[A1]")
    plt.plot(St, C1_actions, label="[C1]")
    plt.xlim(St.min(), St.max())
    plt.ylim(0, 3)
    plt.xlabel(r'$\mathcal{S}$', fontsize=16)
    plt.ylabel(r"$Action\ Policy\ \{a_t\}$")
    plt.title("Optimal and Approximately optimal action policy")
    plt.legend()
    plt.grid(True)
    plt.show()