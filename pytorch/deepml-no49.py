# Implement Adam Optimization Algorithm

import torch
import math

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10) -> torch.Tensor:
    """
    Implements Adam optimization algorithm using PyTorch's built-in optimizer.

    Args:
        f: The objective function to be optimized
        grad: A function that computes the gradient (unused; autograd is used instead)
        x0: Initial parameter values (torch.Tensor)
        learning_rate: The step size (default: 0.001)
        beta1: Exponential decay rate for the first moment estimates (default: 0.9)
        beta2: Exponential decay rate for the second moment estimates (default: 0.999)
        epsilon: A small constant for numerical stability (default: 1e-8)
        num_iterations: Number of iterations to run the optimizer (default: 10)

    Returns:
        torch.Tensor: Optimized parameters
    """
    m_t = 0
    v_t = 0
    for t in range(1,1 + num_iterations):
        gradient = grad(x0)
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * gradient ** 2
        m_hat = m_t / (1 - beta1 ** t)
        v_hat = v_t / (1 -  beta2 ** t)
        x0 -= learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
    return x0
    
    
def main():
    def objective_function(x):
        return x[0]**2 + x[1]**2

    def gradient(x):
        return torch.tensor([2*x[0], 2*x[1]], dtype=torch.float64)

    x0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    x_opt = adam_optimizer(objective_function, gradient, x0)
    print([round(v, 8) for v in x_opt.numpy().tolist()])
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-03 11:22:04
#
