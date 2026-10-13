
# Implement a Simple RNN with Backpropagation Through Time (BPTT)

import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initializes the RNN with random weights and zero biases using PyTorch modules.
        """
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size, nonlinearity='tanh')
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.optimizer = None
        # self.hidden_state = []
        
        nn.init.normal_(self.rnn_cell.weight_ih, 0, 0.01)
        nn.init.normal_(self.rnn_cell.weight_hh, 0, 0.01)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        
        nn.init.zeros_(self.rnn_cell.bias_ih)
        nn.init.zeros_(self.rnn_cell.bias_hh)
        nn.init.zeros_(self.fc.bias)
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RNN for a given sequence of inputs.
        """
        h_t = x.new_zeros(self.hidden_size)
        outputs = []
        for x_t in x:
            h_t = self.rnn_cell(x_t,h_t)
            y_t = self.fc(h_t)
            outputs.append(y_t)
        return torch.stack(outputs)
        

    def backward(self, x: torch.Tensor, y: torch.Tensor, learning_rate: float) -> None:
        """
        Backpropagation through time to adjust weights based on error gradient.
        """
        self.zero_grad()
        
        y_pred = self.forward(x)
        loss = 0.5 * ((y_pred - y) ** 2).sum()
        loss.backward()
        
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    p -= learning_rate * p.grad


def main():
    torch.manual_seed(42)

    input_sequence = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    expected_output = torch.tensor([[2.0], [3.0], [4.0], [5.0]], dtype=torch.float32)

    rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1)

    for epoch in range(100):
        rnn.backward(input_sequence, expected_output, learning_rate=0.01)

    output = rnn.forward(input_sequence)
    print(output.detach().numpy().tolist())
    
if __name__ == "__main__":
    main()


#
# Created By jing At 2026-10-11 13:06:16
#

