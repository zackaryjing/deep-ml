# Implementing a Simple RNN

import torch

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> torch.Tensor:
    """
    Implements a simple RNN cell forward pass using PyTorch.

    Args:
        input_sequence: List of input vectors for each time step.
        initial_hidden_state: The initial hidden state vector.
        Wx: Weight matrix for input-to-hidden connections.
        Wh: Weight matrix for hidden-to-hidden connections.
        b: Bias vector.

    Returns:
        torch.Tensor: The final hidden state after processing the entire sequence,
                      rounded to four decimal places.
    """
    input_sequence = torch.tensor(input_sequence,dtype=torch.float)
    initial_hidden_state = torch.tensor(initial_hidden_state,dtype=torch.float)
    Wx = torch.tensor(Wx,dtype=torch.float)
    Wh = torch.tensor(Wh,dtype=torch.float)
    b = torch.tensor(b,dtype=torch.float)
    seq_len = len(input_sequence)
    h_t = initial_hidden_state
    for i in range(seq_len):
        h_t = torch.tanh(Wx @ input_sequence[i] + Wh @ h_t + b)
    return h_t
    

def main():
    result = rnn_forward([[1.0], [2.0], [3.0]], [0.0], [[0.5]], [[0.8]], [0.0])
    print(result.numpy().tolist())
    

if __name__ == "__main__":
    main()


#
# Created By jing At 2026-09-04 14:38:27
#
