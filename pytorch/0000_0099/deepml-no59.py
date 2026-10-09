# problem: Implement Long Short-Term Memory (LSTM) Network

import torch


class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases as float64 tensors
        self.Wf = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wi = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wc = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wo = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)

        self.bf = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bi = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bc = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bo = torch.zeros(hidden_size, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor, initial_hidden_state: torch.Tensor, initial_cell_state: torch.Tensor):
        """
        Processes a sequence of inputs and returns the hidden states,
        final hidden state, and final cell state.

        Args:
            x: Input tensor of shape (seq_len, input_size)
            initial_hidden_state: Initial hidden state of shape (hidden_size, 1)
            initial_cell_state: Initial cell state of shape (hidden_size, 1)

        Returns:
            outputs: Tensor of hidden states at each time step
            h: Final hidden state tensor
            c: Final cell state tensor
        """
        seq_len, input_size = x.shape
        ht = initial_hidden_state.clone()
        ct = initial_cell_state.clone()
        hts = []
        for xt in x:
            xt = xt.reshape(input_size, 1)
            combined = torch.cat([ht, xt], dim=0)
            ft = torch.sigmoid(self.Wf @ combined + self.bf)
            it = torch.sigmoid(self.Wi @ combined + self.bi)
            c_tilde = torch.tanh(self.Wc @ combined + self.bc)
            ct = ft * ct + it * c_tilde
            ot = torch.sigmoid(self.Wo @ combined + self.bo)
            ht = ot * torch.tanh(ct)
            hts.append(ht)
        return torch.stack(hts,dim=0),ht,ct


def main():
    input_sequence = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    initial_hidden_state = torch.zeros((1, 1), dtype=torch.float64)
    initial_cell_state = torch.zeros((1, 1), dtype=torch.float64)

    lstm = LSTM(input_size=1, hidden_size=1)
    outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

    print(final_h)


if __name__ == "__main__":
    main()

#
# Created By jing At 2026-09-05 14:02:16
#
