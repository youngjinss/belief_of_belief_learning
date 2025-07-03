import torch
import torch.nn as nn

"""
@Author Filip Borowiak 
"""


class LSTM(nn.Module):
  def __init__(self, input_size: int, hidden_size: int):
    super(LSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size=hidden_size
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

  def forward(self, x):
    # x.shape -> (seq_len, batch_size, input_size)
    h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device) # initial hidden state
    c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device) # initial cell state
    out, _ = self.lstm(x, (h0, c0))

    return out[:, -1, :] # out shape -> (seq_len, batch_size, hidden_size)