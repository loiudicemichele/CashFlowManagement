import torch.nn as nn

class CashFlowLSTM(nn.Module):
    """
    ### LSTM Model Architecture
    Defining the Recurrent Neural Network using PyTorch's nn.Module.
    The architecture consists of an LSTM layer followed by a fully connected (Linear) layer.
    """
    def __init__(self, 
                 input_dim, # Number of features given to a single LSTM block
                 hidden_dim, # Number of hidden states into the LSTM block
                 num_layers, # Number of LSTM block stacked on each other
                 output_dim, # Number of output of the LSTM
                 dropout = 0.2 # Dropout rate to enhance generalization
                 ):
        super(CashFlowLSTM, self).__init__()
        
        # Inner LSTM block parameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Layer (Readout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Many-to-One architecture, i'm interested in the last output of the network 
        # Extract the output of the last time step for sequence-to-one prediction
        last_time_step_out = lstm_out[:, -1, :] 
        
        # Passing the output through the Linear Layer
        out = self.fc(last_time_step_out)
        return out