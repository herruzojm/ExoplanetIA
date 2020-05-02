import torch

# Definicion del modelo para red LSTM
class ModeloLSTM(torch.nn.Module):
    # Por defecto, tendremos cuatro capas de redes LSTM
    # y una capa densa final
    def __init__(self, input_size = 3197, hidden_size = 100, output_size = 2, layers = 4, dropout = 0.2):
        super(ModeloLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, layers, dropout = dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, hidden = self.lstm(x)
        out = out.view(-1, self.hidden_size)
        return self.fc(out)