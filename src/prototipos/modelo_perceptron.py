import torch

# Definicion del modelo de perceptron
class Perceptron(torch.nn.Module):
    #definimos la estructura de la red
    #capa de entrada x con activacion relu
    #capa oculta con activacion relu
    def __init__(self, output = 2):
        super(Perceptron, self).__init__()
        self.fc1 = torch.nn.Linear(3197, 1000)
        self.fc2 = torch.nn.Linear(1000, 250)
        self.fc3 = torch.nn.Linear(250, 20)
        self.fc4 = torch.nn.Linear(20, output)
        self.relu = torch.nn.ReLU()
        
    #definimos el flujo 
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x