import torch as tch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        tch.manual_seed(2020)
        self.fc1 = nn.Linear(48, 96)
        self.fc2 = nn.Linear(96, 192)
        self.fc3 = nn.Linear(192, 384)
        self.out = nn.Linear(384, 1)
        self.relu = nn.ReLU()
        self.final = nn.Sigmoid()
    
    def forward(self, x):
        op = self.fc1(x)
        op = self.relu(op)
        op = self.fc2(op)
        op = self.relu(op)
        op = self.fc3(op)
        op = self.relu(op)
        op = self.out(op)
        y = self.final(op)
        return y
    
class NeuralNetworkWithDOL(nn.Module):
    #Adding dropout layers within Neural Network to reduce overfitting
    def __init__(self):
        super().__init__()
        tch.manual_seed(2020)
        self.fc1 = nn.Linear(48, 96)
        self.fc2 = nn.Linear(96, 192)
        self.fc3 = nn.Linear(192, 384)
        self.relu = nn.ReLU()
        self.out = nn.Linear(384, 1)
        self.final = nn.Sigmoid()
        self.drop = nn.Dropout(0.1) #Dropout Layer

    def forward(self, x):
        op = self.drop(x) #Dropout for input layer
        op = self.fc1(op)
        op = self.relu(op)
        op = self.drop(op) #Dropout for hidden layer 1
        op = self.fc2(op)
        op = self.relu(op)
        op = self.drop(op) #Dropout for hidden layer 2
        op = self.fc3(op)
        op = self.relu(op)
        op = self.drop(op) #Dropout for hidden layer 3
        op = self.out(op)
        y = self.final(op)
        return y

class NeuralNetworkL1L2DOL(nn.Module):
    def __init__(self):
        super().__init__()
        tch.manual_seed(2020)
        self.fc1 = nn.Linear(48, 96)
        self.fc2 = nn.Linear(96, 192)
        self.fc3 = nn.Linear(192, 384)
        self.relu = nn.ReLU()
        self.out = nn.Linear(384, 1)
        self.final = nn.Sigmoid()
        self.drop = nn.Dropout(0.1) #Dropout Layer
    
    def forward(self, x):
        op = self.drop(x) #Dropout for input layer
        op = self.fc1(op)
        op = self.relu(op)
        op = self.drop(op) #Dropout for hidden layer 1
        op = self.fc2(op)
        op = self.relu(op)
        op = self.drop(op) #Dropout for hidden layer 2
        op = self.fc3(op)
        op = self.relu(op)
        op = self.drop(op) #Dropout for hidden layer 3
        op = self.out(op)
        y = self.final(op)
        return y