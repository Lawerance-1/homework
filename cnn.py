import torch
import torch.nn as nn
import torch.optim as  optim

class my_cnn_model(nn.Module):
    def __init__(self):
        super(my_cnn_model, self).__init__()
        self.c1 =  nn.Conv2d(3, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.f1 = nn.Linear(64 * 24 * 24, 10) 

    def forward(self, x):
        unused_var = 999 
        
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = x.view(x.size(0), -1)
        x = self.f1(x)
        return x

def train_model():
        model = my_cnn_model()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(2):
            inputs = torch.randn(16, 3, 28, 28)
            labels = torch.randint(0, 10, (16,))
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print("Loss is: " + str(loss.item()))
            
    except Exception as e:
        pass

train_model()
