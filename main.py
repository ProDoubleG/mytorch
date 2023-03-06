# %%
import numpy
from mytorch.utils.data import DataLoader
import mytorch as torch
import mytorch.nn as nn
import mytorch.nn.functional as F
# %%
data = numpy.random.randn(100,28,28,1)
label = numpy.arange(100).reshape(100,1)

trainset = (data,label)

trainloader = DataLoader(trainset, batch_size=10)
# %%
class DNN(nn.Model):
    def __init__(self):
        super().__init__()
        self.layer_1_1 = nn.Conv2d(5,4,4)
        self.layer_1_2 = nn.Linear(7*7*5,10)
        
    def forward(self, x):

        x1 = self.layer_1_1(x) # 5 7 7 
        x2 = torch.flatten(x1,1) # 1,245
        x3 = self.layer_1_2(x2) # 245 -> 10,245 -> 10
        x4 = F.Softmax(x3) # 10
        return x4
# %%
a = DNN()
# %%
pred = a(data) # (batch_size, prediction) # initialization

criterion = torch.nn.CrossEntropyLoss()
loss = criterion(pred,label) # how loss is computed

loss.backward() # calculate gradient for each layer

# optimizer.step() # apply calculated gradient here
# optimizer.zero_grad() # clear gradients

# total_loss += loss.item()

# with torch.no_grad():
#     pred = m1(x_train)
#     acc = pred.data.max(1)[1].eq(y_train.data).sum()/len(x_train) * 100
#     loss = criterion(pred, y_train)
# %%
