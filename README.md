# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/user-attachments/assets/d41c9417-237a-4d9c-a810-20a47d6c86e4)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: K SANTHAN KUMAR
### Register Number: 212223240065
```python
class NeuralNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.n1=nn.Linear(1,10)
        self.n2=nn.Linear(10,20)
        self.n3=nn.Linear(20,30)
        self.n4=nn.Linear(30,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
    
    def forward(self,x):
        x=self.relu(self.n1(x))
        x=self.relu(self.n2(x))
        x=self.relu(self.n3(x))
        x=self.n4(x)
        return x

# Initialize the Model, Loss Function, and Optimizer

my_nn=NeuralNet()
criteria=nn.MSELoss()
optimizer=optim.RMSprop(my_nn.parameters(),lr=0.001)


def train_model(my_nn,X_train,y_train,criteria,optmizer,epochs=3000):
    for i in range(epochs):
        optimizer.zero_grad()
        loss=criteria(my_nn(X_train),y_train)
        loss.backward()
        optimizer.step()
        
        my_nn.history['loss'].append(loss.item())
        if i%200==0:
            print(f"Epoch [{i}/epochs], loss: {loss.item():.6f}")

```
## Dataset Information

![image](https://github.com/user-attachments/assets/9689f57b-1267-46f3-bc04-2c3d5df837b9)


## OUTPUT

![image](https://github.com/user-attachments/assets/211b71ea-4499-439f-97e6-1468ce7d0bf1)

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/b5711f72-80a2-4cb7-973d-ae3e34bdd866)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/490f3da4-efb0-49ec-abcb-fac14ad59b3c)

## RESULT

Successfully executed the code to develop a neural network regression model.
