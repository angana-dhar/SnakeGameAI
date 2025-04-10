import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 

class Linear_QNet(nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) #input layer to hidden layer
        self.linear2 = nn.Linear(hidden_size, output_size) #hidden layer to output layer
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    

    def save(self, file_name='model.pth'):
        model_folder = './model'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        file_path = os.path.join(model_folder, file_name)
        torch.save(self.state_dict(), file_name)



class QTrainer:
    def __init__(self, agent ,model, lr, gamma):
        self.agent = agent
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    def train_step(self, state, action, reward, next_state, done):
        # convert to tensor
        state = torch.tensor(state, dtype=torch.float) # convert to tensor``
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long) # convert to tensor
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x) shape
            state = torch.unsqueeze(state, 0) # add batch dimension
            next_state = torch.unsqueeze(next_state, 0) #unsqueeze the tensor to add batch dimension
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
       
       
       
        # 1: predicted Q values with current state

        # Q value from the model
        pred = self.model(state)


        #2: Q_new = r + y * max(next_predicted Q value) #next predicted Q value
        #pred.clone()
        #preds[argmax(action).item()] = Q_new

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        #2: Q_new = r + y * max(next_predicted Q value) #next predicted Q value
        #pred.clone()
        #preds[argmax(action).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) #target is Qnew and pred is Q value from the model
        loss.backward()


        self.optimizer.step() #update the model using the optimizer