from fileinput import filename
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name = "best_instance_dqn.pth"):
        model_folder_path = "./neural-net"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

        # Made to be the same as NEAT sacing method
        # best_instance = {
        #     'net': self.state_dict
        # }
        # with open(file_name, 'wb') as output:
        #     pickle.dump(best_instance, output, pickle.HIGHEST_PROTOCOL)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            #(1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        #1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for ind in range(len(game_over)):
            q_new = reward[ind]
            if not game_over[ind]:
                q_new = reward[ind] + self.gamma * torch.max(self.model(next_state[ind]))
            target[ind][torch.argmax(action).item()] = q_new

        #2: r + y * max(next_predicted q val)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
