#import gym
import math
import random
import numpy as np
import pickle
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
#from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#Deep Q-Network
class DQN(nn.Module):
#     def __init__(self, img_height, img_width):
    # Food coord and snakes coords will be fead into the network as board height and widths are already established
    def __init__(self, board_height, board_width, total_snake_counts):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(in_features=board_width*board_height+total_snake_counts+1, out_features=24)
        # +1 because of turn requires another input node
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=4)

    def forward(self, t):
        #t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        print("fc1",t)
        t = F.relu(self.fc2(t))
        print("fc2",t)
        t = self.out(t)
        print("out",t)
        return t

#Replay Memory
class ReplayMemory():
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experiece):
        if len(self.memory) < self.capacity:
            self.memory.append(experiece)
        else:
            self.memory[self.push_count % self.capacity] = experiece
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

#Epsilon Greedy Strategy
class EpsilonGreedyStrategy():
    def __init__(self,start,end,decay):
        #Start, end and decay values
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end)*math.exp(-1*current_step*self.decay)

#Reinforcement Learning Agent
class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            print("Exploring")
            return random.randrange(self.num_actions) #explore
        else:
            with torch.no_grad():
                print("Exploiting")
                return policy_net(state).argmax(dim=1).item() #exploit

#Q-Value Calculator
#Static methods means that we can call those mehtods without creating
# an instance of the class first

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def action_tuple_to_tensor(self,action):
        """
        Converts the action from a tuple to a tensor
        Args: action(tuple)
        Return: action(tensor)
        """
        action_tensor = torch.tensor([1,0,0,0])
        print("222",action_tensor[0])
        return

    @staticmethod
    def get_current(policy_net, states, actions):
        # print(policy_net(states[0]))
        action_tensor = torch.tensor([0,0,0,0])
        print("action_tensor",action_tensor)
        print("actions",actions)
        action_tensor[actions] = actions
        return policy_net(states).gather(dim=0,index=action_tensor)
#     def get_current(policy_net, states, actions):
#         return policy_net(states).gather(dim=0,index=actions)

    @staticmethod
    def get_next(target_net, next_states, actions):
        #final_state_locations = next_states.flatten(start_dim=1)
        action_tensor = torch.tensor([0,0,0,0])
        action_tensor[actions] = actions
        return target_net(next_states).gather(dim=0,index=action_tensor)

#         final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
#         non_final_state_locations = (final_state_locations == False)
#         batch_size = next_states.shape[0]
#         values = torch.zeros(batch_size).to(QValues.device)
#         values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

#Create input to feed into policy net

class SnakeEnv():
    #input_data = torch.zeros([11, 11], dtype=torch.float32)
    #food = data_json['board']['food']
    # def __init__(self,board_height, board_width, my_snake_health, total_snake_counts, food):
    #     self.board_width = board_height
    #     self.board_width = board_width


    def get_snakes_health(self, board_info):
        #Enemy snakes health (This may include my snake's health)
        enemy_snake_health = []
        enemy_snake_health.append(board_info['board']['snakes'][0]['health'])

        #My snake health
        my_snake_health = []
        my_snake_health.append(board_info['you']['health'])

        snakes_health = []
        snakes_health.append(enemy_snake_health)
        snakes_health.append(my_snake_health)
        snakes_health = torch.flatten(torch.tensor(snakes_health, dtype=torch.float32))

        return snakes_health

    def find_other_snakes(self, board_info):
        snakes_coords = []
        #Enemy snakes coords
        for i in board_info['board']['snakes']:
            snake_x = i['body'][0]['x']
            snake_y = i['body'][0]['y']
            snake_coords = i['body'][0]
            snakes_coords.append(snake_coords)

        #My snake coords
        my_snake_coords = board_info['you']['body'][0]
        snakes_coords.append(my_snake_coords)
        return snakes_coords

    def my_snake_coords(self, board_info):
        """
        Finds my snake's coordinates (head)
        Args:
            board_info (Tensor): Board containing all environment info

        Returns:
            my_snake_coords (Tensor): Coordinates of my snake's head
        """
        my_snake_coords = board_info['you']['body'][0]
        return my_snake_coords

    def input_for_policy_net(self, input_data, snakes_coords, food_coords, snakes_health):
        """
        Creates a map of the inputs and flattens the tensor to feed into a DQN
        Args:
            input_data (Tensor): Empty board input tensor
            snakes_coords (List): A list of dictionary of coordinates
            food_coords (List): Coordinates of food
            snakes_health (Tensor): A tensor of snakes healthes

        Returns:
            input_data (Tensor): Flattened tensor containing board information
        """
        input_data[food_coords[0]['x']-1][food_coords[0]['y']-1] = 10 #10 is the reward of where the food is
        for i in snakes_coords:
            input_data[i['x']-1][i['y']-1] = -1 #-1 denotes bad locations to move to
        input_data = torch.flatten(input_data)
        input_data = torch.cat((input_data,snakes_health),0)
        return input_data

    def next_state(self, current_state, action):
        """
        Since I don't know what the next states of other snakes will be at, memory batch must contain consecutive
        snake moves. Use one for policy net (first_state) and the other for target net (second_state).
        """
        #Probably just endup using the input_for_policy_net
        return

    def reward_map(self, new_board, snakes_coords, food_coords):
        """
        Creates a mapping of reward based on the map
        Args:
            new_board (Tensor): Empty board input tensor
            snakes_coords (List): A list of dictionary of coordinates

        Returns:
            board_info (Tensor): A reward map
        """
        #Need to name sure all coords where other snakes can head to will be -1 too.

        NEGATIVE_REWARD = -1
        POSITIVE_REWARD = 10
        for snake_coords in snakes_coords:
            new_board[snake_coords['x']-1][snake_coords['y']-1] = NEGATIVE_REWARD

        new_board[food_coords[0]['x']-1][food_coords[0]['y']-1] = POSITIVE_REWARD

        return new_board

    def reward(self, action, reward_map, my_snake_coord):
        """
        Gets the reward value of the action taken.
        Args:
            action (int): 0 = Up, 1 = Right, 2 = Down, 3 = Left
            reward_map (Tensor): Mapping of rewards
            my_snake_coord (Tensor): Head coordinate of my snake
        """
        if action == 0:
            direction = torch.tensor([0,1])
        elif action == 1:
            direction = torch.tensor([1,0])
        elif action == 2:
            direction = torch.tensor([0,-1])
        else:
            direction = torch.tensor([-1,0])

        reward = reward_map[my_snake_coord['x'] + direction[0]][my_snake_coord['y'] + direction[1]]

        return reward

#Tensor processing

def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    print(batch)

    t1 = batch.state
    t2 = batch.action
    t3 = batch.reward
    t4 = batch.next_state
    # t1 = torch.tensor(batch.state)
    # t2 = torch.tensor(batch.action)
    # t3 = torch.tensor(batch.reward)
    # t4 = torch.tensor(batch.next_state)

    return (t1,t2,t3,t4)

#Experience class
Experience = namedtuple(
    'Experience',
    ('state','action','next_state','reward')
)

#Main Program
#data_json = data
#data_json = data[0][0] #Only the initial state o set up the required parameters, ie board_width
def run_game(game_data):
    data_json = game_data
    batch_size = 30
    gamma = 0.999
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 10 #How often the target network will update from policy network's weights
    memory_size = 1000
    lr = 0.001
    num_episodes = 1
    num_actions_available = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategy = EpsilonGreedyStrategy(eps_start,eps_end, eps_decay)
    agent = Agent(strategy,num_actions_available,device)
    memory = ReplayMemory(memory_size)
    snake_env = SnakeEnv()

    #DQN input data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board_height = data_json['board']['height']
    board_width = data_json['board']['width']
    turn = data_json['turn']
    my_snake_health = data_json['you']['name']
    total_snakes_count = len(data_json['board']['snakes']) + 1
    food = data_json['board']['food']

    policy_net = DQN(board_height, board_width, total_snakes_count).to(device)
    target_net = DQN(board_height, board_width, total_snakes_count).to(device)
    target_net.load_state_dict(policy_net.state_dict()) #Uploads the weights of the policy net to the target net
    target_net.eval() #Target net not in training mode
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    #r = json.dumps(data)
    #data_json = json.loads(r)
    episode_durations = []
    count = 0
    previous_state = 0
    #Where loop begins
    for episode in range(num_episodes):
        count += 1
        #data_json = json.loads(r) #Loads the initial state
        input_data = torch.zeros([11, 11], dtype=torch.float32)
        my_snake_coords = snake_env.my_snake_coords(data_json)
        snakes_coords = snake_env.find_other_snakes(data_json)
        snakes_health = snake_env.get_snakes_health(data_json)
        state = snake_env.input_for_policy_net(input_data,snakes_coords,food,snakes_health)
        episode_ended = False

        while episode_ended == False:
            # count = 0
            action = agent.select_action(state,policy_net) #Execute selected action
            print("action",action)
            #Observe reward and next state
            next_state = None #Use a get request to receive new state.
            reward_map = snake_env.reward_map(input_data, snakes_coords, food)
            reward = snake_env.reward(action, reward_map, my_snake_coords)
            # print("reward: ", reward)
            #Both action and reward are sigle numbers where as state and next_state are tensors so should convert all to
            #tensors
            # pickle.dump(previous_state,open("previous.p","wb"))
            # pickle.dump(memory.memory,open("save.p", "wb"))

            #This part stores the states
            previous_state = pickle.load(open("previous.p","rb"))
            memory.memory = pickle.load(open("save.p", "rb"))
            if (len(memory.memory) < 1):
                previous_state = state

            if (len(previous_state) > 1):
                previous_next_state = state
                memory.push(Experience(previous_state, action,previous_next_state, reward))
                previous_state = state

            # for individual_experience in range(len(experiences)):
    #         if memory.can_provide_sample(batch_size):
    #             experiences = memory.sample(batch_size)
    #             states, actions, rewards, next_state = extract_tensors(experiences)
    #
    #             current_q_values = QValues.get_current(policy_net, states[individual_experience], actions[individual_experience])
    #             next_q_values = QValues.get_next(target_net, next_states[individual_experience], actions[individual_experience])
    #             # target_q_values = (next_q_values * gamma) + rewards
    #             # target_q_values = (torch.max(next_q_values).item() * gamma) + rewards[individual_experience]
    #             target_q_values = next_q_values*gamma
    #             target_q_values[actions[individual_experience]] + rewards[individual_experience]
    #
    #             loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step() #updates the weights and biases
    #
    #         if episode_ended:
    #             episode_durations.append(episode)
    # #             plot(episode_durations, 100)
    # #             break

            # if count > batch_size:
            episode_ended = True

            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            print("count",count)

    pickle.dump(previous_state,open("previous.p","wb"))
    pickle.dump(memory.memory,open("save.p", "wb"))
    print("memory: ",memory.memory)
    print(len(memory.memory),count)

    return action
    #         if reward == -1:
    #             episode ended = True

def training(memory):

    batch_size = 30
    gamma = 0.999
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 10 #How often the target network will update from policy network's weights
    memory_size = 1000
    lr = 0.001
    num_episodes = 1
    num_actions_available = 4
    board_width = 11
    board_height = 11
    total_snakes_count = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(board_height, board_width, total_snakes_count).to(device)
    target_net = DQN(board_height, board_width, total_snakes_count).to(device)
    target_net.load_state_dict(policy_net.state_dict()) #Uploads the weights of the policy net to the target net
    target_net.eval() #Target net not in training mode
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    experiences = memory.sample(batch_size)
    states, actions, rewards, next_states = extract_tensors(experiences)
    for individual_experience in range(len(experiences)):

        current_q_values = QValues.get_current(policy_net, states[individual_experience], actions[individual_experience])
        next_q_values = QValues.get_next(target_net, next_states[individual_experience], actions[individual_experience])
        # target_q_values = (torch.max(next_q_values).item() * gamma) + rewards[individual_experience]
        target_q_values = next_q_values*gamma
        target_q_values[actions[individual_experience]] + rewards[individual_experience]

        print("cuurentQ",current_q_values)
        print("targetQ",target_q_values)
        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #updates the weights and biases
    print("fc1 param",list(policy_net.fc1.weight))
    print("fc2 param",list(policy_net.fc2.weight))
    print("out param",list(policy_net.out.weight))
    return

if __name__ == '__main__':
    print("ok")
    memory = ReplayMemory(30)
    memory.memory = pickle.load(open("save.p", "rb"))
    training(memory)
