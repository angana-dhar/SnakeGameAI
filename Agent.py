import torch
import random
import numpy as np 
from collections import deque
from game import SnakeGameAI, Direction, Point 
from model import Linear_QNet, QTrainer
from helper import plot 

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # control exploration or randomness
        self.gamma = 0.9 #discounting rate
        self.memory = deque(maxlen=MAX_MEMORY)   #so deque is used here , if the memory is full, it will remove the oldest memory (remove element from the left)(popleft())
        self.model = Linear_QNet(11,256,3) #todo
        self.trainer = QTrainer(self, self.model, lr=LR, gamma=self.gamma) #
        #TODO: model , trainer
        



    def get_state(self,game):
        head = game.snake[0]  #head of the snake
        # 4 points to check the direction of the snake
        point_l = Point(head.x - 20, head.y) #left point
        point_r = Point(head.x + 20, head.y) #right point
        point_u = Point(head.x, head.y - 20) #up point
        point_d = Point(head.x, head.y + 20) #down point


        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN


        food_x = game.food.x
        food_y = game.food.y
        

        state = [
            # 1. danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)), 
            
            #danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),


            #danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or   
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
        

        #move direction 
        dir_l,
        dir_r,
        dir_u,
        dir_d,
    
        #food location
        food_x < head.x , #food is left of the snake head
        food_x > head.x , #food is right of the snake head
        food_y < head.y , #food is above the snake head
        food_y > head.y  #food is below the snake head
        
        ]
        
        return np.array(state,dtype=int)  ##convert to numpy array and convert to int
    # 0 or 1 , so we can use intger
    
        

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #append to the memory    mn

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)  #list of tuple #randomly sample from the memory
        else:
            mini_sample = self.memory

        state, actions, rewards, next_states, dones = zip(*mini_sample) #unzip the mini sample

        self.trainer.train_step(state,actions , rewards , next_states , dones) #train the model using the mini sample

    def train_short_memory(self,state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done) #train the model



    #using for loop 
        # for state , action , reward , next_state , done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        #random moves : tradeoff exploration vs exploitation
        self.epsilon = max(5, 80 - self.n_games)
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) #predict the action using the model
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move #return the action 

#globaal functions
def train():
    plot_scores = []   #initially it will be a emply list
    plot_mean_scores = [] #mean score of the game 
    total_score = 0  #initially it will be 0
    record = 0
    agent = Agent()
    game=SnakeGameAI()
    while True:
        #get old state or current state 
        state_old = agent.get_state(game)


        #get move 
        final_move = agent.get_action(state_old)
        #perform move and get new state
        reward,done,score = game.play_step(final_move)   #reward is the score of the game , done is True or False if the game is over or not and get this from game.py file
        state_new = agent.get_state(game)
        
        
        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember the new state
        agent.remember(state_old,final_move, reward , state_new, done)


        if done:
            #train long memory , plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save() #save the model if the score is greater than the record score

            print('Game', agent.n_games,  'Score', score, 'Reward', reward, 'Record', record)
            
            plot_scores.append(score)
            total_score += score 
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores) #plot the scores and mean scores
            


if __name__=="__main__":
    train()