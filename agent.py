LR = 0.01

import os
import time
import numpy as np
import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from snake import Game
import keras
from multiprocessing import Process, freeze_support



def CreateModel(input_shape, num_actions):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, input_shape=input_shape, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(num_actions, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model


def sigmoid_exploration(snake_length, move_count, k=0.01, C=20):
    # Calculate sigmoid value based on snake length and move count
    x = snake_length
    y = move_count
    sigmoid_value = 1 / (1 + np.exp(k * (x + y - C)))  # Notice the positive k
    return sigmoid_value



def train(model, episodes, gamma, epsilon, snakeSize=3, silent=True):
    # Accumulators for batch training
    state_batch = []
    target_f_batch = []

    for episode in range(episodes):
        game = Game(5, snakeSize)
        state = game.GetState()
        state = np.array(state).reshape(1, 14)
        done = False
        averageReward = 0
        while not done:
            if(game.GetScore(snakeSize) <= -250):
                break

            possible_moves = game.GetMoves()

            if np.random.rand() < epsilon:
                action = np.random.randint(0, len(possible_moves))
            else:
                action = np.argmax(model.predict(state))
                if action >= len(possible_moves):
                    action = np.random.randint(0, len(possible_moves))

            move = possible_moves[action]
            next_state, reward, done = game.Play(move[0], move[1], defaultSnakeSize=snakeSize)
            next_state = np.array(next_state).reshape(1, 14)

            # Calculate target for the Q value
            target = reward + gamma * np.max(model.predict(next_state))
            target_f = model.predict(state).copy()
            target_f[0][action] = target

            # Store experience in the batch
            state_batch.append(state[0])
            target_f_batch.append(target_f[0])

            

            state = next_state
            if (reward < -1000):
                print("Bad game", episode, reward, game.GetScore(snakeSize))


            averageReward += reward
            if episode % 5 == 0 and not silent:
                game.ShowBoard()
            if episode % 20 == 0 and silent:
                game.ShowBoard()
          

        if episode % 10 == 0:
            print(f"Episode: {episode}, Reward: {averageReward/10}, Epsilon: {epsilon}")
            averageReward = 0
            epsilon = epsilon * (1 - LR) if epsilon > 0.01 else 0.01

        # Train the model with the accumulated batch
        # Model learns after a game is finished
        model.fit(np.array(state_batch), np.array(target_f_batch), epochs=1, verbose=0)
        state_batch, target_f_batch = [], []  # Clear the batch

    return model



def PlayWithTraining(model):
    game = Game(5, 3)
    state = game.GetState()
    state = np.array(state).reshape(1, 14)
    done = False
    while not done:
        # if(game.GetScore() <= -250):
        #     break
        game.ShowBoard()
        possible_moves = game.GetMoves()
        action = np.argmax(model.predict(state))
        if(action >= len(possible_moves)):
            action = np.random.randint(0, len(possible_moves))
        move = possible_moves[action]
        next_state, reward, done = game.Play(move[0], move[1],  defaultSnakeSize=3)
        
        next_state = np.array(next_state).reshape(1, 14)
        state = next_state
        time.sleep(0.1)
    print(f"Score: {game.GetScore(3)} Length: {len(game.snake.body)}")

def PlayWithTrainingEnsemble(ensemble, models):
    game = Game(5, 3)
    state = game.GetState()
    state = np.array(state).reshape(1, 14)
    done = False
    while not done:
        if(game.GetScore(3) <= -250):
            break
        game.ShowBoard()
        possible_moves = game.GetMoves()
        model_outputs = [np.argmax(model.predict(state)) for model in models]
        game_descriptors = np.array([game.moveCount, len(game.snake.body)])
        ensemble_input = np.concatenate([model_outputs, game_descriptors], axis=-1)
        ensemble_input = ensemble_input.reshape(1, 5)
        action = np.argmax(ensemble.predict(ensemble_input))
        if(action >= len(possible_moves)):
            action = np.random.randint(0, len(possible_moves))
        move = possible_moves[action]
        next_state, reward, done = game.Play(move[0], move[1],  defaultSnakeSize=3)

        next_state = np.array(next_state).reshape(1, 14)
        state = next_state
        
        time.sleep(0.1)
    print(f"Score: {game.GetScore(3)} Length: {len(game.snake.body)}")

def MakeModel(stringSnakeSize, snakeSize=3):
    model = CreateModel((14,), 4)
    model = train(model, 500, 0.9, 1, snakeSize=snakeSize)
    model.save(f"./submodels/{stringSnakeSize}/{len(os.listdir(f'./submodels/{stringSnakeSize}'))}.h5")
    return model

def MakeEnsembleModel(models):
    ensemble = CreateModel((5,), 4)
    ensemble = TrainEnsembleModel(ensemble, models)
    return ensemble


# We take in models for snake sizes of x y z and we average the output of the models to get the final output
# Lets train the ensemble model
def TrainEnsembleModel(ensemble, models: list[keras.Model], episodes=250, epsilon = 1.0, gamma = 0.9):
    for episode in range(episodes):
        game = Game(5, 3)
        state = game.GetState()
        state = np.array(state).reshape(1, 14)
        done = False

        while not done:

            if game.GetScore(3) <= -250:
                break

            # Make predictions for each model and concatenate them
            model_outputs = [np.argmax(model.predict(state)) for model in models]
            
            # Include game state descriptors
            game_descriptors = np.array([game.moveCount, len(game.snake.body)])
            ensemble_input = np.concatenate([model_outputs, game_descriptors], axis=-1)
            ensemble_input = ensemble_input.reshape(1, 5)

            # Use ensemble to predict and select action
            action_probs = ensemble.predict(ensemble_input)
            action = np.argmax(action_probs)

            # Check if action is valid or if random action should be taken
            possible_moves = game.GetMoves()
            if action >= len(possible_moves) or np.random.rand() < epsilon:
                action = np.random.randint(0, len(possible_moves))

            # Execute chosen action
            move = possible_moves[action]
            next_state, reward, done = game.Play(move[0], move[1], defaultSnakeSize=3)
            next_state = np.array(next_state).reshape(1, 14)

            # Prepare for next iteration and model training
            model_outputs = [np.argmax(model.predict(next_state)) for model in models]
            game_descriptors = np.array([game.moveCount, len(game.snake.body)])
            ensemble_input = np.concatenate([model_outputs, game_descriptors], axis=-1)
            ensemble_input = ensemble_input.reshape(1, 5)

            # Update target and train
            target = reward + 0.9 * np.max(ensemble.predict(ensemble_input))
            target_f = action_probs.copy()
            target_f[0][action] = target
            ensemble.fit(ensemble_input, target_f, epochs=1, verbose=0)
            state = next_state
            epsilon = epsilon * (1 - LR) if epsilon > 0.01 else 0.01
            if episode % 5 == 0:
                game.ShowBoard()

        if episode % 10 == 0:
            print(f"Episode: {episode}, Reward: {game.GetScore(3)}")

    return ensemble


            



            


                    