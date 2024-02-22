import random
from ast import literal_eval
import torch
import torch.nn as nn
import torch.optim as optim
from checkers.game import Game

"""
https://towardsdatascience.com/everything-you-need-to-know-about-saving-weights-in-pytorch-572651f3f8de

SAVE MODEL:
torch.save(PLAYER1.q_neural_network.state_dict(), 'weights_only.pth')

LOAD MODEL:
checkpoint = torch.load('weights_only.pth')
PLAYER1.q_neural_network.load_state_dict(checkpoint)
"""

class DataLoader():
    """
    Dataloader which gets a random batch sample
    """
    def __init__(self, data, data_actions):
        self.training_data = data
        self.training_data_actions = data_actions

    def get_batch(self, amount):
        check_duplicates = [] # make sure it doesn't choose a sample already in the batch
        batch_sample = [] # the batch sample
        batch_sample_actions = [] # the batch sample with corresponding list of possible moves at current_state

        training_data = self.training_data
        training_data_actions = self.training_data_actions

        while len(batch_sample) < amount:
            rand = random.randint(0, len(training_data)-1)
            if rand not in check_duplicates:
                check_duplicates.append(rand)
                batch_sample.append(training_data[rand])
                batch_sample_actions.append(training_data_actions[rand])
        return batch_sample, batch_sample_actions

class DQN():
    """
    DQN (Deep Q Network) model which utilises 2 neural networks (q_neural_network and target_neural_network).
    """
    def __init__(self, player):
        self.player = player
        self.training_data = [] # the batch sample
        self.current_state_actions = [] # the batch sample with corresponding list of possible moves at current_state
        self.epsilon = 1 # epsilon value initialised as 1 (exploration)
        self.game = {} # the game state

        """
        Parameters
        """
        self.number_games = 300 # the number of games played
        self.init_games = 5 # the number of states we add to the batch dataset initially (experience replay)
        self.batch_size = 3 # the size of the batch list

        # training tracking data
        self.player_1_wins = 0
        self.player_2_wins = 0
        self.draws = 0

        NUMBER_PIECES = 24 # number of pieces on the board
        MAX_MOVES = 30 # average number of maximum moves for a given state
        NUMBER_TYPES_PIECES = 2 # king and normal
        self.MAX_ACTIONS = NUMBER_PIECES*MAX_MOVES*NUMBER_PIECES*NUMBER_TYPES_PIECES # Getting maximum actions

        self.STATE_VALUES = 4 # dimension initalised as the number input for the neural network (not technically useful as forward pass changes the input size)

        # q_neural_network model learns (determines the state and action - which is the q_value_predicted)
        self.q_neural_network = DynamicOutputGating(self.STATE_VALUES, 24, self.MAX_ACTIONS, 5)

        # target_neural_network model doesn't learn (determines the maximum q_value_target)
        self.target_neural_network = DynamicOutputGating(self.STATE_VALUES, 24, self.MAX_ACTIONS, 5)

    def opponent_player_move(self):
        while self.game.whose_turn() != self.player:
            # pick random move from the list of possible moves for the player
            possible_moves = self.game.get_possible_moves()
            random_move = random.randint(0, len(possible_moves)-1)
            self.game.move(possible_moves[random_move])

    def input_values(self):
        player_move = []
        print("Player Turn: "+str(self.game.whose_turn()))
        print("Possible Moves: "+str(self.game.get_possible_moves()))
        while player_move not in self.game.get_possible_moves():
            player_move = literal_eval(input("Please enter a move: "))
            print(player_move)
        self.game.move(player_move)

    def play(self):
        self.game = Game()

        while self.game.is_over() == False:
            # experience replay which adds to dataset (running state and action through neural network to q_predicted_value)
            self.experience_replay(self.game.board_state_arr(self.player))

    def train(self):
        for game_index in range(self.number_games):
            # initialising the game
            self.game = Game()

             # if a move which doesn't capture happen in 10000000 turns, then exit the game.
            self.game.consecutive_noncapture_move_limit = 10000000

            # initalising the DQN Experience Replay dataset
            self.initalisation(self.game.board_state_arr(self.player))

            # continue the training process until the game is over
            while self.game.is_over() == False:

                # opponent player makes a move
                if self.game.is_over() == False:
                    self.input_values()
                else:
                    break

                # experience replay which adds to dataset (running state and action through neural network to q_predicted_value)
                self.experience_replay(self.game.board_state_arr(self.player))

                # initialising the dataloader
                dataset = DataLoader(self.training_data, self.current_state_actions)
                # geting batches
                batch, batch_actions = dataset.get_batch(self.batch_size)

                # opponent player makes a move
                if self.game.is_over() == False:
                    self.opponent_player_move()
                else:
                    break

                # initialising the loss criterion for q_neural_network
                loss_class = self.q_neural_network.criterion
                # initialising optimisation for q_neural_network
                optimi_data = self.q_neural_network.optimizer

                for idx in range(len(batch)):
                    # getting sample idx from the batches
                    training_data_sample = batch[idx]
                    number_actions = len(batch_actions[idx])

                    # Zero the gradients
                    optimi_data.zero_grad()

                    # get current_state and action from sample
                    current_state = training_data_sample[0]
                    action = training_data_sample[1]

                    # convert array current_state to tensor
                    t = torch.FloatTensor(current_state)

                    # get q_predicted list
                    q_predicted_list = self.q_neural_network(t, number_actions)
                    # get action_value
                    q_predict = q_predicted_list[action].detach().numpy()

                    # get next_state and action from sample
                    t = torch.FloatTensor(training_data_sample[-1])
                    action = training_data_sample[1]

                    # get q_target list
                    q_target_list = self.target_neural_network(t, number_actions)
                    # get action_value
                    q_target = max(q_target_list.detach().numpy())

                    # get reward value from sample
                    reward = training_data_sample[2]

                    # get target_q_value which is = reward+q_target = reward+max(q_target_list)
                    target_q_value = reward+q_target

                    # Compute loss
                    mseloss = loss_class(q_predicted_list, q_target_list)

                    # Perform backward pass
                    mseloss.backward()

                    # Perform optimization
                    optimi_data.step()

            if self.game.get_winner() == 1:
                self.player_1_wins += 1
            elif self.game.get_winner() == 2:
                self.player_2_wins += 1
            else:
                self.draws += 1

            print("game "+str(game_index)+" finished")

        print("player 1 won: "+str(self.player_1_wins/self.number_games))
        print("player 2 won: "+str(self.player_2_wins/self.number_games))
        print("draws: "+str(self.draws/self.number_games))

    def experience_replay(self, current_state):
        # structured as [current_state, action, reward, next_state]
        experience_replay = []

        # adding current_state
        experience_replay.append(current_state)

        if self.game.whose_turn() == self.player: # our turn

            # get list of possible moves
            possible_moves = self.game.get_possible_moves()

            # add to action batch
            self.current_state_actions.append(possible_moves)

            # get board state (taken as input for neural network)
            board = self.game.board_state_arr(self.player)
            # convert board to tensor
            board_tensor = torch.FloatTensor(board)

            # get q_predicted_values and choose the move based on epsilon greedy
            q_predicted_list = self.q_neural_network(board_tensor, len(possible_moves))
            q_predicted_move = self.epsilon_greedy(q_predicted_list)

            # get reward and next_state based on epsilon greedy move
            reward, next_state = self.get_reward_and_next_state(possible_moves[q_predicted_move])
            print("I made move: "+str(possible_moves[q_predicted_move]))
            print(self.game.print_board())

            # add rest of value to sample experience_replay
            experience_replay.append(q_predicted_move) # add action move to experience_replay
            experience_replay.append(reward) # add reward to experience_replay
            experience_replay.append(next_state) # add next state to experience_replay

            # add experience_replay to the batch (training_data / experience_replay)
            self.training_data.append(experience_replay)
        else:
            self.input_values()
            print(self.game.print_board())

    def experience_replay_init(self, current_state):
        # structured as [current_state, action, reward, next_state]
        experience_replay = []

        # adding current_state
        experience_replay.append(current_state)

        if self.game.whose_turn() == self.player: # our turn

            # get list of possible moves
            possible_moves = self.game.get_possible_moves()

            # add to action batch
            self.current_state_actions.append(possible_moves)

            # get random_move and play it
            random_move = random.randint(0, len(possible_moves)-1)
            reward, next_state = self.get_reward_and_next_state(possible_moves[random_move])

            # add rest of value to sample experience_replay
            experience_replay.append(random_move) # add action move to experience_replay
            experience_replay.append(reward) # add reward to experience_replay
            experience_replay.append(next_state) # add next state to experience_replay

            # add experience_replay to the batch (training_data / experience_replay)
            self.training_data.append(experience_replay)
        else:
            self.input_values()

    def initalisation(self, current_state):
        # execute a self.init_games number of times to initialise the experience_replay dataset
        for _ in range(self.init_games):
            self.experience_replay_init(current_state)

        # Transfering initial weights from Q Neural Network to Target Neural Network
        self.target_neural_network.load_state_dict(self.q_neural_network.state_dict())

    def get_reward_and_next_state(self, action):

        # get current_state and perform action
        current_state = self.game
        current_state.move(action)

        # get next_state
        next_state = self.game
        next_state_value = self.game.board_state_arr(self.player)

        # get reward from the current_state, action and next_state
        reward = self.get_reward(current_state, next_state)

        return reward, next_state_value

    def get_reward(self, current_state, next_state):
        # using current_state and next_state to get the number of pieces on the board
        # using the number of pieces to calculate if lost pieces
        num_pieces_player_1_current, num_pieces_player_2_current = current_state.get_number_pieces()
        num_pieces_player_1_next, num_pieces_player_2_next = next_state.get_number_pieces()

        reward = 0
        if self.player == 1:
            if num_pieces_player_1_next - num_pieces_player_1_current < 0: # if you lose a piece
                reward = -1

            if num_pieces_player_2_next - num_pieces_player_2_current < 0: # if you take a piece
                reward = 1
        else:
            if num_pieces_player_1_next - num_pieces_player_1_current < 0: # if you take a piece
                reward = 1

            if num_pieces_player_2_next - num_pieces_player_2_current < 0: # if you lose a piece
                reward = -1
        return reward

    def epsilon_greedy(self, array):
        epsilon = self.epsilon

        # choosing random number to compare against epsilon
        random_num = random.uniform(0, 1)

        # getting python array type from numpy type
        array = array.detach().numpy()

        if random_num > epsilon: # exploitation
            # Exploitation: Will choose the action with the highest Q-value for its current state from the Q-table
            self.epsilon += 0.1
            # choose the maximum (highest q_predicted)
            max_value = max(array)
            for i in range(len(array)):
                if array[i] == max_value:
                    return i
        else: # exploration
            # Exploration: Randomly choosing its action and exploring what happens in the environment.
            self.epsilon -= 0.1
            random_move = random.randint(0, len(array)-1)
            return random_move

class DynamicOutputGating(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(DynamicOutputGating, self).__init__()

        # Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_hidden_layers)])
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.learning_rate = 0.05

        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, current_state, num_actions):
         # x.size() is 12x24 because size input is 12 and final input is 24
        current_state = nn.functional.relu(self.fc1(current_state))
        for layer in self.hidden_layers:
            current_state = nn.functional.relu(layer(current_state))

        # Sample a gating unit based on the probability distribution
        selected_gating_unit = 4

        # Use the selected gating unit to generate the output
        output = nn.functional.relu(self.hidden_layers[selected_gating_unit](current_state))
        output = nn.functional.linear(output, nn.Parameter(torch.randn((num_actions, self.hidden_size))))
        output = output.transpose(0, 1)
        mean = torch.mean(output, axis = 1)
        return mean

PLAYER1 = DQN(2)
# PLAYER1.train()
PLAYER1.play()