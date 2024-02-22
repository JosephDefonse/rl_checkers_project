# https://github.com/ImparaAI/checkers
from checkers.game import Game
from ast import literal_eval
import math
import random

game = Game()

def get_piece_detail(position):
    for piece in game.board.pieces:
        if piece.position == position:
            print("piece position is "+str(position))
            print("this piece belongs to player"+str(piece.player)) #1 or 2
            # print(piece.other_player) #1 or 2
            print("this piece is a king: "+str(piece.king)) #True or False
            print("this piece is captured: "+str(piece.captured)) #True or False
            print("moves to capture are: "+str(piece.get_possible_capture_moves())) #[[int, int], [int, int], ...]
            print("moves which don't capture are: "+str(piece.get_possible_positional_moves())) #[[int, int], [int, int], ...]

# print(len(game.board))

def board_state_arr():    
    arr = []
    for piece in game.board.pieces:
        total_moves = len(piece.get_possible_capture_moves())+len(piece.get_possible_positional_moves())
        arr.append([piece.position, piece.player, piece.king, piece.captured])
    return arr   

print(board_state_arr())

# [
#     [1, 1, False, False], 
#     [2, 1, False, False], 
#     [3, 1, False, False], 
#     [4, 1, False, False], 
#     [5, 1, False, False], 
#     [6, 1, False, False], 
#     [7, 1, False, False], 
#     [8, 1, False, False], 
#     [9, 1, False, False], 
#     [10, 1, False, False], 
#     [11, 1, False, False], 
#     [12, 1, False, False], 
#     [21, 2, False, False], 
#     [22, 2, False, False], 
#     [23, 2, False, False], 
#     [24, 2, False, False], 
#     [25, 2, False, False], 
#     [26, 2, False, False], 
#     [27, 2, False, False],
#     [28, 2, False, False], 
#     [29, 2, False, False], 
#     [30, 2, False, False], 
#     [31, 2, False, False], 
#     [32, 2, False, False]
# ]

# print(game.whose_turn()) #1 
# print(game.get_possible_moves()) #[[9, 13], [9, 14], [10, 14], [10, 15], [11, 15], [11, 16], [12, 16]]
# print(game.move([9, 13]))

# print(game.print_board())
# print(game.get_number_pieces())
# print(game.get_number_kings())

# print(game.whose_turn()) #2
# print(game.get_possible_moves()) #[[21, 17], [22, 17], [22, 18], [23, 18], [23, 19], [24, 19], [24, 20]]
# print(game.move([22, 17]))

# print(game.print_board())
# print(game.get_number_pieces())
# print(game.get_number_kings())

# print(game.whose_turn()) #1
# print(game.get_possible_moves()) 
# print(game.move([13, 22]))

# print(game.print_board())
# print(game.get_number_pieces())
# print(game.get_number_kings())

# print(game.whose_turn()) #1
# print(game.get_possible_moves()) 

game.consecutive_noncapture_move_limit = 10000000
while game.is_over() == False:
    print("-----------------------------")
    print("Player Turn: "+str(game.whose_turn()))
    print("Possible Moves: "+str(game.get_possible_moves()))
    player_move = []
    while player_move not in game.get_possible_moves():
        player_move = literal_eval(input("Please enter a move: "))
        print(player_move)
    # random_move = random.randint(0, len(game.get_possible_moves())-1)  
    game.move(player_move)
    print(game.print_board())
    print("-----------------------------")

print(game.get_winner())