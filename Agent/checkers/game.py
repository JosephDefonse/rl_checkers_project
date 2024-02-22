from .board import Board
import math

class Game:

	def __init__(self):
		self.board = Board()
		self.moves = []
		self.consecutive_noncapture_move_limit = 40
		self.moves_since_last_capture = 0

	def move(self, move):
		if move not in self.get_possible_moves():
			raise ValueError('The provided move is not possible')

		self.board = self.board.create_new_board_from_move(move)
		self.moves.append(move)
		self.moves_since_last_capture = 0 if self.board.previous_move_was_capture else self.moves_since_last_capture + 1

		return self

	def board_state_arr(self, player):    
		arr = []
		total_moves = 0
		for piece in self.board.pieces:
			if piece.player == player and piece.position != None:
				possible_capture_moves = piece.get_possible_capture_moves()
				possible_positional_moves = piece.get_possible_positional_moves()
				total_moves += len(possible_capture_moves)+len(possible_positional_moves)
				arr.append([piece.position, piece.player, piece.king, piece.captured])
		# arr.append(total_moves)
			# arr.append([piece.position, piece.player, piece.king, piece.captured, piece.get_possible_capture_moves(), piece.get_possible_positional_moves()])
		return arr   

	def get_number_pieces(self):
		pieces_player_1 = 0
		pieces_player_2 = 0
		for piece in self.board.pieces:
			if piece.player == 1:
				if piece.captured == False:
					pieces_player_1 += 1
			else:
				if piece.captured == False:
					pieces_player_2 += 1
		return pieces_player_1, pieces_player_2

	def get_number_kings(self):
		pieces_player_1 = 0
		pieces_player_2 = 0
		for piece in self.board.pieces:
			if piece.player == 1:
				if piece.captured == False:
					if piece.king == True:
						pieces_player_1 += 1
			else:
				if piece.captured == False:
					if piece.king == True:
						pieces_player_2 += 1
		return pieces_player_1, pieces_player_2


	def print_board(self):
		graph = [
		]

		for _ in range(8):
			row = []        
			for _ in range(8):
				row.append('-')
			graph.append(row)
		
		"""
		--------
		--------
		--------
		--------
		--------
		--------
		--------
		--------
		""" 
		# print(graph)

		# darker colour moves first
		# assume player 1 is dark (x)
		# assume player 2 is light (o)
		for piece in self.board.pieces:
			if piece.captured == False:
				position = piece.position
				# print("position: ", str(piece.position))

				index_row = len(row)-(math.ceil(position/4)-1)-1

				if ((position%4) == 0) and (index_row%2 == 0): # odd
					index_column = 1
				elif ((position%4) == 0) and (index_row%2 != 0): # even
					index_column = 0
				elif (index_row%2 != 0):
					index_column = len(row)-(((position%4)*2))
				elif (index_row%2 == 0):
					index_column = len(row)-(((position%4)*2)-1)

				# print("row: ", str(index_row))
				# print("column: ", str(index_column))
				

				if piece.player == 1:
					graph[index_row][index_column] = 'x'

				else:
					graph[index_row][index_column] = 'o'
		
		string = ''
		for i in range(len(graph)):
			sub_string = ''
			for j in range(len(graph[i])):
				sub_string += graph[i][j]
			sub_string += '\n'
			string += sub_string

		return (string)

	def move_limit_reached(self):
		return self.moves_since_last_capture >= self.consecutive_noncapture_move_limit

	def is_over(self):
		return self.move_limit_reached() or not self.get_possible_moves()

	def get_winner(self):
		if self.whose_turn() == 1 and not self.board.count_movable_player_pieces(1):
			return 2
		elif self.whose_turn() == 2 and not self.board.count_movable_player_pieces(2):
			return 1
		else:
			return None

	def get_possible_moves(self):
		return self.board.get_possible_moves()

	def whose_turn(self):
		return self.board.player_turn
