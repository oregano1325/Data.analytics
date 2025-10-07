'''## **2048 Solver Using Minimax**

This project implements the game **2048** along with an automated **solver** that plays the game using a **Minimax algorithm** and a custom evaluation function to determine the best possible moves.

Each board configuration has been represented as a 2D ARRAY until 2048 is reached or the game is terminated.

The depth search is 3 to both increase the accuracy and speed of solving


## **2048 Solver Using Minimax**

This project implements the game **2048** along with an automated **solver** that plays the game using a **Minimax algorithm** and a custom evaluation function to determine the best possible moves.


## **How the Solver Works**

The solver uses a **Minimax algorithm** with a fixed search depth to simulate possible moves and choose the one with the highest expected outcome. It alternates between:



* **Maximizing** moves (the player's turns), and 

* **Minimizing** outcomes (worst-case random tile placements) 


At each node, an **evaluation function** scores the board based on multiple factors:


## **Evaluation Function Breakdown**

The evaluation function is a combination of the following components:


### **1. Empty Tiles (Weighted)**



* More empty tiles give the player more flexibility and reduce the chance of losing. 

* Each empty tile adds `+5` points to the scores. 

* The more empty tiles the higher the probability the game goes on longer


### **2. Directional Monotonicity (Grid Orderliness)**



* Rewards rows and columns where values decrease or increase consistently. 

* Prevents disordered merges and helps stack larger tiles in one area. 

* For every adjacent pair where `tile[i] >= tile[i+1]`, 1 point is added.

 \



### **3. Tile Difference Penalty**



* Penalizes large differences between adjacent tiles, encouraging smooth merges. 

* Applies to both rows and columns. 

* Formula: `-abs(tile[i] - tile[i+1])` for each adjacent non-zero pair. 

* Ensures a number is not trapped in between 



### **4. Corner Bonus**



* Encourages keeping the **largest tile in the top-left corner**, which is a common human strategy. \

* If the highest tile is in the `[0, 0]` corner, a **bonus equal to 10 times the tile's value** is added to the score. \

* Large number in a corner tile is favourable

Final Formula

empty_tiles*5+direction_score(board)+diff_score(board)+corner_bonus

The objective is to maximise this score!

I have not been able to reach a 100% accuracy of creating 2048 but out of 50 trial runs done it made 2048 35 times, (70%) accuracy, although it made past 512 90% of the times.'''

import numpy as np
import random

GRID_SIZE = 4
SEARCH_DEPTH = 3  # Depth for minimax search

# Initialize 4x4 array filled with zeroes
def generate_empty_board():
    return np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

def add_random_tile(board):
    #Add a 2 or 4 to a random empty tile, based on the games probability
    empty_tiles = list(zip(*np.where(board == 0)))
    if empty_tiles:
        r, c = random.choice(empty_tiles)
        board[r, c] = 2 if random.random() < 0.9 else 4

# Board movement functions
def compress(row):
    #Move non zero elements to left
    new_row = row[row != 0]
    new_row = np.append(new_row, [0] * (GRID_SIZE - len(new_row)))
    return new_row

def merge(row):
  #Add equal tiles
    for i in range(GRID_SIZE - 1):
        if row[i] == row[i + 1] and row[i] != 0:
            row[i] *= 2
            row[i + 1] = 0
    return row

def move_left(board):
    new_board = np.zeros_like(board)
    for i in range(GRID_SIZE):
        new_row = compress(board[i])
        new_row = merge(new_row)
        new_row = compress(new_row)
        new_board[i] = new_row
    return new_board
#Right move is left move when board is flipped
def move_right(board):
    return np.fliplr(move_left(np.fliplr(board)))

def move_up(board):
    return np.rot90(move_left(np.rot90(board, -1)), 1)

def move_down(board):
    return np.rot90(move_left(np.rot90(board, 1)), -1)

def is_move_possible(board):
    #Check for any possible move
    if np.any(board == 0):
        return True
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            if board[i, j] == board[i, j + 1] or board[j, i] == board[j + 1, i]:
                return True
    return False

#Best move function
def evaluate_board(board):
    #Evaluate the best case secnario with the following parameters
    empty_tiles = np.sum(board == 0)

    def direction_score(board):
        score = 0
        for row in board:
            score += sum(row[i] >= row[i + 1] for i in range(GRID_SIZE - 1))
        for col in board.T:
            score += sum(col[i] >= col[i + 1] for i in range(GRID_SIZE - 1))
        return score

    def diff_score(board):
        diff = 0
        for row in board:
            for i in range(GRID_SIZE - 1):
                if row[i] > 0 and row[i + 1] > 0:
                    diff -= abs(row[i] - row[i + 1])
        for col in board.T:
            for i in range(GRID_SIZE - 1):
                if col[i] > 0 and col[i + 1] > 0:
                    diff -= abs(col[i] - col[i + 1])
        return diff

    max_tile = np.max(board)
    corner_bonus = max_tile * 10 if board[0, 0] == max_tile else 0

    return (empty_tiles * 5) + direction_score(board) + diff_score(board) + corner_bonus

# Minimax Algorithm: at chance nodes, we assume the worst-case (min) outcomes
def minimax(board, depth, is_max_player):
    if depth == 0 or not is_move_possible(board):
        return evaluate_board(board)

    if is_max_player:  # Player's Turn(maximize score)
        best_score = float("-inf")
        for move in [move_left, move_right, move_up, move_down]:
            new_board = move(board)
            if not np.array_equal(new_board, board):  # Valid move
                score = minimax(new_board, depth - 1, False)
                best_score = max(best_score, score)
        return best_score

    else:  # Chance node: assume the worst-case tile is placed
        empty_tiles = list(zip(*np.where(board == 0)))
        if not empty_tiles:
            return evaluate_board(board)

        worst_score = float("inf")
        for (r, c) in empty_tiles:
            for tile_value in [2, 4]:
                new_board = board.copy()
                new_board[r, c] = tile_value
                score = minimax(new_board, depth - 1, True)
                worst_score = min(worst_score, score)
        return worst_score

# Get the best move using Minimax
def best_move(board):
    best_score = float("-inf")
    best_action = None

    for move, direction in zip([move_left, move_right, move_up, move_down],
                                ["left", "right", "up", "down"]):
        new_board = move(board)
        if not np.array_equal(new_board, board):  # Valid move
            score = minimax(new_board, SEARCH_DEPTH, False)
            if score > best_score:
                best_score = score
                best_action = direction

    return best_action

# Play the game automated
def play_game():
    board = generate_empty_board()
    add_random_tile(board)
    add_random_tile(board)
    move_count = 0  # Track number of moves

    while is_move_possible(board):
        # Break if 2048 tile is achieved
        if np.max(board) >= 2048:
            print("2048 achieved!")
            break

        print(f"Move {move_count + 1}:")
        print(board, "\n")
        move_direction = best_move(board)
        if move_direction == "left":
            board = move_left(board)
        elif move_direction == "right":
            board = move_right(board)
        elif move_direction == "up":
            board = move_up(board)
        elif move_direction == "down":
            board = move_down(board)

        add_random_tile(board)
        move_count += 1

    print("Game Over!")
    print(board)
    print("Total moves:",move_count)

if __name__ == "__main__":
    play_game()
