import sys
import numpy as np
import time
import statistics
import matplotlib.pyplot as plt
import mnist
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

try:
    from sudoku_solver.sudoku_pl import solve_sudoku
except Exception:
    print('-->> Prolog not installed')
    



data_path = 'data/'

def get_number_img(num,labels= mnist.test_labels(),images=mnist.test_images()):
    idxs = np.where(labels == num)[0]
    idx = np.random.choice(idxs, 1)
    img = images[idx].reshape((28, 28)).astype(int)
    background = 255
    img = background - img  # makes background white
    return img


def expand_line(line):
    base = 3
    return line[0] + line[5:9].join([line[1:5] * (base - 1)] * base) + line[9:13]


def check_input_board(input_board,pred_board):
    input_board = input_board.reshape(81)
    pred_board = pred_board.reshape(81)
    for i in range(81):
        if input_board[i] != 0:
            if input_board[i] != pred_board[i]:
                return
    return True


def check_consistency_board(pred_board):
    board = pred_board.reshape(9,9)
 # Check row
    for k in range(9):
        row = board[k]
        if 9 != len(set(row)):
            return False
        column = [board[j][k] for j in range(9)]
        if 9 != len(set(column)):
            return False
        box = [(3*(k//3),j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)] + \
                [(3*(k//3)+1,j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)] + \
                [(3*(k//3)+2,j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)]
        box = [board[i][j] for (i,j) in box]
        if 9 != len(set(box)):
            return False
    return True





class Board:
    def __init__(self, board_init=None):
        if board_init is None:
            self.board = np.zeros((9, 9), dtype=int)
        else:
            self.board = np.array(board_init)
        self.visual_board = None

    def generate_mnist_board(self,labels,images):
        board_img = np.empty((28 * 9, 28 * 9))
        board_img.fill(255)
        for i in range(9):
            for j in range(9):
                num = self.board[i][j]
                if (num is not None) and (num != 0):
                    num_img = get_number_img(num,labels,images)
                    rows = slice(28 * (i % 10), 28 * ((i + 1) % 10))
                    cols = slice(28 * (j % 10), 28 * ((j + 1) % 10))
                    board_img[rows, cols] = num_img
        self.visual_board = board_img


    def visualize(self,file_name='board.png'):
        fig = plt.figure(figsize=(5, 5), dpi=100)
        figure = fig.add_subplot(111)
        major_ticks = np.arange(0, 252+28, step=84)
        minor_ticks = np.arange(0, 252+28, step=28)
        figure.set_xticks(major_ticks)
        figure.set_xticks(minor_ticks, minor=True)
        figure.set_yticks(major_ticks)
        figure.set_yticks(minor_ticks, minor=True)
        figure.grid(True, which='both', color='k', linestyle='-')
        plt.grid()
        figure.grid(True, which='major', alpha=1, linewidth=1)
        figure.grid(True, which='minor', alpha=0.5, linewidth=0.5)
        figure.set_xticklabels([])
        figure.set_yticklabels([])
        for tick in figure.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in figure.xaxis.get_minor_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in figure.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in figure.yaxis.get_minor_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        plt.xlim(0, 28 * 9)
        plt.ylim(28 * 9, 0)
        plt.imshow(self.visual_board)
        plt.gray()
        plt.savefig('outputs/images/'+file_name)

    def visualize_shell(self):
        print('\n')
        line0 = expand_line("╔═══╤═══╦═══╗")
        line1 = expand_line("║ . │ . ║ . ║")
        line2 = expand_line("╟───┼───╫───╢")
        line3 = expand_line("╠═══╪═══╬═══╣")
        line4 = expand_line("╚═══╧═══╩═══╝")
        symbol = " 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        nums = [[""] + [symbol[n] for n in row] for row in self.board]
        print(line0)
        for r in range(1, 10):
            print("".join(n + s for n, s in zip(nums[r - 1], line1.split("."))))
            print([line2, line3, line4][(r % 9 == 0) + (r % 3 == 0)])

    def solve(self, solver = 'prolog', prolog_instance = None):
        '''
        @solver : 'prolog', 'backtrack'
        '''
        if self.input_is_valid() == False:
            return False
        if solver == 'prolog':
            if prolog_instance:
                solution = solve_sudoku(self.board, prolog_instance)
            else:
                solution = solve_sudoku(self.board)
            if len(solution)>0:
                self.board = solution
                return True
            else:
                return False
        elif solver == 'backtrack':
            find = self.find_empty()
            if not find:
                return True
            else:
                row, col = find
            for i in range(1,10):
                if self.is_valid(i, (row, col)):
                    self.board[row][col] = i
                    if self.solve('backtrack'):
                        return True
                    self.board[row][col] = 0
            return False

    def board_string(self):
        out = self.board.reshape(81,1).tolist()
        out = [i[0] for i in out]
        out = ''.join(str(i) for i in out)
        return out

    def print_board(self):
        print('\n')
        for i in range(len(self.board)):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - - - - ")
            for j in range(len(self.board[0])):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")
                if j == 8:
                    print(self.board[i][j])
                else:
                    print(str(self.board[i][j]) + " ", end="")

    def find_empty(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 0:
                    return i, j  # row, col
        return None

    def is_valid(self, num, pos):
        # Check row
        for i in range(len(self.board[0])):
            if self.board[pos[0]][i] == num and pos[1] != i:
                return False
        # Check column
        for i in range(len(self.board)):
            if self.board[i][pos[1]] == num and pos[0] != i:
                return False
        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if self.board[i][j] == num and (i,j) != pos:
                    return False
        return True


    def input_is_valid(self):
        n = len(self.board[0])
        for k in range(n):
            row = self.board[k]
            row = list(filter(lambda a: a != 0, row))
            if len(row) != len(set(row)):
                return False
            column = [self.board[j][k] for j in range(n)]
            column = list(filter(lambda a: a != 0, column))
            if len(column) != len(set(column)):
                return False
            box = [(3*(k//3),j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)] + \
                    [(3*(k//3)+1,j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)] + \
                    [(3*(k//3)+2,j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)]
            box = [self.board[i][j] for (i,j) in box]
            box = list(filter(lambda a: a != 0, box))
            if len(box) != len(set(box)):
                return False
        return True      

