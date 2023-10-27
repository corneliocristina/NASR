import os
import json
import argparse
from time import time
import math
import numpy as np
import torch
import torch.nn as nn
import statistics
from models.transformer_sudoku import get_model
from datasets import SudokuDataset_RL
import random
from rl_train_sudoku import compute_reward
from sudoku_solver.board import Board

try:
    from pyswip import Prolog
except Exception:
    print('-->> Prolog not installed')
    
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
import optuna
from sudoku_solver.board import check_input_board,check_consistency_board
from utils.utils import retrieve_hints_from_solution

def init_parser():
    parser = argparse.ArgumentParser(description='Quick testing script')

    # General args
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='Batch size')
    parser.add_argument('--data', type=str, default='big_kaggle',
                        help='dataset name between big_kaggle, minimal_17, multiple_sol and satnet')
    parser.add_argument('--noise-setting', default='xxx/yyy.json', type=str,
                        help='Json file of noise setting (dict)')
    parser.add_argument('--solver', type=str, default='prolog',
                        help='symbolic solver to use. available options prolog and backtrack')
    parser.add_argument('--nasr', type=str, default='pretrained',
                        help='choice of nasr with nasr_rl or pretrained (without rl)')
    # Model args
    parser.add_argument('--block-len', default=81, type=int,
                        help='board size')
    parser.add_argument('--code-rate', default=2, type=int,
                        help='Code rate')

    return parser


def final_output(model,ground_truth_sol,solution_boards,ablation_mask_boards, param_abl, args):
    
    ground_truth_boards = torch.argmax(ground_truth_sol,dim=2)
    solution_boards = solution_boards.softmax(dim=2)
    solution_boards_new = torch.argmax(solution_boards,dim=2)+1

    for i in range(len(solution_boards)):
        for j in range(81):
            value = int(solution_boards_new[i][j]-1)
            if solution_boards[i][j][value] < param_abl:
                ablation_mask_boards[i][j] = 0
    
    cleaned_boards = np.multiply(solution_boards_new.cpu(),ablation_mask_boards)
    
    final_boards = []
    if args.solver == "prolog":
        prolog_instance = Prolog()
        prolog_instance.consult("src/sudoku_solver/sudoku_prolog.pl")
    for i in range(len(cleaned_boards)):
        board_to_solver = Board(cleaned_boards[i].reshape((9,9)).int())
        try:
            if args.solver == "prolog":
                solver_success = board_to_solver.solve(solver ='prolog',prolog_instance = prolog_instance)
            else:
                solver_success = board_to_solver.solve(solver ='backtrack')
        except StopIteration:
            solver_success = False
        final_solution = board_to_solver.board.reshape(81,)
        if not solver_success:
            final_solution = solution_boards_new[i].cpu()
        reward = compute_reward(solution_boards_new[i].cpu(),final_solution,ground_truth_boards[i])
        model.rewards.append(reward)
        final_boards.append(final_solution)
    return final_boards


def validate(val_loader, model, param_abl, args):
    model.eval()
    loss_value = 0
    n = 0
    preds = []
    labels = []
    solutions = []
    eps = np.finfo(np.float32).eps.item()
    time_begin = time()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            
            images = images.to(args.gpu_id)
            target = target.to(args.gpu_id)

            solution_boards, _ = model(images, nasr=args.nasr)
            solution_boards_soft, masking_boards = model(images)
            ablation_boards = torch.ones(masking_boards.size())
            pred_final = final_output(model,target,solution_boards_soft,ablation_boards,param_abl,args)
            rewards = np.array(model.rewards)
            rewards = (rewards - rewards.mean())/(rewards.std() + eps)

            preds.append(pred_final)
            labels.append(target)
            solutions.append(solution_boards)            

            n += images.size(0)
            torch.cuda.empty_cache()

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss = (loss_value / n)
                print(f'[Test][{i}/{len(val_loader)}] \t AvgLoss: {avg_loss:.4f}')
    avg_loss = (loss_value / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'AvgLoss {avg_loss:.4f} \t \t Time: {total_mins:.2f}')

    pred_boards_b = torch.cat(solutions, 0).sigmoid().cpu().numpy()
    y_pred_b = []
    for idx in range(len(pred_boards_b)):
        r_pred_board = np.zeros((81), dtype=int)
        pred_board = pred_boards_b[idx]
        for k in range(81):
            pred_cell = pred_board[k].argmax()+1
            r_pred_board[k] = float(pred_cell)
        y_pred_b.append(r_pred_board)

    y_pred = []
    for j in preds:
        for i in j:
            y_pred.append(i) 
    y_targ = torch.cat(labels, 0).cpu().numpy()
    print("-------------------------------")
    print("Improvement:")
    print("-----------")
    performance = eval_improvement(y_pred,y_pred_b,y_targ)
    print("-------------------------------")
    return performance

def eval_improvement(y_pred,y_pred_b,y_target,dataset_name=None):
    y_pred = [l.tolist()for l in y_pred]
    y_pred_b = [l.tolist()for l in y_pred_b]
    y_pred = [list(map(int,i)) for i in y_pred]
    y_pred_b = [list(map(int,i)) for i in y_pred_b]
    y_targ = []
    for board in y_target:
        tmp_board = []
        for cell in board:
            tmp_board.append(cell.argmax())
        y_targ.append(tmp_board)
    y_targ = [list(map(int,i)) for i in y_targ]
    num_total = len(y_pred)
    num_correct_b =0
    num_correct = 0
    num_corrected_boards = 0
    num_wrong_boards = 0
    average_correct_cells = []
    for i in range(len(y_pred)):
        correct_cells=0.0
        if y_pred_b[i] == y_targ[i]:
            num_correct_b += 1
        else:
            if dataset_name == 'multiple_sol':
                input_board = retrieve_hints_from_solution(np.array(y_targ[i]),dataset_name)
                check_input = check_input_board(input_board,np.array(y_pred_b[i]))
                consistent = check_consistency_board(np.array(y_pred_b[i]))
                if check_input and consistent:
                    num_correct_b += 1
                    print('alternative solution found')
                else:
                  if y_pred[i] == y_targ[i]:
                    num_corrected_boards += 1  
            else:
                if y_pred[i] == y_targ[i]:
                    num_corrected_boards += 1
        
        if y_pred[i] == y_targ[i]:
            num_correct += 1
        else:
            if y_pred_b[i] == y_targ[i]:
                num_wrong_boards += 1
        for j,k in zip(y_pred[i],y_targ[i]):
            if j==k:
                correct_cells+=1
        correct_cells/=81
        average_correct_cells.append(correct_cells)
    print(f"* Correct solution boards from Neuro-Solver: {num_correct_b}/{num_total}-- ({(num_correct_b*100.)/(num_total):.2f}%)")
    print(f"* Correct solution boards from ablation pipeline: {num_correct}/{num_total}-- ({((num_correct)*100.)/(num_total):.2f}%)")
    if (num_total-num_correct_b) >0 :
        print(f"* Num. of wrong solution boards (from Neuro-solver) \n  that have been corrected with ablation pipeline: {num_corrected_boards}/{num_total-num_correct_b} -- ({(num_corrected_boards*100.)/(num_total-num_correct_b):.2f}%)")
    if num_correct_b>0 :
        print(f"* Num. of correct solution boards (from Neuro-solver) \n  that have been corrupted with ablation pipeline: {num_wrong_boards}/{num_correct_b} -- ({num_wrong_boards*100./num_correct_b:.2f}%)")
    print(f"* Num correct cells (avg): {statistics.mean(average_correct_cells):.2f}%")
    return ((num_correct)*100.)/(num_total)


def main_train(param_abl, i_data = None):
    parser = init_parser()
    args = parser.parse_args()
    if i_data:
        args.data = i_data
        print(i_data)

    valid_dataset = SudokuDataset_RL(args.data,'-valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)

    # Model
    model = get_model(block_len=args.block_len)
    model.load_pretrained_models(args.data) 
    model.to(args.gpu_id)

    # Main loop
    performance = validate(valid_loader, model, param_abl, args)
    return performance
 


def main_test(param_abl, i_data = None):

    parser = init_parser()
    args = parser.parse_args()

    if i_data:
        args.data = i_data
        print(i_data)
    test_dataset = SudokuDataset_RL(args.data,'-test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)

    # Model
    model = get_model(block_len=args.block_len)
    model.load_pretrained_models(args.data) 
    model.to(args.gpu_id)

    # Main loop
    performance = validate(test_loader, model, param_abl, args)
    return performance
 

def parameter_tuning_big_kaggle():
    study = optuna.create_study()
    study.optimize(objective_big_kaggle, n_trials=100)
    x = study.best_params
    print(f'\nBest param_abl = {x}')

def parameter_tuning_multiple_sol():
    study = optuna.create_study()
    study.optimize(objective_multiple_sol, n_trials=100)
    x = study.best_params
    print(f'\nBest param_abl = {x}')

def parameter_tuning_satnet():
    study = optuna.create_study()
    study.optimize(objective_satnet, n_trials=100)
    x = study.best_params
    print(f'\nBest param_abl = {x}')

def parameter_tuning_minimal_17():
    study = optuna.create_study()
    study.optimize(objective_minimal_17, n_trials=100)
    x = study.best_params
    print(f'\nBest param_abl = {x}')


def objective_big_kaggle(trial):
    param_abl = trial.suggest_float('param_abl',0.24,0.26)
    return 100 - main_train(param_abl, i_data="big_kaggle")

def objective_multiple_sol(trial):
    param_abl = trial.suggest_float('param_abl',0.24,0.26)
    return 100 - main_train(param_abl, i_data="multiple_sol")

def objective_satnet(trial):
    param_abl = trial.suggest_float('param_abl',0.24,0.26)
    return 100 - main_train(param_abl, i_data="satnet")

def objective_minimal_17(trial):
    param_abl = trial.suggest_float('param_abl',0.24,0.26)
    return 100 - main_train(param_abl, i_data="minimal_17")



if __name__ == '__main__':

    # HOW TO RUN
    # choose a dataset, then run the corresponding parameter_tuning function
    # this will give you the best parameter
    # then run main_test with this parameter
    # best param_abl by grid search = 0.253 

    # big_kaggle
    # parameter_tuning_big_kaggle()     # -> param_abl = 0.253
    # main_test(0.253,"big_kaggle")

    # minimal_17
    # parameter_tuning_minimal_17()     # -> param_abl = 0.25325830761837465
    # main_test(0.25325830761837465,"minimal_17") 

    # multiple_sol
    # parameter_tuning_multiple_sol()   # -> param_abl = 0.25119964585119836
    # main_test(0.25119964585119836,"multiple_sol")

    # satnet
    # parameter_tuning_satnet()         # -> param_abl = 0.2533129068004966
    # main_test(0.2533129068004966,"satnet")


    
    
   
    
