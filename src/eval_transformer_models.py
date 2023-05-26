import os
import json
import argparse
from time import time
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import torch.nn as nn
from models.transformer import get_model
from datasets import SudokuDataset_Mask,SudokuDataset_Solver
from sudoku_solver.board import check_input_board,check_consistency_board

def init_parser():
    parser = argparse.ArgumentParser(description='Solver-NN and Mask-Predictor Module for NASR')
    # General args
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='Batch size')
    parser.add_argument('--data', type=str, default='big_kaggle',
                        help='dataset name between [big_kaggle, minimal_17, multiple_sol, satnet]')
    parser.add_argument('--noise-setting', default='xxx/yyy.json', type=str,
                        help='Json file of noise setting (dict)')
    parser.add_argument('--module', type=str, default='mask',
                        help='module name between mask or solvernn')
    # Model args
    parser.add_argument('--pos-weights', default=None, type=float,
                        help='ratio neg/pos examples')
    parser.add_argument('--block-len', default=81, type=int,
                        help='board size')
    return parser


def validate(test_loader, model, criterion, args):
    model.eval()
    loss_val = 0
    n = 0
    preds = []
    labels = []
    inputs = []
    time_begin = time()
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.to(args.gpu_id)
            target = target.to(args.gpu_id)

            pred = model(images)
            loss = criterion(pred, target)
            
            preds.append(pred)
            labels.append(target)
            inputs.append(images)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss = (loss_val / n)
                print(f'[Test][{i}/{len(test_loader)}] \t AvgLoss: {avg_loss:.4f}')

    avg_loss = (loss_val / n)
    total_mins = (time() - time_begin) / 60
    print(f'AvgLoss {avg_loss:.4f} \t \t Time: {total_mins:.2f}')

    if args.module == 'mask':
        y_pred = np.round(torch.cat(preds, 0).sigmoid().cpu().numpy())
        y_targ = torch.cat(labels, 0).cpu().numpy()
        eval_mask(y_pred,y_targ)
    else:
        assert args.module == 'solvernn'
        y_pred = torch.cat(preds, 0).sigmoid().cpu().numpy()
        y_targ = torch.cat(labels, 0).cpu().numpy()
        input_b = torch.cat(inputs, 0).cpu().numpy()
        eval_solver(y_pred,y_targ,input_b,dataset_name=args.data)
    

def eval_mask(y_pred,y_targ):
    num_correct_0 = [0] * len(y_pred)
    num_wrong_0 = [0] * len(y_pred)
    num_correct_1 = [0] * len(y_pred)
    num_wrong_1 = [0] * len(y_pred)
    for i in range(len(y_pred)):
        p = y_pred[i]
        l = y_targ[i]
        num_0=0
        for j in range(len(p)):
            if int(l[j]) == 0.:
                num_0 +=1
                if int(p[j]) == 0:
                    num_correct_0[i]+=1
                else:
                    num_wrong_0[i] +=1 
            else:
                if int(p[j]) == 1.:
                    num_correct_1[i] +=1
                else:
                    num_wrong_1[i] +=1
        if num_0 == 0:
            num_correct_0[i] = 1
        else:
            num_correct_0[i] = num_correct_0[i]/num_0
        if 81-num_0 == 0:
            num_correct_1[i] =1
        else:
            num_correct_1[i] = num_correct_1[i]/(81-num_0)
    print(f'Correctness of 0 (TN!): {np.mean(num_correct_0):.4f}')
    print(f'Correctness of 1 (TP): {np.mean(num_correct_1):.4f}')


def eval_solver(y_pred,y_targ,input_b,dataset_name = None):
    input_boards = []
    pred_boards = []
    targ_boards = []
    for idx in range(len(y_pred)):
        i_board = input_b[idx]
        input_board = np.zeros((9, 9), dtype=int)
        r_pred_board = np.zeros((9, 9), dtype=int)
        r_target_board = np.zeros((9, 9), dtype=int)
        pred_board = y_pred[idx]
        targ_board = y_targ[idx]
        for k in range(81):
            i= k//9
            j= k - (k//9)*9
            input_board[i][j] = (np.where(i_board[k]== 1.)[0][0])
            target_cell_i_j = targ_board[k]
            pred_cell_i_j = pred_board[k]
            target_value = (np.where(target_cell_i_j== 1.)[0][0])+1
            pred_value = pred_cell_i_j.argmax()+1
            r_pred_board[i][j] = float(pred_value)
            r_target_board[i][j] = float(target_value)
        pred_boards.append(r_pred_board)
        targ_boards.append(r_target_board) 
        input_boards.append(input_board)
    stats_input = []
    stats_sol = []
    totally_correct = 0
    for k in range(len(pred_boards)):
        num_preserved_input = 0
        num_non_preserved_input = 0 
        num_correct_solution = 0
        num_non_correct_solution = 0
        input_board = input_boards[k]
        pred_board = pred_boards[k]
        targ_board = targ_boards[k]
        for i in range(9):
            for j in range(9):
                if input_board[i][j] == 0:
                    if targ_board[i][j] == pred_board[i][j] :
                        num_correct_solution +=1
                    else:
                        num_non_correct_solution +=1
                else:
                    assert targ_board[i][j] == input_board[i][j]
                    if targ_board[i][j] == pred_board[i][j] :
                        num_preserved_input +=1
                    else:
                        num_non_preserved_input +=1
        if num_correct_solution+num_preserved_input == 81:
            totally_correct+=1
        else:
            if dataset_name== 'multiple_sol':
                check_input = check_input_board(input_board,pred_board)
                consistent = check_consistency_board(pred_board)
                if check_input and consistent:
                    totally_correct+=1
                    #print('alternative sol found')
        stats_input.append(num_preserved_input/(num_preserved_input+num_non_preserved_input))
        stats_sol.append(num_correct_solution/(num_non_correct_solution+num_correct_solution))
    
    print(f'Preservation of input: {np.mean(stats_input):.4f}')
    print(f'Correctness of solutions: {np.mean(stats_sol):.4f}')
    print(f'Completely correct solution boards: {totally_correct}/{len(pred_boards)} ({totally_correct*100/len(pred_boards):.2f}%)')
    for k in range(len(pred_boards)):
        targ_boards[k] = np.array(targ_boards[k]).reshape(81,)
        pred_boards[k] = np.array(pred_boards[k]).reshape(81,)


def main():
    parser = init_parser()
    args = parser.parse_args()
    # ----------------------------------
    ckpt_path = f'outputs/{args.module}/{args.data}/checkpoint_best.pth'
    # ----------------------------------
    if os.path.isfile(args.noise_setting):
        with open(args.noise_setting) as f:
            noise_setting = json.load(f)
    else:
        noise_setting = {"noise_type": "awgn", "snr": -0.5}
    noise_setting = str(noise_setting).replace(' ', '').replace("'", "")

    assert args.module in ['mask', 'solvernn'], 'error module name, choose between solvernn and mask'
    if args.module == 'mask':
        test_dataset = SudokuDataset_Mask(args.data,'-test')
    elif args.module == 'solvernn':
       test_dataset = SudokuDataset_Solver(args.data,'-test')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    # Model
    num_classes = -1
    in_chans = -1
    if args.module == 'solvernn':
        in_chans = 10
        num_classes = 9
    else:
        assert args.module == 'mask'
        in_chans = 9
        num_classes = 1

    model = get_model(block_len=args.block_len, in_chans = in_chans, num_classes = num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.to(args.gpu_id)

    # Loss
    if args.pos_weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weights))
    else:
        criterion = nn.BCEWithLogitsLoss()
 
    print("Beginning testing")
    validate(test_loader, model, criterion, args)
 

if __name__ == '__main__':    
    main()
