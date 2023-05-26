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
from pyswip import Prolog
from torch.distributions.bernoulli import Bernoulli
from sudoku_solver.board import check_input_board,check_consistency_board
from models.perception import SequentialPerception
from tqdm import tqdm
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
    parser.add_argument('--nasr', type=str, default='rl',
                        help='choice of nasr with nasr_rl or pretrained (without rl)')
    parser.add_argument('--data', type=str, default='big_kaggle',
                        help='dataset name between big_kaggle, minimal_17, multiple_sol and satnet')
    parser.add_argument('--noise-setting', default='xxx/yyy.json', type=str,
                        help='Json file of noise setting (dict)')
    parser.add_argument('--solver', type=str, default='prolog',
                        help='symbolic solver to use. available options prolog and backtrack')
    parser.add_argument('--transform-data', type=str, default=None,
                        help='noise type to add to the images of sudoku digits. available options are blur and rotation')
    parser.add_argument('--transform-data-param', default=0.0, type=float,
                        help='noise value. angle degrees for rotation and sigma for blur.')
    # Model args
    parser.add_argument('--block-len', default=81, type=int,
                        help='board size')
    parser.add_argument('--code-rate', default=2, type=int,
                        help='Code rate')

    return parser


def final_output(model,ground_truth_sol,solution_boards,masking_boards,args):
    ground_truth_boards = torch.argmax(ground_truth_sol,dim=2)
    solution_boards_new = torch.argmax(solution_boards,dim=2)+1
    # using sigmoid_round
    masking_prob = masking_boards.sigmoid()
    b = Bernoulli(masking_prob)
    mask_boards = torch.round(masking_prob)
    model.saved_log_probs = b.log_prob(mask_boards)
    cleaned_boards = np.multiply(solution_boards_new.cpu(),mask_boards.cpu())
    final_boards = []
    if args.solver == "prolog":
        prolog_instance = Prolog()
        prolog_instance.consult("src/sudoku_solver/sudoku_prolog.pl")
    for i in range(len(cleaned_boards)):
        board_to_solver = Board(cleaned_boards[i].reshape((9,9)).int())
        time_begin = time()
        try:            
            if args.solver == "prolog":
                solver_success = board_to_solver.solve(solver ='prolog',prolog_instance = prolog_instance)
            else:
                solver_success = board_to_solver.solve(solver ='backtrack')
        except StopIteration:
            solver_success = False
        time_solver = (time() - time_begin)
        final_solution = board_to_solver.board.reshape(81,)
        if not solver_success:
            final_solution = solution_boards_new[i].cpu()
        reward = compute_reward(solution_boards_new[i].cpu(),final_solution,ground_truth_boards[i])
        model.rewards.append(reward)
        final_boards.append(final_solution)
    return final_boards, time_solver


def validate(val_loader, model, args, epoch=None, time_begin=None):
    model.eval()
    loss_value = 0
    n = 0
    preds = []
    labels = []
    inputs = []
    solutions = []
    mask = []
    eps = np.finfo(np.float32).eps.item()
    time_begin = time()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            
            images = images.to(args.gpu_id)
            target = target.to(args.gpu_id)

            solution_boards, masking_boards = model(images, nasr=args.nasr)
            pred_final, _ = final_output(model,target,solution_boards,masking_boards,args)
            rewards = np.array(model.rewards)
            rewards = (rewards - rewards.mean())/(rewards.std() + eps)
            
            mask.append(torch.round(masking_boards.sigmoid()))
            preds.append(pred_final)
            labels.append(target)
            inputs.append(images)
            solutions.append(solution_boards)
            
            policy_loss = []
            for reward, log_prob in zip(rewards, model.saved_log_probs):
                policy_loss.append(-log_prob*reward)
            policy_loss = (torch.cat(policy_loss)).sum()

            n += images.size(0)
            loss_value += float(policy_loss.item() * images.size(0))
            model.rewards = []
            model.saved_log_probs = []
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
    input_b = torch.cat(inputs, 0).cpu().numpy()
    mask_b = torch.cat(mask, 0).cpu().numpy()
    print("-------------------------------")
    print("Improvement:")
    print("-----------")
    eval_improvement(y_pred,y_pred_b,y_targ,args.data)
    print("-------------------------------")
    intuition_eval(input_b,y_pred,y_pred_b,mask_b,y_targ,args)
    print("-------------------------------")


def intuition_eval(input_boards,y_pred,y_pred_b,mask_b,y_targ,args):
    perception_path = 'outputs/perception/'+args.data+'/checkpoint_best.pth'
    perception = SequentialPerception()
    perception.to(args.gpu_id)
    perception.load_state_dict(torch.load(perception_path, map_location='cpu')) 
    input_boards = (torch.from_numpy(input_boards)).to(torch.float32).to(args.gpu_id)
    perception_boards = None
    with torch.no_grad():
        for i in input_boards:
            result_tensor = perception(i.unsqueeze(dim=0))
            if perception_boards is None:
                perception_boards = result_tensor
            else:
                perception_boards = torch.cat((perception_boards,result_tensor))
    perception_boards = perception_boards.clone().detach()
    perception_boards = torch.exp(perception_boards)
    perception_boards = perception_boards.argmax(dim=2)
    perception_boards = perception_boards.cpu().detach().numpy()
    n_errors_p_corrected_by_ns = []
    n_error_perception_masked = []
    n_error_ns_masked_i = []
    n_error_ns_masked_s = []
    
    for i in tqdm(range(len(y_pred))):
        perception_board = perception_boards[i].reshape(81)
        neuro_solver_board = np.array(y_pred_b[i]).reshape(81)
        mask_board = np.array(mask_b[i]).astype(int)
        pipeline_board = np.array(y_pred[i]).reshape(81)
        gt_board = np.array(y_targ[i]).reshape((81,10))
        gt_board = gt_board.argmax(axis=1)

        input_board = retrieve_hints_from_solution(gt_board,args.data).reshape(81)
        
        nep = 0 # number error perception
        nep_m = 0 # number error perception corrected by the mask
        nep_cnp = 0 # number error perception corrected by solverNN

        nens_s = 0 # number error neuro solver, in the solutions
        nens_i = 0 # number error neuro solver, in the inputs
        nens_m_s = 0 # number error solverNN, in the solutions, that are found by the mask
        nens_m_i = 0 # number error solverNN, in the inputs, that are found by the mask

        for j in range(81):
        
            if input_board[j]!= 0 and input_board[j]!=perception_board[j]: # error perception
                if pipeline_board[j] == gt_board[j]: # has been corrected
                    nep+=1
                    if neuro_solver_board[j] == gt_board[j]:
                        nep_cnp += 1
                    else:
                        if mask_board[j] == 0:
                            nep_m += 1
            
            assert nep == nep_cnp + nep_m

            if neuro_solver_board[j]!=gt_board[j]: # error solverNN 
                if input_board[j]== 0: # error in the solutions
                    nens_s+=1
                    if pipeline_board[j] == gt_board[j]: # has been corrected
                        nens_m_s+=1
                else: # error in the input
                    nens_i+=1
                    if pipeline_board[j] == gt_board[j]: # has been corrected
                        nens_m_i+=1

        if nep!=0:
            n_errors_p_corrected_by_ns.append(nep_cnp/nep)
            n_error_perception_masked.append(nep_m/nep)
        if nens_i!=0:
            n_error_ns_masked_i.append(nens_m_i/nens_i)
        if nens_s!=0:
            n_error_ns_masked_s.append(nens_m_s/nens_s)
        
    print(f"Num errors of the perception corrected by the SolverNN (avg): {statistics.mean(n_errors_p_corrected_by_ns)}%)")   
    print(f"Num errors of the perception corrected by the Mask-Predictor (avg): {statistics.mean(n_error_perception_masked)}%)")   
    print("\n---\n")
    print(f"Num errors of the solverNN in the input corrected (avg): {statistics.mean(n_error_ns_masked_i)}%)")   
    print(f"Num errors of the solverNN in the solutions corrected(avg): {statistics.mean(n_error_ns_masked_s)}%)")   


def eval_improvement(y_pred,y_pred_b,y_target,dataset_name=None):
    time_solutions = []
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
                    #print('alternative solution found')
                else:
                  if y_pred[i] == y_targ[i]:
                    num_corrected_boards += 1  
            else:
                if y_pred[i] == y_targ[i]:
                    num_corrected_boards += 1
        
        if y_pred[i] == y_targ[i]:
            num_correct += 1
            time_solutions.append(1)
        else:
            time_solutions.append(0)
            if y_pred_b[i] == y_targ[i]:
                num_wrong_boards += 1
        for j,k in zip(y_pred[i],y_targ[i]):
            if j==k:
                correct_cells+=1
        correct_cells/=81
        average_correct_cells.append(correct_cells)
    print(f"* Num. of correct solution boards from Neuro-Solver: {num_correct_b}/{num_total}-- ({(num_correct_b*100.)/(num_total):.2f}%)")
    print(f"* Num. of correct solution boards from NASR: {num_correct}/{num_total}-- ({((num_correct)*100.)/(num_total):.2f}%)")
    if (num_total-num_correct_b) >0 :
        print(f"* Num. of wrong solution boards (from Neuro-solver) \n  that have been corrected with NASR: {num_corrected_boards}/{num_total-num_correct_b} -- ({(num_corrected_boards*100.)/(num_total-num_correct_b):.2f}%)")
    if num_correct_b>0 :
        print(f"* Num. of correct solution boards (from Neuro-solver) \n  that have been corrupted with NASR: {num_wrong_boards}/{num_correct_b} -- ({num_wrong_boards*100./num_correct_b:.2f}%)")
    print(f"* Num correct cells (avg): {statistics.mean(average_correct_cells)}%")   
    return time_solutions


def main():
    parser = init_parser()
    args = parser.parse_args()
    # ---------------------------------- checkpoint_best
    #ckpt_path = f'outputs/rl/{args.data}/checkpoint_best_L.pth' # usually worse
    ckpt_path = f'outputs/rl/{args.data}/checkpoint_best_R.pth' 
    # ----------------------------------
    if os.path.isfile(args.noise_setting):
        with open(args.noise_setting) as f:
            noise_setting = json.load(f)
    else:
        noise_setting = {"noise_type": "awgn", "snr": -0.5}
    noise_setting = str(noise_setting).replace(' ', '').replace("'", "")
   
    if args.transform_data:
        test_dataset = SudokuDataset_RL(args.data,'-test',transform=args.transform_data,t_param=args.transform_data_param)
    else:
        test_dataset = SudokuDataset_RL(args.data,'-test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)

    # Model
    model = get_model(block_len=args.block_len)
    if args.nasr == 'rl':
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))  #to check preformance of the RL
    else:
        assert args.nasr == 'pretrained', f'{args.nasr} not supported, choose between pretrained and rl'
        model.load_pretrained_models(args.data) #to check preformance of the pretrained pipeline without RL
    model.to(args.gpu_id)

    # Main loop
    print("Beginning testing")
    validate(test_loader, model, args)
 

if __name__ == '__main__':

    main()