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

from tqdm import tqdm
from operator import add

from src.rl_eval_sudoku import eval_improvement
from src.rl_eval_sudoku import final_output

def init_parser():
    parser = argparse.ArgumentParser(description='Quick testing script')

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
    parser.add_argument('--data', type=str, default='satnet',
                        help='dataset name between big_kaggle, minimal_17, multiple_sol and satnet')
    parser.add_argument('--noise-setting', default='xxx/yyy.json', type=str,
                        help='Json file of noise setting (dict)')
    parser.add_argument('--solver', type=str, default='prolog',
                        help='symbolic solver to use. available options prolog and backtrack')

    # Model args
    parser.add_argument('--block-len', default=81, type=int,
                        help='board size')
    parser.add_argument('--code-rate', default=2, type=int,
                        help='Code rate')

    return parser


def save_time_in_file(time_neuro_solver, time_solver , time_solutions, dataset, flag):
    dict = {}
    time_pipeline = list(map(add,time_neuro_solver, time_solver))
    dict["pipeline"] = time_pipeline
    dict["pipeline_solutions"] = time_solutions
    
    output_file_name = f"outputs/stats/{dataset}/time_pipeline-{flag}.json"
    output_file = open(output_file_name,"w")
    json.dump(dict,output_file)
    output_file.close()


def validate(val_loader, model, args, epoch=None, time_begin=None):
    model.eval()
    time_solver = []
    time_neuro_solver = []
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

            t1 = time()
            solution_boards, masking_boards = model(images, nasr=args.nasr)
            time_neuro_solver.append(time() - t1)

            pred_final, time_solver_tmp = final_output(model,target,solution_boards,masking_boards,args)
            time_solver.append(time_solver_tmp)
            rewards = np.array(model.rewards)
            rewards = (rewards - rewards.mean())/(rewards.std() + eps)

            preds.append(pred_final)
            labels.append(target)
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
    print("-------------------------------")
    print("Improvement:")
    print("-----------")
    time_solutions = eval_improvement(y_pred,y_pred_b,y_targ)
    print("-------------------------------")
    return time_neuro_solver, time_solver, time_solutions


def main(args):
    
    # ---------------------------------- checkpoint_best
    #args.ckpt = f'outputs/rl/{args.data}/checkpoint_best_L.pth' # usually worse
    args.ckpt = f'outputs/rl/{args.data}/checkpoint_best_R.pth' 
    # ----------------------------------
    if os.path.isfile(args.noise_setting):
        with open(args.noise_setting) as f:
            noise_setting = json.load(f)
    else:
        noise_setting = {"noise_type": "awgn", "snr": -0.5}
    noise_setting = str(noise_setting).replace(' ', '').replace("'", "")
    
   
    test_dataset = SudokuDataset_RL(args.data,'-test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)

    # Model
    model = get_model(block_len=args.block_len)
    if args.nasr == 'rl':
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))  #to check preformance of the RL
    else:
        assert args.nasr == 'pretrained', f'{args.nasr} not supported, choose between pretrained and rl'
        model.load_pretrained_models(args.data) #to check preformance of the pretrained pipeline without RL
    model.to(args.gpu_id)

    # Main loop
    print("Beginning testing")
    time_neuro_solver, time_solver, time_solutions= validate(test_loader, model, args)
    save_time_in_file(time_neuro_solver, time_solver , time_solutions, args.data, args.nasr)
    


def compute_time_limit_min_max(dataset): 
    assert dataset in ['big_kaggle', 'minimal_17', 'multiple_sol', 'satnet']   
    max_value = 0
    min_value = 100000

    path = f"outputs/stats/{dataset}/"
    
    # baseline
    file_name = path+"time_baseline.json"  
    with open(file_name,'r') as file:
        dictt = json.load(file)
        values = dictt["baseline"]
        max_v = max(values)
        min_v = min(values)
        if max_v > max_value:
            max_value = max_v
        if min_v < min_value:
            min_value = min_v
        

    # rl pipeline
    file_name = path+"time_pipeline-rl.json"
    with open(file_name,'r') as file:
        dictt = json.load(file)
        values = (dictt["pipeline"])
        max_v = max(values)
        min_v = min(values)
        if max_v > max_value:
            max_value = max_v
        if min_v < min_value:
            min_value = min_v
    
    # pretrained pipeline
    file_name = path+"time_pipeline-pretrained.json"
    with open(file_name,'r') as file:
        dictt = json.load(file)
        values = (dictt["pipeline"])
        max_v = max(values)
        min_v = min(values)
        if max_v > max_value:
            max_value = max_v
        if min_v < min_value:
            min_value = min_v

    # satnet
    file_name = path+"time_satnet_baseline.json"
    with open(file_name,'r') as file:
        dictt = json.load(file)
        values = (dictt["satnet_baseline"])
        max_v = max(values)
        min_v = min(values)
        if max_v > max_value:
            max_value = max_v
        if min_v < min_value:
            min_value = min_v

    # satnet + our
    file_name = path+"time_satnet_our.json"
    with open(file_name,'r') as file:
        dictt = json.load(file)
        values = (dictt["satnet_our"])
        max_v = max(values)
        min_v = min(values)
        if max_v > max_value:
            max_value = max_v
        if min_v < min_value:
            min_value = min_v
    
    print(f'Max value for {dataset}: {max_value}\n')
    print(f'Min value for {dataset}: {min_value}')


def compute_scores(values,values_solutions,min_val,max_val,num_steps):
    num_boards = len(values)
    step = (max_val - min_val)/num_steps
    scores = []
    for i in range(num_steps+1):
        timeout = min_val+ i*step
        score = 0
        for i in range(num_boards):
            if values[i] < timeout and values_solutions[i] == 1:
                score+=1
        scores.append(score/num_boards)
    return scores


def stats_time(dataset, performance, min_max_values): 
    assert dataset in ['big_kaggle', 'minimal_17', 'multiple_sol', 'satnet']       
    min_val = min_max_values[dataset][0]
    max_val = min_max_values[dataset][1]
    num_steps = 100
    
    pareto_front = {}
    time_limit_solutions = {}
    path = f"outputs/stats/{dataset}/"
    
    # baseline
    file_name = path+"time_baseline.json"  
    with open(file_name,'r') as file:
        dictt = json.load(file)
        values = dictt["baseline"]
        values_solutions = dictt["baseline_solutions"]
        value = sum(values)/len(values) 
    pareto_front['baseline'] = [value,performance['baseline'][dataset]]
    time_limit_solutions['baseline'] = compute_scores(values,values_solutions,min_val,max_val,num_steps)

    # rl pipeline
    file_name = path+"time_pipeline-rl.json"
    with open(file_name,'r') as file:
        dictt = json.load(file)
        values = (dictt["pipeline"])
        values_solutions = (dictt["pipeline_solutions"])
        value = sum(values)/len(values) 
    pareto_front['pipeline_rl'] = [value,performance['pipeline_rl'][dataset]]
    time_limit_solutions['pipeline_rl'] = compute_scores(values,values_solutions,min_val,max_val,num_steps)
    
    # pretrained pipeline
    file_name = path+"time_pipeline-pretrained.json"
    with open(file_name,'r') as file:
        dictt = json.load(file)
        values = (dictt["pipeline"])
        values_solutions = (dictt["pipeline_solutions"])
        value = sum(values)/len(values)
    pareto_front['pipeline_pretrained'] = [value,performance['pipeline_pretrained'][dataset]] 
    time_limit_solutions['pipeline_pretrained'] = compute_scores(values,values_solutions,min_val,max_val,num_steps)

    # satnet
    file_name = path+"time_satnet_baseline.json"
    with open(file_name,'r') as file:
        dictt = json.load(file)
        values = (dictt["satnet_baseline"])
        values_solutions = (dictt["satnet_baseline_solutions"])
        value = sum(values)/len(values) 
    pareto_front['satnet_baseline'] = [value,performance['satnet_baseline'][dataset]]
    time_limit_solutions['satnet_baseline'] = compute_scores(values,values_solutions,min_val,max_val,num_steps)

    # satnet + our
    file_name = path+"time_satnet_our.json"
    with open(file_name,'r') as file:
        dictt = json.load(file)
        values = (dictt["satnet_our"])
        values_solutions = (dictt["satnet_our_solutions"])
        value = sum(values)/len(values) 
    pareto_front['satnet_our'] = [value,performance['satnet_our'][dataset]]
    time_limit_solutions['satnet_our'] = compute_scores(values,values_solutions,min_val,max_val,num_steps)
    
    # neurASP 
    value = 10
    pareto_front['neurASP'] = [value,performance['neurASP'][dataset]]
    
    output_file_name = f"outputs/stats/{dataset}/pareto.json"
    output_file = open(output_file_name,"w")
    json.dump(pareto_front,output_file)
    output_file.close()  

    output_file_name = f"outputs/stats/{dataset}/time_limit_solutions.json"
    output_file = open(output_file_name,"w")
    json.dump(time_limit_solutions,output_file)
    output_file.close()

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    #main(args)  
      
    # ------------
    performance = {}
    performance['baseline'] = {'big_kaggle': 0.7456, 'minimal_17': 0.8770, 'multiple_sol': 0.657, 'satnet': 0.564}
    performance['pipeline_rl'] = {'big_kaggle': 0.8424, 'minimal_17': 0.8700, 'multiple_sol': 0.730, 'satnet': 0.822}
    performance['pipeline_pretrained'] = {'big_kaggle': 0.8002, 'minimal_17': 0.0159, 'multiple_sol': 0.600, 'satnet': 0.764}
    performance['satnet_baseline'] = {'big_kaggle': 0.6344, 'minimal_17': 0.0000, 'multiple_sol': 0.0000, 'satnet': 0.601}
    performance['satnet_our'] = {'big_kaggle': 0.6905, 'minimal_17': 0.0002, 'multiple_sol': 0.242, 'satnet': 0.814}
    performance['neurASP'] = {'big_kaggle': 0.0, 'minimal_17': 0.89, 'multiple_sol': 0.0, 'satnet': 0.0}
    performance['neuro_solver'] = {'big_kaggle': 0.4703, 'minimal_17': 0.0000, 'multiple_sol': 0.288, 'satnet': 0.144}
    performance['ablation'] = {'big_kaggle': 0.7311, 'minimal_17': 0.6699, 'multiple_sol': 0.556, 'satnet': 0.428}
    # ------------
    min_max_values = {}
    min_max_values['big_kaggle'] = [ 0.00047707557678222656, 0.8809816837310791]
    min_max_values['minimal_17'] = [0.0005729198455810547 , 0.9443631172180176]
    min_max_values['multiple_sol'] = [ 0.0005402565002441406, 0.8835620880126953]
    min_max_values['satnet'] = [ 0.0004639625549316406, 0.9783685207366943]

    
    if args.data == 'big_kaggle':
        compute_time_limit_min_max('big_kaggle')
        stats_time('big_kaggle', performance, min_max_values)

    if args.data == 'minimal_17':
        compute_time_limit_min_max('minimal_17')
        stats_time('minimal_17', performance, min_max_values)
    
    if args.data == 'multiple_sol':
        compute_time_limit_min_max('multiple_sol')
        stats_time('multiple_sol', performance, min_max_values)
    
    if args.data == 'satnet':
        compute_time_limit_min_max('satnet')
        stats_time('satnet', performance, min_max_values)
