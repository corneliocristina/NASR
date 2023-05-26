import os
import json
import argparse
from time import time
import numpy as np
import torch
import torch.nn as nn
from models.perception import SequentialPerception
import random
from datasets import  SudokuDataset_RL
from sudoku_solver.board import Board
from operator import add
from tqdm import tqdm


def init_parser():
    parser = argparse.ArgumentParser(description='Baseline for NASR')
    parser.add_argument('--gpu-id', default=1, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='Batch size')
    parser.add_argument('--data', type=str, default='big_kaggle',
                        help='dataset name between [big_kaggle, minimal_17, multiple_sol, satnet]')
    parser.add_argument('--solver', type=str, default='prolog',
                        help='symbolic solver to use. available options are prolog and backtrack')
    return parser


def save_time_in_file(time_perception, time_solver, time_solutions, dataset):
    dict = {}
    time_baseline = list(map(add,time_perception, time_solver))
    dict["baseline"] = time_baseline
    dict["baseline_solutions"] = time_solutions
    
    os.makedirs(f'outputs/baseline/{dataset}', exist_ok=True)
    output_file_name = f"outputs/baseline/{dataset}/time_baseline.json"
    output_file = open(output_file_name,"w")
    json.dump(dict,output_file)
    output_file.close()


def validate(val_loader, model, args, epoch=None, time_begin=None):
    model.eval()
    labels = []
    solutions = []
    time_solver = []
    time_perception = []
    time_solutions = []
    time_begin = time()
    with torch.no_grad():
        for _, (images, target) in tqdm(enumerate(val_loader)):
            images = images.to(args.gpu_id)
            target = target.to(args.gpu_id)
            target = target.argmax(dim=2)
            t1 = time()
            boards = model(images)
            time_perception.append(time() - t1)
            boards = torch.exp(boards)
            boards = boards.argmax(dim=2)
            for idx in range(len(boards)):
                labels.append(target[idx])
                board_to_solver = Board(boards[idx].reshape((9,9)).cpu().int())
                
                t2 = time()
                try: 
                    solver_success = board_to_solver.solve(solver =args.solver) 
                except StopIteration:
                    solver_success = False
                time_solver.append(time() - t2)
                
                final_solution = board_to_solver.board.reshape(81,)
                if (not solver_success):
                    final_solution = boards[idx].cpu()
                solutions.append(final_solution)
    
    tot_time = time() - time_begin

    solution_list = [l.tolist() for l in solutions]
    target_list = [l.tolist() for l in labels]
    solution_list = [list(map(int,i)) for i in solution_list]
    target_list = [list(map(int,i)) for i in target_list]
    count_tot = len(solution_list)
    count_correct = 0
    for i in range(len(solution_list)):
        if solution_list[i] == target_list[i]:
            count_correct += 1
            time_solutions.append(1)
        else:
            time_solutions.append(0)

    print(f"Num. of correct boards from perception + symbolic solver: {count_correct}/{count_tot}-- ({(count_correct*100.)/(count_tot):.2f}%)")
    print(f"total time for {len(val_loader)} images: {tot_time:.4f}")

    return time_perception, time_solver , time_solutions
        



def main():
    parser = init_parser()
    args = parser.parse_args()
    # ----------------------------------
    args.ckpt = f'outputs/perception/{args.data}/checkpoint_best.pth' 
    
    test_dataset = SudokuDataset_RL(args.data,'-test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)

    # Model
    model = SequentialPerception()
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model.to(args.gpu_id)

    # Main loop
    print("Beginning testing")
    time_perception, time_solver, time_solutions = validate(test_loader, model, args)
    save_time_in_file(time_perception, time_solver, time_solutions, args.data)
 

if __name__ == '__main__':
    
    main()