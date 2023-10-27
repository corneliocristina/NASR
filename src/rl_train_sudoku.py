import os
import argparse
from token import TILDE
from numpy.testing._private.utils import requires_memory
from torch.functional import atleast_2d
import torch.nn as nn
import torch
import random
from torch.autograd import Variable
import math
from sudoku_solver.board import Board
import json
from time import time
from datasets import SudokuDataset_RL
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import print_loss_graph_from_details_file_rl, print_loss_graph_from_file,print_loss_graph_from_file_rl
from torch.distributions import Categorical
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
from models.transformer_sudoku import get_model
try:
    from pyswip import Prolog
except Exception:
    print('-->> Prolog not installed')
SOLVER_TIME_OUT = 0.5

def init_parser():
    parser = argparse.ArgumentParser(description='Quick training script')

    # General args
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    parser.add_argument('--solver', type=str, default='prolog',
                        help='symbolic solver to use. available options prolog and backtrack')
                        
    # Model args
    parser.add_argument('--block-len', default=81, type=int,
                        help='board size')
    parser.add_argument('--data', type=str, default='big_kaggle',
                        help='dataset name between big_kaggle, minimal_17, multiple_sol and satnet')
    parser.add_argument('--noise-setting', default='xxx/yyy.json', type=str,
                        help='Json file of noise setting (dict)')
    parser.add_argument('--train-only-mask', default = False, type = bool,
                        help='If true, use RL to train only the mask predictor')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=2000, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--warmup', default=10, type=int, 
                        help='number of warmup epochs')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)', dest='batch_size')
    parser.add_argument('--lr', default=0.0005, type=float, 
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=3e-2, type=float, 
                        help='weight decay (default: 3e-2)')
    parser.add_argument('--clip-grad-norm', default=0., type=float, 
                        help='gradient norm clipping (default: 0 (disabled))')
    parser.add_argument('--disable-cos', action='store_true',
                        help='disable cosine lr schedule')
    return parser

def compute_reward(solution_board,final_solution,ground_truth_board):
    solution_board = list(map(int,solution_board.tolist()))
    final_solution = list(map(int, final_solution.tolist() ))
    ground_truth_board = list(map(int,ground_truth_board.tolist()))
    if final_solution == ground_truth_board:
        reward = 10
    else:
        reward = 0
    partial_reward = 0.0
    for i,j in zip(final_solution,ground_truth_board):
        if i == j:
            partial_reward+=1
    reward += partial_reward/81
    return reward


def compute_reward2(masking_board,final_solution,ground_truth_sol,time_solver,solver_success):
    solution_board = list(map(int,solution_board.tolist()))
    final_solution = list(map(int, final_solution.tolist() ))
    ground_truth_board = list(map(int,ground_truth_board.tolist()))
    # HYPERPARAMETERS
    # weights for the different components of the reward
    # (time_reward, mask_reward, cross_entropy_reward, solver_reward, is_correct)
    (a,b,c,d,e) = (1/4,1/4,1/4,1/4,10) 
    # solver_reward either 0 or 1
    solver_reward = 1 
    if not solver_success:
       solver_reward = 0
    # time_reward is in [0,1]
    time_reward = (SOLVER_TIME_OUT - time_solver)/SOLVER_TIME_OUT
    # mask_reward is (81 - num_mask_cells)/81 the proportion of non masked cells
    # mask_reward is in [0,1]
    num_mask_cells = 0
    mask_cells = np.nonzero(masking_board)
    if len(mask_cells)>0 and mask_cells:
       num_mask_cells = len(np.nonzero(masking_board)[0])
    mask_reward = (81 - num_mask_cells)/81
    # cross_entropy_reward
    final_solution_mc = np.zeros((81,9),dtype=int)
    ground_truth_sol_mc = np.zeros((81,9),dtype=int)
    for i in range(len(final_solution)):
       final_solution_mc[i][int(final_solution[i]-1)]=1
       ground_truth_sol_mc[i][int(ground_truth_sol[i]-1)]=1
    loss = nn.BCEWithLogitsLoss()
    cross_entropy_reward = 1- (loss(torch.from_numpy(final_solution_mc.astype('float32')), torch.from_numpy(ground_truth_sol_mc.astype('float32'))).sigmoid())
    is_correct = 0
    if final_solution == ground_truth_sol:
        is_correct = 1
    # reward is in [0,11]
    reward = (a*time_reward + b*mask_reward + c*float(cross_entropy_reward) + d*solver_reward + e*is_correct) 
    return reward


def compute_reward3(solution_board,final_solution,ground_truth_board):
    # reward 1 if completely corrected board or 0 otherwise
    solution_board = list(map(int,solution_board.tolist()))
    final_solution = list(map(int, final_solution.tolist() ))
    ground_truth_board = list(map(int,ground_truth_board.tolist()))
    if final_solution == ground_truth_board:
        reward = 1
    else:
        reward = 0
    return reward

def compute_reward4(solution_board,final_solution,ground_truth_board):
    # reward 1 if completely corrected board or -1 otherwise
    solution_board = list(map(int,solution_board.tolist()))
    final_solution = list(map(int, final_solution.tolist() ))
    ground_truth_board = list(map(int,ground_truth_board.tolist()))
    if final_solution == ground_truth_board:
        reward = 1
    else:
        reward = -1
    return reward


def final_output(model,ground_truth_sol,solution_boards,masking_boards,args):
    ground_truth_boards = torch.argmax(ground_truth_sol,dim=2)
    solution_boards_new = torch.argmax(solution_boards,dim=2)+1
    
    config = 'sigmoid_bernoulli' # best option
    # between sigmoid_bernoulli gumble_round sigmoid_round 
    if config == 'sigmoid_bernoulli':
        masking_prob = masking_boards.sigmoid()
        b = Bernoulli(masking_prob)
        sampled_mask_boards = b.sample()
        model.saved_log_probs = b.log_prob(sampled_mask_boards)
        sampled_mask_boards = np.array(sampled_mask_boards.cpu()).reshape(masking_prob.shape)
        cleaned_boards = np.multiply(solution_boards_new.cpu(),sampled_mask_boards)
    elif config == 'sigmoid_round':
        masking_prob = masking_boards.sigmoid()
        b= Bernoulli(masking_prob)
        sampled_mask_boards = torch.round(masking_prob)
        model.saved_log_probs = b.log_prob(sampled_mask_boards)
        cleaned_boards = np.multiply(solution_boards_new.cpu(),sampled_mask_boards.cpu())
    else: 
        assert(config == 'gumble_round')
        masking_prob = F.gumbel_softmax(masking_boards)
        b = Bernoulli(masking_prob)
        sampled_mask_boards = torch.round(masking_prob)
        model.saved_log_probs = b.log_prob(sampled_mask_boards)
        cleaned_boards = np.multiply(solution_boards_new.cpu(),sampled_mask_boards.cpu())
    
    final_boards = []
    if args.solver == 'prolog':
        prolog_instance = Prolog()
        prolog_instance.consult("src/sudoku_solver/sudoku_prolog.pl")
    for i in range(len(cleaned_boards)):
        board_to_solver = Board(cleaned_boards[i].reshape((9,9)).int())
        try:
            if args.solver == 'prolog':
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



def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def validate(val_loader, model, args, epoch=None, time_begin=None):
    model.eval()
    loss_value = 0
    reward_value = 0
    n = 0
    eps = np.finfo(np.float32).eps.item()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(args.gpu_id)
            target = target.to(args.gpu_id)

            solution_boards, masking_boards = model(images)
            final_output(model,target,solution_boards,masking_boards,args) # this populates model.rewards 
            rewards = np.array(model.rewards)
            rewards_mean = rewards.mean()
            reward_value += float(rewards_mean * images.size(0))
            rewards = (rewards - rewards.mean())/(rewards.std() + eps)
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
                print(f'[rl][Epoch {epoch}][Val][{i}] \t AvgLoss: {avg_loss:.4f}  \t AvgRewards: {rewards_mean:.4f}')
    
    avg_reward = (reward_value/n)      
    avg_loss = (loss_value / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'----[rl][Epoch {epoch}] \t \t AvgLoss {avg_loss:.4f} \t \t AvgReward {avg_reward:.4f} \t \t Time: {total_mins:.2f} ')

    return avg_loss, rewards_mean
    
    
def train(train_loader, model, optimizer, epoch, args):
    model.train()
    loss_value = 0
    n = 0
    
    # to train only the mask predictor
    if args.train_only_mask == True:
        # to not train nn_solver
        model.nn_solver.requires_grad_(requires_grad = False)
        for param in model.nn_solver.parameters():
            param.requires_grad = False
        # to not train perception
        model.perception.requires_grad_(requires_grad = False)
        for param in model.perception.parameters():
            param.requires_grad = False

    eps = np.finfo(np.float32).eps.item()
    for i, (images, target) in enumerate(train_loader):        
        images = images.to(args.gpu_id)
        target = target.to(args.gpu_id)
        solution_boards, masking_boards = model(images)
        final_output(model,target,solution_boards,masking_boards,args) # this populates model.rewards 
        rewards = np.array(model.rewards)
        rewards_mean = rewards.mean()
        rewards = (rewards - rewards.mean())/(rewards.std() + eps)
        policy_loss = []
        for reward, log_prob in zip(rewards, model.saved_log_probs):
            policy_loss.append(-log_prob*reward)
        optimizer.zero_grad()
        
        criterion = nn.BCEWithLogitsLoss()
        loss_nn_solver = criterion(solution_boards, target[:,:,1:])
        #policy_loss = (torch.cat(policy_loss)).sum() + loss_nn_solver
        policy_loss = (torch.cat(policy_loss)).sum()

        n += images.size(0)
        loss_value += float(policy_loss.item() * images.size(0))
        policy_loss.backward()
       
        #if args.clip_grad_norm > 0:
        #    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

        optimizer.step()
             
        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss = (loss_value / n)
            print(f'[rl][Epoch {epoch}][Train][{i}/{len(train_loader)}] \t AvgLoss: {avg_loss:.4f} \t AvgRewards: {rewards_mean:.4f}')
            stats2 = {'epoch': epoch, 'train': i, 'avr_train_loss': avg_loss, 
                    'avr_train_reward': rewards_mean}
            with open(f"outputs/rl/{args.data}/detail_log.txt", "a") as f:
                f.write(json.dumps(stats2) + "\n")
        model.rewards = []
        model.saved_log_probs = []
        torch.cuda.empty_cache()

    avg_loss = (loss_value / n)
    return avg_loss, rewards_mean


def main():
    parser = init_parser()
    args = parser.parse_args()

    train_dataset = SudokuDataset_RL(args.data,'-train')
    val_dataset = SudokuDataset_RL(args.data,'-valid')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)
    
    # Model 
    model = get_model(block_len=args.block_len)
    model.load_pretrained_models(args.data)
    model.to(args.gpu_id)

    # load pre_trained models
    if args.train_only_mask == True:
        # only training the mask network
        optimizer = torch.optim.AdamW(model.mask_nn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # to exclude only the perception from the training
        #optimizer = torch.optim.AdamW(model.mask_nn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #optimizer = torch.optim.AdamW(model.nn_solver.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Main loop 
    print("Beginning training")
    ckpt_path = os.path.join('outputs', 'rl/'+args.data)
    os.makedirs(ckpt_path, exist_ok=True)
    best_loss = None
    best_reward = None
    time_begin = time()
    with open(f"{ckpt_path}/log.txt", 'w'): pass
    with open(f"{ckpt_path}/detail_log.txt", 'w'): pass
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)

        train_loss, train_rewards = train(train_loader, model, optimizer, epoch, args)
        loss, valid_rewards = validate(val_loader, model, args, epoch=epoch, time_begin=time_begin)
        
        if best_reward is None or valid_rewards > best_reward :
            best_reward = valid_rewards
            torch.save(model.state_dict(), f'{ckpt_path}/checkpoint_best_R.pth')
        
        if best_loss is None or loss < best_loss :
            best_loss = loss
            torch.save(model.state_dict(), f'{ckpt_path}/checkpoint_best_L.pth')

        stats = {'epoch': epoch, 'lr': lr, 'train_loss': train_loss, 
                    'val_loss': loss, 'best_loss': best_loss , 
                    'train_rewards': train_rewards, 'valid_rewards': valid_rewards}
        with open(f"{ckpt_path}/log.txt", "a") as f:
            f.write(json.dumps(stats) + "\n")

    total_mins = (time() - time_begin) / 60
    print(f'[rl] finished in {total_mins:.2f} minutes, '
          f'best loss: {best_loss:.6f}, '
          f'final loss: {loss:.6f}')
    torch.save(model.state_dict(), f'{ckpt_path}/checkpoint_last.pth')
    print_loss_graph_from_file_rl(f"{ckpt_path}/log.txt",f"{ckpt_path}/loss_rl_sudoku")
    #print_loss_graph_from_details_file_rl('r',f"{ckpt_path}/detail_log.txt",f"{ckpt_path}/detail_reward") 
    #print_loss_graph_from_details_file_rl('l',f"{ckpt_path}/detail_log.txt",f"{ckpt_path}/detail_loss") 


if __name__ == '__main__':

    main()

    
    # TO PRINT LOSS and REWARDS from LOGs
    # Datasets available: [big_kaggle, minimal_17, multiple_sol, satnet].

    # print loss
    #print_loss_graph_from_file("outputs/rl/big_kaggle/log.txt","outputs/rl/big_kaggle/loss_rl_sudoku") 
    
    #print rewards
    #print_loss_graph_from_file_rl("outputs/rl/big_kaggle/log.txt","outputs/rl/big_kaggle/loss_rl_sudoku_rl") 
    
    #print details loss
    #print_loss_graph_from_details_file_rl('l',"outputs/rl/big_kaggle/detail_log.txt","outputs/rl/big_kaggle/detail_loss_rl_sudoku") 
    
    #print details rewards
    #print_loss_graph_from_details_file_rl('r',"outputs/rl/big_kaggle/detail_log.txt","outputs/rl/big_kaggle/detail_reward_rl_sudoku") 
