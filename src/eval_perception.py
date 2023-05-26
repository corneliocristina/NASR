import torch
import torch.nn.functional as F
import argparse
import argparse
import torch
from models.perception import SequentialPerception
from datasets import SudokuDataset_Perception
from time import time
import numpy as np
import random


def init_parser():
    parser = argparse.ArgumentParser(description='Perception Module for NASR')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--data', type=str, default='big_kaggle',
                        help='dataset name between [big_kaggle, minimal_17, multiple_sol, satnet]')
    parser.add_argument('--gpu-id', default=0, type=int, help='preferred gpu id')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    args = parser.parse_args()
    return args


def validate(args, model, device, test_loader):
    model.eval()
    loss_val = 0
    n = 0
    predictions= []
    targets = []
    time_begin = time()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            preds = model(data)
            preds = torch.exp(preds)
            predictions.append(preds)
            targets.append(target)
            a = preds.size(dim=0)
            b = preds.size(dim=1)
            preds = preds.view(a*b,-1)
            target = target.view(a*b,-1).argmax(dim=1).long()
            loss = F.nll_loss(preds, target)  
            n += data.size(0)
            loss_val += float(loss.item() * data.size(0))
            if args.print_freq >= 0 and batch_idx % args.print_freq == 0:
                avg_loss = (loss_val / n)
                print(f'[Test][{batch_idx}/{len(test_loader)}] \t AvgLoss: {avg_loss:.4f}')
    avg_loss = (loss_val / n)
    total_mins = (time() - time_begin) / 60
    print(f'AvgLoss {avg_loss:.4f} \t \t Time: {total_mins:.2f}')
    predictions = torch.cat(predictions, 0).cpu().numpy()
    targets = torch.cat(targets, 0).cpu().numpy()
    eval(predictions,targets)

def eval(predictions,labels):
    f_correct = []
    entire_board = 0
    for preds,target in zip(predictions,labels):
        correct = 0
        for cell_id in range(81):
            cell_p = preds[cell_id].argmax()
            cell_t = target[cell_id].argmax()
            if cell_p == cell_t:
                correct += 1
        if correct == 81:
            entire_board += 1
        f_correct.append(correct/81.)
    print(f"Accuracy Perception: {np.mean(f_correct)}")
    print(f"Entire boards correct: {entire_board}/{len(predictions)} ({entire_board*100/len(predictions):.2f}%)")


def main():
    args = init_parser()
    ckpt_path = f'outputs/perception/{args.data}/checkpoint_best.pth'

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.gpu_id if use_cuda else "cpu")
    
    test_dataset = SudokuDataset_Perception(args.data,'-test')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    # model
    model = SequentialPerception()
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.to(device)

    #main loop
    print("Beginning testing")
    validate(args, model, device, test_loader)

if __name__ == '__main__':

    main()