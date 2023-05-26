import torch
import torch.nn.functional as F
import argparse
import argparse
import torch
import random
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models.perception import SequentialPerception
from datasets import SudokuDataset_Perception
from time import time
import os,json
import numpy as np
from src.utils.utils import print_loss_graph_from_file


def init_parser():
    parser = argparse.ArgumentParser(description='Perception Module for NASR')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data', type=str, default='big_kaggle',
                        help='dataset name between [big_kaggle, minimal_17, multiple_sol, satnet]')
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    args = parser.parse_args()
    return args


def train(args, model, device, train_loader, optimizer, epoch ):
    model.train()
    loss_val = 0
    n = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)

        a = pred.size(dim=0)
        b = pred.size(dim=1)
        pred = pred.view(a*b,-1)
        target = target.view(a*b,-1).argmax(dim=1).long()

        loss = F.nll_loss(pred, target) 
        loss_val += float(loss.item() * data.size(0))
        n += data.size(0)
        loss.backward()
        optimizer.step()
        if args.print_freq >= 0 and batch_idx % args.print_freq == 0:
            avg_loss = (loss_val / n)
            print(f'[{args.name}][Epoch {epoch}][Train][{batch_idx}] \t AvgLoss: {avg_loss:.4f}')
    avg_loss = (loss_val / n)
    return avg_loss 

def validate(args, model, device, val_loader, epoch, time_begin):
    model.eval()
    loss_val = 0
    n = 0
    predictions= []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            preds = model(data)
            predictions.append(preds)
            targets.append(target)

            a = preds.size(dim=0)
            b = preds.size(dim=1)
            preds = preds.view(a*b,-1)
            target = target.view(a*b,-1).argmax(dim=1).long()
            loss = F.nll_loss(preds, target)  
            n += data.size(0)
            loss_val += float(loss.item() * data.size(0))
            if args.print_freq >= 0 and batch_idx % args.print_freq == 0 :
                avg_loss = (loss_val / n)
                print(f'[{args.name}][Epoch {epoch}][Val][{batch_idx}] \t AvgLoss: {avg_loss:.4f}')
    avg_loss = (loss_val / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'[{args.name}][Epoch {epoch}] \t \t AvgLoss {avg_loss:.4f} \t \t Time: {total_mins:.2f}')
    
    predictions = torch.cat(predictions, 0).sigmoid().cpu().numpy()
    targets = torch.cat(targets, 0).sigmoid().cpu().numpy()

    return avg_loss

def eval(predictions,labels):
    f_correct = []
    for preds,target in zip(predictions,labels):
        correct = 0
        for cell_id in range(81):
            cell_p = preds[cell_id].argmax()
            cell_t = target[cell_id].argmax()
            if cell_p == cell_t:
                correct+= 1
        f_correct.append(correct/81.)
    print(f"Total accuracy: {np.mean(f_correct):.4f}")

def main():
    args = init_parser()
    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device(args.gpu_id if use_cuda else "cpu")

    train_dataset = SudokuDataset_Perception(args.data,'-train')
    val_dataset = SudokuDataset_Perception(args.data,'-valid')
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    # model
    model = SequentialPerception()
    model.to(device)

    best_loss = None
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    #main loop
    print("Beginning training")
    args.name = os.path.join('outputs', 'perception/'+args.data)
    os.makedirs(args.name, exist_ok=True)
    time_begin = time()
    
    with open(f"{args.name}/log.txt", 'w'): pass
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        loss = validate(args, model, device, val_loader, epoch, time_begin)
        scheduler.step()

        if best_loss is None or loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), f'{args.name}/checkpoint_best.pth')

        stats = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': loss, 'best_loss': best_loss}
        with open(f"{args.name}/log.txt", "a") as f:
            f.write(json.dumps(stats) + "\n")

    total_mins = (time() - time_begin) / 60
    print(f'[{args.name}] finished in {total_mins:.2f} minutes, '
          f'best loss: {best_loss:.6f}, '
          f'final loss: {loss:.6f}')
    torch.save(model.state_dict(), f'{args.name}/checkpoint_last.pth')
    print_loss_graph_from_file(f"{args.name}/log.txt",f"{args.name}/loss")


if __name__ == '__main__':

    main()
