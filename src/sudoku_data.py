from genericpath import isdir
from tabnanny import check
import numpy as np
import time
import statistics
import matplotlib.pyplot as plt
import mnist
import os
from tqdm import tqdm
from sudoku_solver.board import Board
import torch
import h5py
import random
import shutil
import multiprocessing as mp
from joblib import Parallel, delayed
import argparse

try:
    from pyswip import Prolog
except Exception:
    print('-->> Prolog not installed')
    
def init_parser():
    parser = argparse.ArgumentParser(description='Solver-NN and Mask-Predictor Module for NASR')
    # General args
    parser.add_argument('--solver', type=str, default='default',
                        help='symbolic solver to use. available options are default, prolog and backtrack')
    parser.add_argument('--min_noise', type=int, default=None,
                        help='min amount of noise in the data generation')
    parser.add_argument('--max_noise', type=int, default=None,
                        help='max amount of noise in the data generation')
    parser.add_argument('--dataset', type=str, default='all',
                        help='dataset to generate between [multiple_sol,minimal_17,big_kaggle,satnet_data,all]')
    return parser



def process_satnet_data(args, noise_input=False):
    print('----------------------------------------------')
    print('Processing dataset satnet')
    print('----------------------------------------------')
    # for training our pipeline [0-20]
    # for satnet+mask use noise [10-30]
    data_name = 'satnet'
    min_noise=0
    max_noise=20
    if args.min_noise and args.max_noise and args.min_noise < args.max_noise:
        min_noise=args.min_noise
        max_noise=args.max_noise

    if not isdir('data/'+data_name):
        os.mkdir('data/'+data_name)
    
    final_data, final_images, final_solutions = format_conversion_satnet()
    noise_data, mask_data = mask_data_generation_satnet(min_noise,max_noise,final_data, final_solutions)
    train_val_test_split_satnet(data_name, final_data, final_images, final_solutions, noise_data, mask_data)


def format_conversion_satnet():
    with open('data/original_data/features.pt', 'rb') as f:
            X_in = torch.load(f)
    with open('data/original_data/features_img.pt', 'rb') as f:
        Ximg_in = torch.load(f)
    with open('data/original_data/labels.pt', 'rb') as f:
        Y_in = torch.load(f)

    data_s, data_images, label, _ = process_inputs(X_in, Ximg_in, Y_in)

    final_data = {}
    for idx, dat in enumerate(data_s):
        symbolic = dat.reshape([81,9])
        a = torch.zeros((1,81))
        symbolic = torch.cat((torch.t(a),symbolic),1)
        for i in range(81):
            if 1 not in symbolic[i]:
                symbolic[i][0]= 1.
        symbolic = symbolic.argmax(dim=1)
        symbolic = symbolic.unsqueeze(dim=0)
        final_data[idx] = symbolic.numpy().astype(int).reshape(9,9)

    
    final_labels = {}
    for idx, ll in enumerate(label):
        symbolic_label = ll.reshape([81,9])
        symbolic_label = symbolic_label.argmax(dim=1)+1
        symbolic_label = symbolic_label.unsqueeze(dim=0)
        final_labels[idx] = symbolic_label.numpy().astype(int).reshape(9,9)
    
    final_images = {}
    for idx, img in enumerate(data_images):
        final_images[idx] = img.numpy()

    return final_data, final_images, final_labels

def mask_data_generation_satnet(min_noise,max_noise,in_boards,in_solutions):
    print('Creating data mask predictor')
    mask_input = {}
    mask_labels = {}
    key = 0
    factor_num = 10
    for key in tqdm(range(len(in_boards))):
        mask_input_list, mask_labels_list = add_noise(in_boards[key],in_solutions[key],min_noise,max_noise,factor=factor_num,noise_input=False)
        for j in range(len(mask_input_list)):
            idx = key*factor_num+j
            mask_input[idx] = mask_input_list[j]
            mask_labels[idx] = mask_labels_list[j]
    return mask_input, mask_labels
    

def train_val_test_split_satnet(data_name, final_data, final_images, final_solutions, noise_data, mask_data):

    print('Splitting train-test-validation')
    file1 = f'data/{data_name}/{data_name}'
    file2 = f'data/{data_name}/{data_name}_noise'   
    file3 = f'data/{data_name}/{data_name}_sol'
    file4 = f'data/{data_name}/{data_name}_mask'

    data_lenght = len(final_data)
    num_train_data = int(len(final_data)*0.9)
    num_val_data = int((len(final_data)-num_train_data)/2)
    num_test_data = data_lenght - num_train_data - num_val_data

    file_list = [file1,file2,file3,file4]
    
    for file in tqdm(file_list):
        if file == file1:
            data = final_data
        elif file == file2:
            data = noise_data
        elif file == file3:
            data = final_solutions
        else:
            assert file == file4
            data = mask_data

        train_data = {k:data[k] for k in range(num_train_data) if k in data}
        validation_data = {(k - num_train_data):data[k] for k in range(num_train_data,num_train_data + num_val_data) if k in data} 
        test_data = {(k - num_train_data - num_val_data):data[k] for k in range(num_train_data + num_val_data,data_lenght) if k in data} 
        train_file = file + '-train.npy'
        validation_file = file + '-valid.npy'
        test_file = file + '-test.npy'
        np.save(train_file, train_data)
        np.save(validation_file, validation_data)
        np.save(test_file, test_data)

    # final_images
    hdf5_file_name = f'data/{data_name}/{data_name}_imgs-train.hdf5'
    hdf5_file = h5py.File(hdf5_file_name,'w') 
    print('- Saving hdf5 train..')
    for i in tqdm(range(num_train_data)):
        visual_board = final_images[i]
        hdf5_file.create_dataset(str(i),(81,1,28,28),dtype='float64',data=visual_board)
    
    hdf5_file_name = f'data/{data_name}/{data_name}_imgs-valid.hdf5'
    hdf5_file = h5py.File(hdf5_file_name,'w') 
    print('- Saving hdf5 validation..')
    for i in tqdm(range(num_val_data)):
        visual_board = final_images[num_train_data + i]
        hdf5_file.create_dataset(str(i),(81,1,28,28),dtype='float64',data=visual_board)

    hdf5_file_name = f'data/{data_name}/{data_name}_imgs-test.hdf5'
    hdf5_file = h5py.File(hdf5_file_name,'w')
    print('- Saving hdf5 test..')
    for i in tqdm(range(num_test_data)):
        visual_board = final_images[num_train_data + num_val_data + i]
        hdf5_file.create_dataset(str(i),(81,1,28,28),dtype='float64',data=visual_board)


def images_generation(data_name,flag):
    print('Generating board images - ', flag)
    if flag == 'test':
        labels =  mnist.test_labels()
        images = mnist.test_images()
    elif flag == 'train':
        labels = mnist.train_labels()[:-10000]
        images = mnist.train_images()[:-10000]
    else:
        assert(flag == 'valid')
        labels = mnist.train_labels()[-10000:]
        images = mnist.train_images()[-10000:]
    data_in = f'data/{data_name}/{data_name}-{flag}.npy'
    data_out_imgs_path = f'data/{data_name}/images/'
    try:
        shutil.rmtree(data_out_imgs_path)
    except OSError:
        pass
    os.mkdir(data_out_imgs_path)
    boards_dict = np.load(data_in,allow_pickle=True).item()
    num_cores =  mp.cpu_count() # change to mp.cpu_count() or custom number for faster generation
    Parallel(n_jobs = num_cores)(delayed(images_generation_core)(data_out_imgs_path,boards_dict[key],key,labels,images) for key in tqdm(boards_dict))


def images_generation_core(data_out_imgs_path,bb,key,labels,images):
    img_file_name = data_out_imgs_path+str(key)
    board = Board(bb)
    board.generate_mnist_board(labels,images)
    np.save(img_file_name, board.visual_board) 


def dataset_generation(data_name,solver):
    print('Generating solutions')
    stats = []
    data_in = f'data/{data_name}/{data_name}.npy'
    data_out_sol = f'data/{data_name}/{data_name}_sol'
    solutions = {}
    counter = 0
    prolog_instance = None
    if solver == 'prolog':
        prolog_instance = Prolog()
        prolog_instance.consult("src/sudoku_solver/sudoku_prolog.pl")
    boards_dict = np.load(data_in,allow_pickle=True).item()
    for key in tqdm(boards_dict):
            board = Board(boards_dict[key])
            time1 = time.time()
            board.solve(solver, prolog_instance)
            time2 = time.time()
            solutions[counter] = board.board
            stats.append(time2 - time1)
            counter += 1
    np.save(data_out_sol, solutions)
    print('sudoku solved: ',len(stats))
    print(f'tot time: {sum(stats):4f}')
    print(f'mean time:  {statistics.mean(stats):4f}')
    print(f'max time:  {max(stats):4f}')
    print(f'min time:  {min(stats):4f}')    


def mask_data_generation(data_name,min_noise,max_noise,factor_num,noise_input = False):
    print('Creating data mask predictor')
    # load
    mask_input = {}
    mask_labels = {}
    in_boards = {}
    in_solutions = {}
    data_in = f'data/{data_name}/{data_name}.npy'
    data_sol = f'data/{data_name}/{data_name}_sol.npy'
    data_out_noise = f'data/{data_name}/{data_name}_noise'
    data_out_mask = f'data/{data_name}/{data_name}_mask'
    in_boards = np.load(data_in,allow_pickle=True).item()
    in_solutions = np.load(data_sol,allow_pickle=True).item()

    for key in tqdm(in_boards.keys()):
    #for key in tqdm(range(len(in_boards))):
        mask_input_list, mask_labels_list = add_noise(in_boards[key],in_solutions[key],min_noise,max_noise,factor=factor_num,noise_input=noise_input,min_noise_i=0,max_noise_i=10)
        for j in range(len(mask_input_list)):
            idx = key*factor_num+j
            mask_input[idx] = mask_input_list[j]
            mask_labels[idx] = mask_labels_list[j]
    np.save(data_out_noise, mask_input)
    np.save(data_out_mask, mask_labels)


def add_noise(board,solution_board,min_noise,max_noise,factor,noise_input=True,min_noise_i=None,max_noise_i=None):     

    mask_input_list = []
    mask_labels_list = []

    for _ in range(factor):

        noise_amount = np.random.randint(min_noise,max_noise)/100
        noise_board = np.copy(solution_board)
        labels =  np.ones((9, 9), dtype=int)
        zeros = np.nonzero(board==0)
        solutions = np.empty(len(zeros[0]), dtype=int)
        for index in range(len(zeros[0])):  
            i = zeros[0][index]
            j = zeros[1][index]
            solutions[index] = solution_board[i][j]
        shuffled_indices = np.random.choice(len(solutions), size=int(len(solutions)*noise_amount), replace=False)
        solutions_shuffled = solutions[shuffled_indices]
        np.random.shuffle(solutions_shuffled)
        solutions[shuffled_indices] = solutions_shuffled
        for index in range(len(zeros[0])):  
            i = zeros[0][index]
            j = zeros[1][index]
            if noise_board[i][j] != solutions[index]:
                noise_board[i][j] = solutions[index]
                labels[i][j] = 0

        if noise_input:
            # add noise input
            noise_amount_input = np.random.randint(min_noise_i,max_noise_i)/100
            input_cells = np.nonzero(board)
            for index in range(len(input_cells[0])):
                if np.random.choice(2,1,p=[noise_amount_input,1-noise_amount_input]) == 0:
                    # add noise
                    i = input_cells[0][index]
                    j = input_cells[1][index]
                    values_vector = [kk for kk in range(1,10)]
                    values_vector.remove(board[i][j])
                    noise_board[i][j] = np.random.choice(values_vector,1)
                    labels[i][j] = 0

        mask_input_list.append(noise_board)
        mask_labels_list.append(labels)
    
    if factor == 1:
        return mask_input_list[0], mask_labels_list[0]

    return mask_input_list, mask_labels_list


def format_conversion(data_name,data_new_name):
    '''
    limit: 100000 data points
    '''
    print('Converting input format')
    data_in = f'data/original_data/{data_name}' 
    data_out = f'data/{data_new_name}/{data_new_name}'
    file_in = open(data_in, 'r')
    lines = file_in.readlines()
    data = {}
    data_list = []
    for line in tqdm(lines):
        if '#' not in line and len(line)>80:
            input_line = line.replace('.','0').replace('\n','')
            input_line = np.array([int(i) for i in input_line])
            input_line = input_line.reshape(9,9)
            data_list.append(input_line)
    file_in.close()
    # shuffle the dataset
    indices = [i for i in range(len(data_list))]
    random.shuffle(indices)
    for i in range(len(data_list)):
        data[indices[i]] = data_list[i]
        if i > 100000:
            break
    data = dict(sorted(data.items()))
    np.save(data_out, data)



def train_val_test_split(data_name):
    train_param = 0.7
    valid_param = 0.2
    print('Splitting train-test-validation')
    file1 = f'data/{data_name}/{data_name}'
    file2 = f'data/{data_name}/images/'
    file3 = f'data/{data_name}/{data_name}_noise'   
    file4 = f'data/{data_name}/{data_name}_sol'
    file5 = f'data/{data_name}/{data_name}_mask'
    data_lenght = len(np.load(file1+'.npy',allow_pickle=True).item())
    num_train_data = int(data_lenght * train_param)
    num_val_data = int(data_lenght * valid_param)
    num_test_data = data_lenght - num_train_data - num_val_data
    file_list = [file1,file3,file4,file5]
    
    for file in tqdm(file_list):
        data =  np.load(file+'.npy',allow_pickle=True).item()
        train_data = {k:data[k] for k in range(num_train_data) if k in data}
        validation_data = {k-num_train_data:data[k] for k in range(num_train_data,num_train_data+num_val_data) if k in data} 
        test_data = {k-(num_train_data+num_val_data):data[k] for k in range(num_train_data+num_val_data,data_lenght) if k in data} 
        train_file = file + '-train.npy'
        validation_file = file + '-valid.npy'
        test_file = file + '-test.npy'
        np.save(train_file, train_data)
        np.save(validation_file, validation_data)
        np.save(test_file, test_data)

    images_generation(data_name,'train')
    hdf5_file_name = f'data/{data_name}/{data_name}_imgs-train.hdf5'
    hdf5_file =h5py.File(hdf5_file_name,'w') 
    print('- Saving hdf5 train..')
    for i in tqdm(range(num_train_data)):
        visual_board = np.load(file2+str(i)+'.npy',allow_pickle=True)
        hdf5_file.create_dataset(str(i),(252,252),dtype='float64',data=visual_board)
        os.remove(file2+str(i)+'.npy')
    shutil.rmtree(f'data/{data_name}/images/')
    
    images_generation(data_name,'valid')
    hdf5_file_name = f'data/{data_name}/{data_name}_imgs-valid.hdf5'
    hdf5_file =h5py.File(hdf5_file_name,'w') 
    print('- Saving hdf5 validation..')
    for i in tqdm(range(num_val_data)):
        visual_board = np.load(file2+str(i)+'.npy',allow_pickle=True)
        hdf5_file.create_dataset(str(i),(252,252),dtype='float64',data=visual_board)
        os.remove(file2+str(i)+'.npy')
    shutil.rmtree(f'data/{data_name}/images/')

    images_generation(data_name,'test')
    hdf5_file_name = f'data/{data_name}/{data_name}_imgs-test.hdf5'
    hdf5_file =h5py.File(hdf5_file_name,'w')
    print('- Saving hdf5 test..')
    for i in tqdm(range(num_test_data)):
        visual_board = np.load(file2+str(i)+'.npy',allow_pickle=True)
        hdf5_file.create_dataset(str(i),(252,252),dtype='float64',data=visual_board)
        os.remove(file2+str(i)+'.npy')
    shutil.rmtree(f'data/{data_name}/images/')


def process_big_kaggle(args):
    print('----------------------------------------------')
    print('Processing dataset big_kaggle (puzzles0_kaggle)')
    print('----------------------------------------------')
    # best with noise in [0,10]
    data_name = 'puzzles0_kaggle'
    data_new_name = 'big_kaggle'
    min_noise=0
    max_noise=10
    if args.min_noise and args.max_noise and args.min_noise < args.max_noise:
        min_noise=args.min_noise
        max_noise=args.max_noise
    if not isdir('data/' + data_new_name):
        os.mkdir('data/' + data_new_name)
    format_conversion(data_name,data_new_name)
    assert args.solver in ['default','prolog','backtrack'] , 'choose a solver in [default, prolog, backtrack]'
    solver = args.solver
    if args.solver== 'default':
        solver = 'backtrack'        
    dataset_generation(data_new_name,solver)
    mask_data_generation(data_new_name,min_noise,max_noise,factor_num=1,noise_input=True)
    train_val_test_split(data_new_name)
    

def process_minimal_17(args):
    print('----------------------------------------------')
    print('Processing dataset minimal_17 (puzzles2_17_clue)')
    print('----------------------------------------------')
    # best with noise in [20,40]
    data_name = 'puzzles2_17_clue'
    data_new_name = "minimal_17"
    min_noise=20
    max_noise=40
    if args.min_noise and args.max_noise and args.min_noise < args.max_noise:
        min_noise=args.min_noise
        max_noise=args.max_noise

    if not isdir('data/' + data_new_name):
        os.mkdir('data/' + data_new_name)
    format_conversion(data_name,data_new_name)
    assert args.solver in ['default','prolog','backtrack'] , 'choose a solver in [default, prolog, backtrack]'
    solver = args.solver
    if args.solver== 'default':
        solver = 'prolog'

    dataset_generation(data_new_name,solver)
    mask_data_generation(data_new_name,min_noise,max_noise,factor_num=1,noise_input=True)
    train_val_test_split(data_new_name)


def process_multiple_sol(args):
    print('----------------------------------------------')
    print('Processing dataset multiple_sol (puzzles7_serg_benchmark)')
    print('----------------------------------------------')
    # best with noise in [0,10]
    data_name = 'puzzles7_serg_benchmark'
    data_new_name = 'multiple_sol'
    min_noise=0
    max_noise=10
    if args.min_noise and args.max_noise and args.min_noise < args.max_noise:
        min_noise=args.min_noise
        max_noise=args.max_noise
    if not isdir('data/' + data_new_name):
        os.mkdir('data/' + data_new_name)
    format_conversion(data_name,data_new_name)
    assert args.solver in ['default','prolog','backtrack'] , 'choose a solver in [default, prolog, backtrack]'
    solver = args.solver
    if args.solver== 'default':
        solver = 'backtrack'

    dataset_generation(data_new_name,solver)
    mask_data_generation(data_new_name,min_noise,max_noise,factor_num=10,noise_input=True)
    train_val_test_split(data_new_name)



def test_load(filename,idx=0):
    
    data_type = '-train'
    solutions = np.load(f'data/{filename}/{filename}_sol{data_type}.npy',allow_pickle=True).item()
    solutions = solutions[idx]
    initial = np.load(f'data/{filename}/{filename}{data_type}.npy',allow_pickle=True).item()
    initial = initial[idx]
    image_file = h5py.File(f'data/{filename}/{filename}_imgs{data_type}.hdf5','r')[str(idx)]
    image = Board()
    image.visual_board = np.array(image_file)
    image.visualize('test_load.png')
    print('---> Initial board:')
    Board(initial).visualize_shell()
    print('\n\n---> Solution board:')
    Board(solutions).visualize_shell()
    print('Image generated at: data/out/test_load.png')


def statistics_datasets(dataset_name):
    print(f'---- Statistics for {dataset_name} dataset ----')
    data_in = 'data/original_data/'+ dataset_name 
    file_in = open(data_in, 'r')
    lines = file_in.readlines()
    data_list = []
    non_zero = 0
    min_nz = 81
    max_nz = 0
    count = 0
    for line in tqdm(lines):
        if '#' not in line and len(line)>80:
            count += 1
            input_line = line.replace('.','0').replace('\n','')
            input_line = np.array([int(i) for i in input_line])
            data_list.append(input_line)
            non_zero_tmp = np.count_nonzero(input_line)
            non_zero += non_zero_tmp
            if non_zero_tmp < min_nz:
                min_nz=non_zero_tmp
            if non_zero_tmp > max_nz:
                max_nz=non_zero_tmp
    file_in.close()
    non_zero /= len(data_list)
    print(f'Non zero avg for {dataset_name}: {non_zero}')
    print(f'Min: {min_nz}')
    print(f'Max: {max_nz}')
    print(f'Size: {count}')


def process_inputs(X, Ximg, Y):
        is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()
        Ximg = Ximg.flatten(start_dim=1, end_dim=2)
        Ximg = Ximg.unsqueeze(2).float()
        X = X.view(X.size(0), -1)
        Y = Y.view(Y.size(0), -1)
        is_input = is_input.view(is_input.size(0), -1)
        return X, Ximg, Y, is_input


def statistics_satnet():
    print('---- Statistics for satnet dataset ----')
    with open('data/original_data/features.pt', 'rb') as f:
        X_in = torch.load(f)
    with open('data/original_data/features_img.pt', 'rb') as f:
        Ximg_in = torch.load(f)
    with open('data/original_data/labels.pt', 'rb') as f:
        Y_in = torch.load(f)
    data_s, _, _, _ = process_inputs(X_in, Ximg_in, Y_in)
    num_hints = []
    min_nz = 81
    max_nz = 0
    non_zero = 0
    for i in data_s:
        input_line = i.reshape([81,9])
        a = torch.zeros((1,81))
        input_line = torch.cat((torch.t(a),input_line),1)
        for i in range(81):
            if 1 not in input_line[i]:
                input_line[i][0]= 1.
        input_line = input_line.argmax(dim=1)
        non_zero_tmp = np.count_nonzero(input_line)
        num_hints.append(non_zero)
        if non_zero_tmp < min_nz:
                min_nz=non_zero_tmp
        if non_zero_tmp > max_nz:
            max_nz=non_zero_tmp
        non_zero += non_zero_tmp
    non_zero /= len(data_s)
    print(f'Non zero avg for satnet: {non_zero}')
    print(f'Min: {min_nz}')
    print(f'Max: {max_nz}')
    print(f'Size: {len(data_s)}')


def main_data_gen():
    parser = init_parser()
    args = parser.parse_args()

    if args.dataset == 'multiple_sol':
        process_multiple_sol(args) # multiple_sol
    elif args.dataset == 'minimal_17':
        process_minimal_17(args) # minimal_17
    elif args.dataset == 'big_kaggle':
        process_big_kaggle(args) # big_kaggle
    elif args.dataset == 'satnet_data':
        process_satnet_data(args) # satnet_data
    else:
        print(' Generating all datasets...')
        process_multiple_sol(args) # multiple_sol
        process_minimal_17(args) # minimal_17
        process_big_kaggle(args) # big_kaggle
        process_satnet_data(args) # satnet_data

        
    



if __name__=='__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    main_data_gen()

    # uncomment the following line for a dataset loading test 
    # test_load('puzzles0_kaggle',42)

    # uncomment the following block for datasets statistics
    # statistics_datasets('puzzles0_kaggle')
    # statistics_datasets('puzzles7_serg_benchmark')
    # statistics_datasets('puzzles2_17_clue')
    # statistics_satnet()
