# python step_dualpipe.py

import torch.distributed as dist
import torch.multiprocessing as mp

def step_schedule(x, phase, rank, world_size):

    it_idx = []
    is_in_first_half = rank < world_size // 2
    is_in_second_half = rank >= world_size // 2
    f_idx_0 = 0
    f_idx_1 = 0
    b_idx_0 = 0
    b_idx_1 = 0

    # step1: f0
    '''
    rank 0~7 -> 4,3,2,1, 1,2,3,4
    '''
    step = abs(world_size // 2 - rank) + is_in_second_half
    it_idx.extend([-1] * (world_size // 2 - step))
    for i, _ in enumerate(range(step)):
        if is_in_first_half:
            it_idx.extend([x[0][f_idx_0]])
            f_idx_0 += 1
        else:
            it_idx.extend([x[1][f_idx_1]])
            f_idx_1 += 1

    # step2: f0f1
    '''
    rank 0~7 -> 0,1,2,3, 3,2,1,0
    '''
    step = world_size // 2 - (abs(world_size // 2 - rank) + is_in_second_half)
    for i, _ in enumerate(range(step)):
        if is_in_first_half:
            it_idx.extend([x[1][f_idx_1], x[0][f_idx_0]]) 
        else:
            it_idx.extend([x[0][f_idx_0], x[1][f_idx_1]]) 
        f_idx_0 += 1
        f_idx_1 += 1
    
    # step2: f1b1
    '''
    rank 0~7 -> 4,3,2,1, 1,2,3,4
    '''
    step = abs(world_size // 2 - rank) + is_in_second_half
    for i, _ in enumerate(range(step)):
        if is_in_first_half:
            it_idx.extend([x[1][f_idx_1], x[1][b_idx_1]])
            f_idx_1 += 1
            b_idx_1 += 1
        else:
            it_idx.extend([x[0][f_idx_0], x[0][b_idx_0]]) 
            f_idx_0 += 1
            b_idx_0 += 1

    # step3: b0b1
    '''
    rank 0~7 -> 0,1,2,3, 3,2,1,0
    '''
    step = world_size // 2 - (abs(world_size // 2 - rank) + is_in_second_half)
    for i, _ in enumerate(range(step)):
        if is_in_first_half:
            it_idx.extend([x[0][b_idx_0], x[1][b_idx_1]]) 
        else:
            it_idx.extend([x[1][b_idx_1], x[0][b_idx_0]]) 
        b_idx_1 += 1
        b_idx_0 += 1
    
    # step4: b1
    step = abs(world_size // 2 - rank) + is_in_second_half
    for i, _ in enumerate(range(step)):
        if is_in_first_half:
            it_idx.extend([x[0][b_idx_0]])
            b_idx_0 += 1
        else:
            it_idx.extend([x[1][b_idx_1]]) 
            b_idx_1 += 1
        it_idx.extend([-1]) 
    return it_idx
        

def run_step(rank, master_addr, master_port, world_size, backend='gloo'):
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)

    # x = torch.tensor([[0,1,2,3],
    #                   [10,11,12,13]])
    x = [[0,1,2,3],[10,11,12,13]]
    it_idx = step_schedule(x, 0, rank, world_size )
    print(f'rank[{rank}]: {it_idx}')

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run_step, args=("127.0.0.1", "12801", 8, ), nprocs=8)
