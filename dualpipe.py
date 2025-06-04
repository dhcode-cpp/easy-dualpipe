# python dualpipe.py

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from zero_bubble import ZeroBubbleModel

class DualPipe:
    def __init__(self, 
                 models: Tuple[nn.Module, nn.Module],
                 dim, 
                 rank = 0, 
                 world_size = 1):
        super(DualPipe, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.dim = dim
        self.is_first_rank = self.rank == 0
        self.is_last_rank = self.rank == self.world_size - 1
        self.is_in_first_half = self.rank < self.world_size // 2
        self.is_in_second_half = self.rank >= self.world_size // 2
        self.is_middle_rank = (self.rank == self.world_size // 2 - 1) or (self.rank == self.world_size // 2) # 3,4
        self.models = models

        self.stage_output_lists : Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([0,1] , [])
        self.stage_output_grad_lists : Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([0,1], [])
        self.stage_output_tmp_lists : Tuple[List[List[torch.Tensor]], List[List]] = ([0,1], [])

        self.forward_idxs = [0, 0]
        self.backward_idxs = [0, 0]

        self.comm_ops: List[dist.P2POp] = []
    
    def forward(self, phase, x,):
        f_idx =  self.forward_idxs[phase] 
        stage_output_tmp, stage_output = self.models[phase](x)
        self.stage_output_tmp_lists[phase][f_idx] = stage_output_tmp
        self.stage_output_lists[phase][f_idx] = stage_output

    
    def backward(self, phase, grad):
        b_idx = self.backward_idxs[phase] 
        dx = self.models[phase].backward_zero_bubble(self.stage_output_tmp_lists[phase][b_idx], grad, b_idx, is_send = False)
        self.stage_output_grad_lists[phase][b_idx] = dx
    
    def recv_output(self, phase):
        is_first_stage = (self.is_first_rank and phase == 0) or (self.is_last_rank and phase == 1)
        if is_first_stage:
            return
        
        f_idx =  self.forward_idxs[phase] 
        tmp_tensor = torch.zeros_like(self.known_tensor)

        src_phase = (-1) ** phase

        self.comm_ops.append(dist.P2POp(dist.irecv, tmp_tensor, self.rank - src_phase))
        self.stage_output_lists[phase][f_idx] = tmp_tensor

    def send_output(self, phase):
        is_last_stage = (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0)
        if is_last_stage:
            return
        
        f_idx =  self.forward_idxs[phase] 
        tmp_tensor = torch.zeros_like(self.known_tensor)

        dst_phase = (-1) ** phase 
        self.comm_ops.append(dist.P2POp(dist.isend, tmp_tensor, self.rank + dst_phase))

        self.stage_output_lists[phase][f_idx] = tmp_tensor

    def recv_gradient(self, phase):
        is_last_stage = (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0)
        if is_last_stage:
            return  

        b_idx = self.backward_idxs[phase] 
        tmp_tensor = torch.zeros_like(self.known_tensor)
        src_phase = (-1) ** phase 
        self.comm_ops.append(dist.P2POp(dist.irecv, tmp_tensor, self.rank + src_phase))
        self.stage_output_grad_lists[phase][b_idx] = tmp_tensor

    def send_gradient(self, phase):
        is_first_stage = (self.is_first_rank and phase == 0) or (self.is_last_rank and phase == 1)
        if is_first_stage:
            return
        
        b_idx = self.backward_idxs[phase] 
        tmp_tensor = torch.zeros_like(self.known_tensor)
        dst_phase =  (-1) ** phase 
        # dist.send(tmp_tensor, dst = self.rank - dst_phase, tag = 10086)
        self.comm_ops.append(dist.P2POp(dist.isend, tmp_tensor, self.rank - dst_phase))
        self.stage_output_grad_lists[phase][b_idx] = tmp_tensor
    
    def forward_step(self, phase = 0, x = None, is_recv = True, is_send = True):
        if is_recv:
            self.recv_output(phase)
        self.comm_wait()

        if x != None:
            self.forward( phase, x)
        else:
            self.forward( phase, self.stage_output_lists[phase][self.forward_idxs[phase]] )
        if is_send:
            self.send_output(phase)
        self.forward_idxs[phase] += 1
        return 
    
    def backward_step(self, phase = 0, y = None, is_recv = True, is_send = True):
        if is_recv:
            self.recv_gradient(phase)
        self.comm_wait()

        b_idx = self.backward_idxs[phase]
        if y != None:
            do = self.models[phase].layers[-1].loss_backward(self.stage_output_lists[phase][b_idx], y)
            self.backward( phase, do)
        else:
            self.backward( phase, self.stage_output_grad_lists[phase][b_idx] )

        if is_send:
            self.send_gradient(phase)
        self.backward_idxs[phase] += 1
        return 
    
    def comm_wait(self, ):
        if self.comm_ops :
            reqs = dist.batch_isend_irecv(self.comm_ops)
            for req in reqs:
                req.wait()
            self.comm_ops = []


    def step(self, x, y, known_shape):
        '''
        easy-dualpipe design a simplest bi-directional pipeline parallelism's schedule.
        '''
        # known tensor 是为了便于接收rank之间传递的tensor, 而前提是我们需要知道tensor的尺寸
        self.known_tensor = torch.zeros(known_shape)
        for i in range(self.world_size):
            for j in range(2): # phase
                self.stage_output_grad_lists[j].append(self.known_tensor.clone())
                self.stage_output_lists[j].append(self.known_tensor.clone())
                self.stage_output_tmp_lists[j].append([])

        rank = self.rank
        world_size = self.world_size
        is_in_first_half = self.is_in_first_half
        is_in_second_half = self.is_in_second_half

        cur_phase = is_in_second_half
        next_phase = is_in_first_half

        # step1: f0
        step = abs(world_size // 2 - rank) + is_in_second_half
        for i in range(step):
            self.forward_step(phase = cur_phase, x = x[i]) 

        # # step2: f1f0
        # print('-----f1f0-----')
        step = world_size // 2 - (abs(world_size // 2 - rank) + is_in_second_half)
        for i in range(step):
            self.forward_step(phase = next_phase, x = x[i])
            self.forward_step(phase = cur_phase, x = x[i])
        
        # step2: f1b1
        # print('-----f1b1-----')
        step = abs(world_size // 2 - rank) + is_in_second_half
        for i in range(step):
            self.forward_step(phase = next_phase, x = x[i])  
            self.backward_step(phase = next_phase, y = y[i])  
        
        # step3: b0b1
        step = world_size // 2 - (abs(world_size // 2 - rank) + is_in_second_half)
        # b0b1 实际计算
        for i in range(step):
            self.backward_step(phase = cur_phase, y = y[i]) 
            self.backward_step(phase = next_phase, y = y[i]) 
        
        # step4: b0
        step = abs(world_size // 2 - rank) + is_in_second_half
        for i in range(step):
            self.backward_step(phase = cur_phase, y = y[i])  
        
        self.comm_wait()

        #TODO: reduce-gradient and update

        return 
             

def run(rank, master_addr, master_port, world_size, backend='gloo'):

    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)

    # 准备数据
    dim = 512
    num_blocks = 16 
    bs = 32
    micro_batch_size = world_size 

    if rank == 0 or rank == world_size -1:
        x = torch.randn(bs, dim)
        y = torch.randn(bs, dim)
        x_list = list(torch.chunk(x, micro_batch_size, dim = 0)) # phase 0, phase n
        y_list = list(torch.chunk(y, micro_batch_size, dim = 0)) # phase n, phase 1
    else:
        x_list = [None] * micro_batch_size
        y_list = [None] * micro_batch_size
    print(f'[rank{rank}] x_list:{x_list[0]}, y_list:{y_list[0]}')
    tmp_shape = [bs // world_size, dim]
    
    # reverse parameters
    # model_0:  [layer0, layer1], [layer2, layer3], ..., [layer6, layer7]
    # model_1:  [layer6, layer7], [layer_4, layer5], ..., [layer0, layer1]
    pipe_model_0 = ZeroBubbleModel(dim, num_blocks=num_blocks, rank = rank, world_size=world_size)
    pipe_model_1 = ZeroBubbleModel(dim, num_blocks=num_blocks, rank = rank, world_size=world_size)
    dualpipe = DualPipe([pipe_model_0, pipe_model_1], dim = dim, rank = rank, world_size=world_size)

    # pair: (x_list_a, y_list_a), (x_list_b, y_list_b)
    # rank 0: x_list_a = [xa1, xa2, xa3, xa4], y_list_b = [yb1, yb2, yb3, yb4]
    # rank 1: x_list = [-, -, -, -], y_list = [-, -, -, -]
    # rank i: x_list = [-, -, -, -], y_list = [-, -, -, -]
    # rank N: x_list_b = [xb1, xb2, xb3, xb4], y_list_a = [ya1, ya2, ya3, ya4]
    dualpipe.step( x = x_list, y = y_list, known_shape = tmp_shape)
            
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run, args=("127.0.0.1", "12801", 8, ), nprocs=8)
