import torch
import torch.nn as nn
import torch.distributed as dist

class ZeroBubbleMLP(nn.Module):
    def __init__(self, dim, rank = 0, world_size = 1):
        super(ZeroBubbleMLP, self).__init__()
        self.dim = dim
        self.rank = rank
        self.world_size = world_size
        self.w1 = nn.Linear(dim, dim * 4, bias = False)
        self.w2 = nn.Linear(dim * 4, dim, bias = False) 
    
    def forward(self, x):
        h = self.w1(x) 
        o = self.w2(h)
        return h, o 
    
    def loss_backward(self, o, label):
        do = 2 * (o - label) / o.numel()
        return do

    def backward_for_input(self, do):
        '''
        ZeroBubble Backward for dx
        '''
        dh = do @ self.w2.weight 
        dx = dh @ self.w1.weight
        return dh, dx

    def backward_for_weight(self, do, dh, h, x):
        '''
        ZeroBubble Backward for dw
        '''
        self.w2.weight.grad = do.t() @ h 
        self.w1.weight.grad = dh.t() @ x 
        return None


class ZeroBubbleModel(nn.Module):
    def __init__(self, dim, num_blocks, rank = 0, world_size = 1):
        super(ZeroBubbleModel, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.dim = dim
        self.num_blocks = num_blocks
        self.layers = torch.nn.ModuleList()
        self.local_num_blocks = num_blocks // world_size
        for i in range(self.local_num_blocks):
            self.layers.append(ZeroBubbleMLP(self.dim, self.rank, self.world_size))
        
    def forward(self, x): 
        '''
        return [[x1,h1], [x2, h2]] |  x3
        '''
        layers_output = [ [] for i in range(self.local_num_blocks)]
        for i, layer in enumerate(self.layers):
            layers_output[i].append(x)
            h, x = layer(x)
            layers_output[i].append(h)
        return layers_output, x # [[input, hidden],...], output
    
    def backward_zero_bubble(self, layers_output, do, b_idx, is_send = True, dst= None):
        # step1: ZeroBubble backward dx      
        # dx = None
        for layer, layer_output in zip(reversed( self.layers), reversed(layers_output)):
            dh, dx = layer.backward_for_input(do)
            layer_output.append(dh)
            layer_output.append(dx)

        # step2: isend dx
        if self.rank != 0 and is_send:
            if dst == None:
                req = dist.isend(dx, dst = self.rank-1, tag=10086)
                print(f'[rank{self.rank}] micro_batch:{self.world_size-b_idx-1}, dx isend-backward')
            else:
                req = dist.isend(dx, dst = dst, tag=10086)
                print(f'[rank{self.rank}] dst:{dst}, dx isend-backward')
            req.wait()
        
        # step3: ZeroBubble backward dw
        for layer, layer_output in zip(reversed( self.layers), reversed(layers_output)):
            layer.backward_for_weight(
                x = layer_output[0],
                h = layer_output[1],
                dh = layer_output[2],
                do = do,
            )
        return dx