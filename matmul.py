import torch
import numpy as np

def shiftadd_matmul(bWeight, alpha, activation):
    # bWeight: [K/8, nbits, N], uint8
    # alpha: [K/8, N*8/groupsize, nbits], float32
    # groupsize: 128
    K_div_8, nbits, N = bWeight.shape
    bWeight = bWeight.transpose(2, 1, 0).reshape(1, 1, N * nbits, K_div_8)   # [N * nbits, K/8]

    bs, seq_length, _ = activation.shape
    activation_split = activation.reshape(bs, seq_length, -1, 8)

    sign_combinations = np.array(np.meshgrid(*[[-1, 1]] * 8)).T.reshape(1, 1, -1, 8) #形成正负情况的遍历组合
    LUT = np.matmul(sign_combinations, activation_split.transpose(0,1,3,2))
    result = np.take_along_axis(LUT, bWeight, axis=2)
    result = result.reshape(bs, seq_length, N, nbits, K_div_8).transpose(0, 1, 2, 4, 3)

    alpha_expanded = np.repeat(alpha, 16, axis=1).transpose(1, 0 ,2)[np.newaxis, np.newaxis, :] # [N, K/8, nbits], default groupsize=128 
    result = result * alpha_expanded
    result = result.sum(axis=4).sum(axis=3).reshape(bs,seq_length,-1)
    return result

if __name__ == '__main__':
    weight_shape = (4096, 4096)
    act_shape = (1, 128, 4096)
    K, N = weight_shape
    nbits = 3
    groupsize = 128
    bWeight = np.random.randint(256, size=(K//8, nbits, N), dtype=np.uint8)
    alpha = np.random.randn(K//8, N*8 // groupsize, nbits)
    activation = np.random.randn(*act_shape)
    output = matmul(bWeight, alpha, activation)
    print(output.shape)
