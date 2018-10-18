import numpy as np
import torch
import torch.nn.functional as F


def conv2d(input, weight, bias, stride, padding, dilation, groups, mode, q_w):

    b, c_in, h_in, w_in = input.data.size()
    c_out, c_in, h, w = weight.data.size()

    h_out = np.int(np.floor((h_in+2*padding[0]-dilation[0]*(h-1)-1)/stride[0]) + 1)
    w_out = np.int(np.floor((w_in+2*padding[1]-dilation[1]*(w-1)-1)/stride[0]) + 1)

    input =  F.pad(input, (padding[1], padding[1], padding[0], padding[0]), 'constant', 0)

    weight = weight.view(c_out, c_in*h*w)
    # print(weight)

    convolvedimage = torch.autograd.Variable(torch.zeros(c_out, b*h_out*w_out))
    imagePatch = torch.autograd.Variable(torch.zeros(b, h_out*w_out, c_in*h*w))

    for im in range(b):
        for row in range(h_out):
            for col in range(w_out):
            # the im, row, col: help to choose a convolved piece from input
                imagePatch[im, row*w_out+col] = input[im, :, row:row+h, col:col+w].contiguous().view(-1)
                # imagePatch[im, row*w_out+col] = is_no_delay(input[im,:,row:row+h, col:col+w].contiguous().view(-1), q_in)
    imagePatch = imagePatch.contiguous().view(b*h_out*w_out, c_in*h*w)

    for row in range(b*h_out*w_out):
        image = imagePatch[row, :].contiguous().view(1, -1)
        for w_c_out in range(c_out):
            kernel = is_delay_happened(weight[w_c_out, :].contiguous().view(-1), mode, q_w)
            # test_weight(weight[w_c_out, :].view(-1), kernel)
            kernel = kernel.contiguous().view(1, -1)
            convolvedimage[w_c_out, row] = kernel.matmul(image.t()).item()

    convolvedimage = convolvedimage.contiguous().view(c_out, b*h_out*w_out) + bias.contiguous().view(-1, 1)
    convolvedimage = torch.transpose(convolvedimage.contiguous().view(c_out, b, h_out, w_out), 1, 0)
    return convolvedimage

def linear(input, weight, mode, q_w, bias=None):


    row_num = input.data.size()[0]
    col_num = weight.data.size()[0]

    linearedimage = torch.autograd.Variable(torch.zeros(row_num, col_num))

    for row in range(row_num):
        input_tensor = input[row, :].contiguous().view(1, -1)

        for col in range(col_num):
            weight_tensor = is_delay_happened(weight[col, :].contiguous().view(-1), mode, q_w)
            # test_weight(weight[col, :].view(-1), weight_tensor)
            weight_tensor = weight_tensor.contiguous().view(1, -1)
            linearedimage[row, col] = input_tensor.matmul(weight_tensor.t()).item()
    linearedimage = linearedimage.contiguous().view(row_num, col_num)

    if bias is not None:
        linearedimage = linearedimage + bias.contiguous().view(1, -1)
    return linearedimage

def is_delay_happened(input, mode, q_in):
    num_element = input.size()[0]
    input_1 = input.clone()
    for i in range(num_element):
        Q_, is_fault_happend, fault_list = rand_list(mode, q_in) 
        if is_fault_happend:
            if i == 0:
                pre_input_value = 0
            else:
                pre_input_value = float2Qcode(input[i-1], q_in)
            input_Q = float2Qcode(input[i], q_in)
            input_Q = insert_fault(input_Q, pre_input_value, fault_list, Q_)
            input_1[i] = Qcode2float(input_Q, q_in)
        else:
            pass
    return input_1

def float2Qcode(data, q):
    q2 = 2**q
    if q >= 8:
        qmax = 2**15
    else:
        qmax = 2**7
    results = int(data*q2)

    if results > qmax-1:
        results = qmax-1
    if results < -qmax:
        results = -qmax
    return results

def Qcode2float(data, q):
    q2 = 2**q
    results = float(data)/q2
    return results

def rand_list(prob, Q):
    """
    producing whether this column fails and its fault bits list
    """
    fault_list = []
    i = 1
    if Q >= 8:
        Q_ = 16
    else:
        Q_ = 8
    while np.random.rand() <= prob ** i:
        fault_list.append(int(np.random.rand() * Q_))
        i = i + 1
    if len(fault_list) == 0:
        flagx = False
    else:
        flagx = True
    return Q_, flagx, tuple(fault_list)

def insert_fault(data, data_1, fault_list, Q_):
    Q = Q_ - 1
    bitmask = 0
    for nb in fault_list:
        if nb == Q:
            m = -2**Q
        else:
            m = int(2**nb)
        bitmask = m | bitmask
    pre_bits = bitmask & data_1
    remain_bits = data & ~bitmask
    value = remain_bits | pre_bits
    return value

def test_weight(weight, kernel):
    kernel_len = kernel.size()[0]
    for i in range(kernel_len):
        if weight[i] != kernel[i]:
            print('weight: ', weight[i], '  &  kernel: ', kernel[i])
        else:
            pass

