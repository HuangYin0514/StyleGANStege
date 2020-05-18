
import torch
import numpy as np


def compare_num_tenToTwo(a, b, num_len):
    numberof2_list_a = str(bin(a)).split('0b')
    numberof2_list_b = str(bin(b)).split('0b')
    numberof2_list_a = list(numberof2_list_a[1])[-num_len:]
    numberof2_list_b = list(numberof2_list_b[1])[-num_len:]
    while len(numberof2_list_a) < num_len:
        numberof2_list_a = ['0'] + numberof2_list_a
    while len(numberof2_list_b) < num_len:
        numberof2_list_b = ['0'] + numberof2_list_b
    result = [
        int(numberof2_list_a[i]) - int(numberof2_list_b[i])
        for i in range(len(numberof2_list_a))
    ]
    return np.sum(np.abs(result))


def compute_BER(input_noise, output_noise, sigma):
    number_row = input_noise.size(0)
    number_line = input_noise.size(1)
    input_noise_to_msg = torch.floor((input_noise + 1) * 2**(sigma - 1))
    output_noise_to_msg = torch.floor((output_noise + 1) * 2**(sigma - 1))
    if len(input_noise_to_msg.shape) > 2:
        input_noise_to_msg = input_noise_to_msg.squeeze(3).squeeze(2)
    if len(output_noise_to_msg.shape) > 2:
        output_noise_to_msg = output_noise_to_msg.squeeze(3).squeeze(2)
    input_noise_to_msg_numpy = np.array(input_noise_to_msg.detach().cpu(),
                                        dtype=np.integer)
    output_noise_to_msg_numpy = np.array(output_noise_to_msg.detach().cpu(),
                                         dtype=np.integer)
    error_counter = 0
    for row in range(number_row):
        result = [
            compare_num_tenToTwo(input_noise_to_msg_numpy[row][i],
                                 output_noise_to_msg_numpy[row][i], sigma)
            for i in range(number_line)
        ]
        error_counter += np.sum(result)
    ber = error_counter / (number_row * number_line * sigma)
    return ber
