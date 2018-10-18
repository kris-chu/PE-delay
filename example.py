import functionsal_1 as f
import numpy as np

fault_list = [7, 6, 3]

data_list = []
for i in range(16):
    data_list.append(f.float2Qcode(np.random.rand(), 7))
    data_list.append(f.float2Qcode(-np.random.rand(), 7))
print(data_list)

for i in range(1, 32):
    print(data_list[i], data_list[i-1])
    print(f.insert_fault(data_list[i], data_list[i-1], fault_list))