import numpy as np
import zipfile
import shutil
import random
import os
import cv2

# 初始化模型
with zipfile.ZipFile('code/data_set.zip', 'r') as zfile:
	zfile.extract(zfile.namelist()[0], path = './data')
	data_set = np.load('data/data_set.npy', allow_pickle = True).item()
	shutil.rmtree('data') # 递归地删除 ./data 目录和其中的所有文件

# print(len(data_set['data']))
# print(type(data_set['data'][6]))
# print(data_set['data'][6].shape)
print(len(data_set['label']))
print(data_set['label'])
# # print(data_set['label'][9099])

# with zipfile.ZipFile('code/data_set.zip', 'r') as zfile:
# 	# 解压缩文件
#     extracted_file = zfile.namelist()[0]  # 获取解压文件的原始名称
#     zfile.extract(extracted_file, path='./data')  # 解压到 './data' 目录
# 	# 构造解压后的文件路径
#     original_file_path = os.path.join('./data', extracted_file)
#     new_file_path = os.path.join('./data', 'expert_data_set.npy')  # 新文件名

    # # 重命名解压后的文件
    # os.rename(original_file_path, new_file_path)

	# # 加载重命名后的文件
    # expert_data_set = np.load(new_file_path, allow_pickle=True).item()

    # # 新的数据和标签 (假设是展平后的图像数据和对应的标签)
    # new_data = np.zeros(16384)  # 这是一个假设的状态数据，展平后的图像数组
    # new_label = 3  # 对应的新标签，假设这个数据对应动作 3

    # # 添加新的数据和标签
    # expert_data_set['data'].append(new_data)
    # expert_data_set['label'].append(new_label)

    # # 打印修改后的数据大小
    # print(f"修改后数据大小: {len(expert_data_set['data'])}")
    # print(f"修改后标签大小: {len(expert_data_set['label'])}")
    # print()

    # # 保存修改后的数据集到原文件或新文件中
    # np.save('data/expert_data_set.npy', expert_data_set)
    


# 加载现有的 .npy 文件
# expert_data_set = np.load('data/expert_data_set.npy', allow_pickle=True).item()
# print(len(expert_data_set['label']))

# 将 .npy 文件压缩为 .zip 文件
# 压缩 expert_data_set.npy 到 code/data_set.zip 中，不包含 data/ 目录
# with zipfile.ZipFile('code/data_set.zip', 'w') as zipf:
#     zipf.write('data/expert_data_set.npy', arcname='expert_data_set.npy', compress_type=zipfile.ZIP_DEFLATED)


# def pre_process(ob, size = (128, 128)):
# 	obs = ob.copy() # 防止直接修改原始图像ob
# 	# obs[obs == 236] = 0 # 去掉幽灵的影响
# 	obs_ = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
# 	obs_ = cv2.resize(obs_, size)
# 	obs_ = obs_.reshape(-1)
# 	return obs_

# ob = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)  # 模拟一个随机图像
# processed = pre_process(ob)
# print(processed)
# print(processed.shape)  # 应该输出：<class 'numpy.ndarray'>



# print可得: 
# data_set['data']是所有状态图像对应的np.array的list
# 每个array大小为16384 * 1 (= 128 * 128的灰度图展开), 每个元素代表每个点的灰度值, 范围似乎是0~176?
# data_set['label']是所有状态图像对应的下一步专家动作int的list, 动作范围为0~7
# 总共有9100条数据
# print(data_set['data'][0])
# print(data_set['label'])

# a = np.array([0, 1, 3, 4, 2, 3])
# print(a == 3) # [False False  True False False  True]
# print(a[a==3]) # [3 3]
# a[a==3] = -1
# print(a) # [ 0  1 -1  4  2 -1]

