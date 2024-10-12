import numpy as np
from collections import Counter

d1 = [np.array([1, 3, 1, 2]), np.array([2, 4, 3, 4])]
print(d1[0].shape)
d2 = np.array(d1)
print(d2.shape)

l1 = [1, 2]
l2 = np.array(l1)
print(l2.shape)
# # 预处理并构建哈希表，允许每个 numpy 数组对应多个标签
# def build_data_hash_map(data_set):
#     # 创建字典，其中键是字符串形式的 data，值是对应的标签列表
#     data_hash_map = {}
#     for i, data in enumerate(data_set['data']):
#         key = tuple(data)  # 将 numpy 数组转换为 tuple 作为键
#         if key not in data_hash_map:
#             data_hash_map[key] = []  # 初始化标签列表
#         data_hash_map[key].append(data_set['label'][i])  # 添加标签
#     return data_hash_map

# # 查找 obs_ 对应的标签众数
# def find_label_mode_with_hash_map(obs_, data_hash_map):
#     # 将 obs_ 转换为 tuple 形式并进行查找
#     labels = data_hash_map.get(tuple(obs_), None)  # 如果找不到，返回 None
#     if labels is None:
#         return None
    
#     # 使用 Counter 统计每个标签的出现次数，返回出现频率最高的标签
#     label_counts = Counter(labels)
#     most_common_label = label_counts.most_common(1)[0][0]  # 返回出现频率最高的标签
#     return most_common_label

# # 示例使用
# if __name__ == "__main__":
#     # 模拟一个 data_set，里面有一些数据和对应的标签
#     data_set = {
#         'data': [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8]), np.array([3, 4, 5]), np.array([3, 4, 5])],
#         'label': [0, 1, 2, 3, 3]  # 注意，两个相同的数组有不同的标签
#     }

#     # 构建哈希表
#     data_hash_map = build_data_hash_map(data_set)

#     # 预处理过的 obs_
#     obs_ = np.array([3, 4, 5])

#     # 使用哈希表查找并返回出现频率最高的标签
#     most_common_label = find_label_mode_with_hash_map(obs_, data_hash_map)

#     # 输出众数标签
#     print(f"出现频率最高的标签: {most_common_label}")
