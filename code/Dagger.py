import numpy as np
from abc import abstractmethod
from sklearn.neighbors import KNeighborsClassifier
import joblib

class DaggerAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass


# here is an example of creating your own Agent
class MyDaggerAgent(DaggerAgent):
	def __init__(self, necessary_parameters=None):
		super(DaggerAgent, self).__init__()
		# init your model
		self.model = KNeighborsClassifier(n_neighbors=1)


	# train your model with labeled data
	def update(self, data_batch, label_batch):
		data_batch = np.array(data_batch)  # 转换为二维数组 (n_samples, 16384)

		# 假设 train_data_set['label'] 是整数的列表
		label_batch = np.array(label_batch)  # 转换为一维数组 (n_samples,)
		self.model.fit(data_batch, label_batch)  # 使用 KNN 模型进行训练
		self.save()

	# select actions by your model
	# 注意, 只能输出0~7
	def select_action(self, data_batch):
		# 将一维数组转换为二维数组，确保数据格式正确
		if data_batch.ndim == 1:
			data_batch = data_batch.reshape(1, -1)
		label_predict = self.model.predict(data_batch)
		return int(label_predict[0])

	# save your model in specific path
	def save(self, path = './model.pkl'):
		joblib.dump(self.model, path)
		# my_model = joblib.load(path) 读取模型



