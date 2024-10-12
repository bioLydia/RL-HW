from arguments import get_args
from Dagger import DaggerAgent, MyDaggerAgent
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from sklearn.neighbors import KNeighborsClassifier
import os
import cv2
import zipfile
import shutil

# 我们需要用到的动作

action_map = dict([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 11), (7, 12)])
action_map_inverse = dict([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (11, 6), (12, 7)])


def plot(record):
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(record['steps'], record['mean'],
	        color='blue', label='reward')
	ax.fill_between(record['steps'], record['min'], record['max'],
	                color='blue', alpha=0.2)
	ax.set_xlabel('number of steps')
	ax.set_ylabel('Average score per episode')
	ax1 = ax.twinx()
	ax1.plot(record['steps'], record['query'],
	         color='red', label='query')
	ax1.set_ylabel('queries')
	reward_patch = mpatches.Patch(lw=1, linestyle='-', color='blue', label='score')
	query_patch = mpatches.Patch(lw=1, linestyle='-', color='red', label='query')
	patch_set = [reward_patch, query_patch]
	ax.legend(handles=patch_set)
	fig.savefig('performance.png')


# the wrap is mainly for speed up the game
# the agent will act num_stacks frames instead of one frame
class Env(object):
	def __init__(self, env_name, num_stacks):
		self.env = gym.make(env_name)
		# num_stacks: the agent acts every num_stacks frames
		# it could be any positive integer
		self.num_stacks = num_stacks
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space

	def step(self, action):
		reward_sum = 0
		for stack in range(self.num_stacks):
			obs_next, reward, done, _, info = self.env.step(action)
			reward_sum += reward
			if done:
				self.env.reset()
				return obs_next, reward_sum, done, info
		return obs_next, reward_sum, done, info

	def reset(self):
		return self.env.reset()[0]
	
	
def pre_process(ob, size = (128, 128)):
	obs = ob.copy() # 防止直接修改原始图像ob
	obs_ = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
	obs_ = cv2.resize(obs_, size)
	obs_ = obs_.reshape(-1)
	return obs_


def main():
	# load hyper parameters
	args = get_args()
	num_updates = int(args.num_frames // args.num_steps)
	start = time.time()
	record = {'steps': [0],
	          'max': [0],
	          'mean': [0],
	          'min': [0],
	          'query': [0]}
	# query_cnt counts queries to the expert
	query_cnt = 0

	# environment initial
	envs = Env(args.env_name, args.num_stacks)

	# action_shape = envs.action_space.n
	# observation_shape = envs.observation_space.shape
	# print(action_shape, observation_shape)

	# agent initial
	agent = MyDaggerAgent()

	# You can play this game yourself for fun
	if args.play_game:
		# 建立经验专家
		with zipfile.ZipFile('code/data_set.zip', 'r') as zfile:
	    # 解压缩文件
			extracted_file = zfile.namelist()[0]  # 获取解压文件的原始名称
			zfile.extract(extracted_file, path='./code')  # 解压到 './code' 目录
		expert_data_set = np.load('code/data_set.npy', allow_pickle = True).item()
		obs = envs.reset()
		obs_ = pre_process(obs)
		i = 0
		while True:
			im = Image.fromarray(obs)
			im.save('code/imgs/' + str('screen') + '.jpeg')
			new_action = -1
			i += 1
			while (new_action not in action_map_inverse.keys()):
				# action范围是0,1,2,3,4,5,11,12
				expert_input = input(f'sample {i} expert input action ')
				new_action = -1 if expert_input == "" else int(expert_input)

			new_label = action_map_inverse[new_action]
			expert_data_set['data'].append(obs_)
			expert_data_set['label'].append(new_label)

			print('actual action', new_action)
			print('actual label', new_label)
			obs_next, reward, done, _ = envs.step(new_action)
			obs = obs_next
			obs_ = pre_process(obs)
			if done:
				obs = envs.reset()
				obs_ = pre_process(obs)
			if i % 10 == 0:
				# 保存 expert_data_set 为 .npy 文件
				np.save('code/data_set.npy', expert_data_set)

				# 将 .npy 文件压缩为 .zip 文件
				with zipfile.ZipFile('code/data_set.zip', 'w') as zipf:
					zipf.write('code/data_set.npy', arcname='data_set.npy', compress_type=zipfile.ZIP_DEFLATED)

    			# 删除原始 .npy 文件，保留压缩包
				os.remove('code/data_set.npy')
				print(f'i = {i} 数据保存并压缩完成：code/data_set.zip')



	train_data_set = {'data': [], 'label': []}

	# 建立经验专家
	with zipfile.ZipFile('code/data_set.zip', 'r') as zfile:
	    # 解压缩文件
		extracted_file = zfile.namelist()[0]  # 获取解压文件的原始名称
		zfile.extract(extracted_file, path='./data')  # 解压到 './data' 目录
		data_set = np.load('data/data_set.npy', allow_pickle = True).item()

		shutil.rmtree('data')
	

	expert_model = KNeighborsClassifier(n_neighbors=1)

	expert_data_batch = np.array(data_set['data'])  # 转换为二维数组 (n_samples, 16384)
	expert_label_batch = np.array(data_set['label'])  # 转换为一维数组 (n_samples,)
	expert_model.fit(expert_data_batch, expert_label_batch)  # 使用 KNN 模型进行训练


	epsilon = 1
	# start train your agent
	for i in range(num_updates): # performance0: 30000 / 300 = 100
		# an example of interacting with the environment
		# we init the environment and receive the initial observation
		
		obs = envs.reset()
		obs_ = pre_process(obs)

		# we get a trajectory with the length of args.num_steps
		for step in range(args.num_steps): # 250
			im = Image.fromarray(obs)
			im.save('code/imgs/' + str('screen') + '.jpeg')

			new_label = expert_model.predict(obs_.reshape(1,-1))[0]

			# 数据集里的label范围都是0~7
			train_data_set['data'].append(obs_)
			train_data_set['label'].append(new_label)

			# Sample actions
			if i == 0 or np.random.rand() < epsilon: # performance0
				# we choose an action by asking the expert
				action = action_map[new_label]
				print(f'expert chooses action {action}')
				query_cnt += 1

			else:
				# we choose a special action according to our model
				action_label = agent.select_action(obs_)
				action = action_map[action_label]
				print(f'agent chooses action {action}')

			# interact with the environment
			# we input the action to the environments and it returns some information
			# obs_next: the next observation after we do the action
			# reward: (float) the reward achieved by the action
			# down: (boolean)  whether it’s time to reset the environment again.
			#           done being True indicates the episode has terminated.
			obs_next, reward, done, _ = envs.step(action)
			# we view the new observation as current observation
			obs = obs_next
			obs_ = pre_process(obs)
			# if the episode has terminated, we need to reset the environment.
			if done:
				envs.reset()
				
		epsilon = epsilon * 0.95 # performance0

		# design how to train your model with labeled data
		agent.update(train_data_set['data'], train_data_set['label'])

		if (i + 1) % args.log_interval == 0:
			total_num_steps = (i + 1) * args.num_steps
			obs = envs.reset()
			obs_ = pre_process(obs)
			reward_episode_set = []
			reward_episode = 0
			# evaluate your model by testing in the environment
			for step in range(args.test_steps):
				im = Image.fromarray(obs)
				im.save('code/imgs/' + str('screen') + '.jpeg')
				action_label = agent.select_action(obs_)
				action = action_map[action_label]
				# you can render to get visual results
				# envs.render()
				obs_next, reward, done, _ = envs.step(action)
				reward_episode += reward
				obs = obs_next
				obs_ = pre_process(obs)
				if done or step == args.test_steps - 1:
					reward_episode_set.append(reward_episode)
					reward_episode = 0
					envs.reset()

			end = time.time()
			print(
				"TIME {} Updates {}, num timesteps {}, FPS {} \n query {}, avrage/min/max reward {:.1f}/{:.1f}/{:.1f}"
					.format(
					time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
					i, total_num_steps,
					int(total_num_steps / (end - start)),
					query_cnt,
					np.mean(reward_episode_set),
					np.min(reward_episode_set),
					np.max(reward_episode_set)
				))
			record['steps'].append(total_num_steps)
			record['mean'].append(np.mean(reward_episode_set))
			record['max'].append(np.max(reward_episode_set))
			record['min'].append(np.min(reward_episode_set))
			record['query'].append(query_cnt)
			plot(record)


if __name__ == "__main__":
	main()
