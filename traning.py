import gymnasium as gym
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import numpy as np
import math
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob

class Critic(nn.Module):
    def __init__(self, state_dim,net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)
        return v

class PPO_agent():
	def __init__(self):
	
		self.state_dim = 8
		self.action_dim = 4
		self.net_width = 64
		self.T_horizon = 2048
		self.batch_size =  64
		
		# Training parameters
		self.gamma = 0.99
		self.lambd = 0.95
		self.clip_rate = 0.2
		self.K_epochs = 10
		self.entropy_coef = 1e-3
		self.entropy_coef_decay = 0.99
		self.l2_reg = 1e-3

		# Build actor and optimizer
		self.actor = Actor(self.state_dim, self.action_dim, self.net_width).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=2e-4)

		# Build Critic and optimizer
		self.critic = Critic(self.state_dim, self.net_width).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-4)

		# Build Trajectory holder
		self.s_hoder = torch.zeros((self.T_horizon, self.state_dim), dtype=torch.float32).to(device)
		self.a_hoder = torch.zeros((self.T_horizon, self.action_dim), dtype=torch.float32).to(device)
		self.r_hoder = torch.zeros((self.T_horizon, 1), dtype=torch.float32).to(device)
		self.s_next_hoder = torch.zeros((self.T_horizon, self.state_dim), dtype=torch.float32).to(device)
		self.logprob_a_hoder = torch.zeros((self.T_horizon, self.action_dim), dtype=torch.float32).to(device)
		self.done_hoder = torch.zeros((self.T_horizon, 1), dtype=torch.float32).to(device)
		self.dw_hoder = torch.zeros((self.T_horizon, 1), dtype=torch.float32).to(device)

	def put_data(self, now_state, action, reward, next_state, logprob_a, done, dw, idx):
		self.s_hoder[idx] = torch.from_numpy(now_state).float().to(device)
		self.a_hoder[idx] = torch.tensor(action).float().to(device)
		self.s_next_hoder[idx] = torch.from_numpy(next_state).float().to(device)
		self.logprob_a_hoder[idx] = torch.tensor(logprob_a, dtype=torch.float32).to(device)
		self.r_hoder[idx] = torch.tensor([reward], dtype=torch.float32).to(device)
		self.done_hoder[idx] = torch.tensor([done], dtype=torch.float32).to(device)
		self.dw_hoder[idx] = torch.tensor([dw], dtype=torch.float32).to(device)

	def select_action(self, s, deterministic):
		s = torch.from_numpy(s).float().to(device)
		with torch.no_grad():
			pi = self.actor.pi(s, softmax_dim=0)
			if deterministic:
				a = torch.argmax(pi).item()
				return a, None
			else:
				m = Categorical(pi)
				a = m.sample().item()
				pi_a = pi[a].item()
				return a, pi_a

			
	def train(self):
		self.entropy_coef *= self.entropy_coef_decay #exploring decay
		'''Prepare PyTorch data from Numpy data'''
		s = self.s_hoder
		a = self.a_hoder
		r = self.r_hoder
		s_next = self.s_next_hoder
		old_prob_a = self.logprob_a_hoder
		done = self.done_hoder
		dw = self.dw_hoder

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(s)
			vs_ = self.critic(s_next)

			'''dw(dead and win) for TD_target and Adv'''
			deltas = r + self.gamma * vs_ * (1-dw) - vs
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (1-done)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(device)
			td_target = adv + vs

		"""PPO update"""
		#Slice long trajectopy into short trajectory and perform mini-batch PPO update
		optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))

		for _ in range(self.K_epochs):
			#Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(device)
			s, a, td_target, adv, old_prob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

			'''mini-batch PPO update'''
			for i in range(optim_iter_num):
				index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))

				'''actor update'''
				prob = self.actor.pi(s[index], softmax_dim=1)
				entropy = Categorical(prob).entropy().sum(0, keepdim=True)
				prob_a = prob.gather(1, a[index].long())
				ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))  # a/b == exp(log(a)-log(b))

				
				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

				'''critic update'''
				c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
				for name, param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

def test_for_save(agent):
	env_test = gym.make("LunarLander-v3")
	run_reward = 0
	done = False
	now_state = env_test.reset()[0]
	while not done :
		action, logprob = agent.select_action(now_state, deterministic=True)
		next_state, reward, dw, tr, info = env_test.step(action)

		run_reward += reward
		done = (dw or tr)
		now_state = next_state
	
	env_test.close()
	return int(run_reward)

def main():
	env = gym.make("LunarLander-v3")
	agent = PPO_agent()
	best_run = 100
	totel_train, traj_lenth, totel_step = 0, 0, 0
	while totel_train < 1000:
		run_reward = 0
		done = False
		now_state = env.reset()[0]
		while not done :
			action, logprob = agent.select_action(now_state, deterministic=False)
			next_state, reward, dw, tr, info = env.step(action)

			totel_step += 1
			run_reward += reward
			done = (dw or tr)

			agent.put_data(now_state, action, reward, next_state, logprob, done, dw, idx = traj_lenth)
			now_state = next_state
			traj_lenth += 1

			'''Update if its time'''
			if traj_lenth % 2048 == 0:
				totel_train += 1
				agent.train()
				traj_lenth = 0

		print(f"totel_step : {totel_step} | totel_train : {totel_train} | run_reward : {run_reward}")

		if run_reward > 100:
			best = test_for_save(agent)
			print(f"best : {best}")
			if best > best_run:
				best_run = best
				torch.save(agent.actor.state_dict(), f"actor-{best_run}.pt")

main()