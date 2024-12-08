import gymnasium as gym
import torch.nn.functional as F
import torch.nn as nn
import torch

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

class PPO_agent():
	def __init__(self):
		self.state_dim = 8
		self.action_dim = 4
		self.net_width = 64
		# Build actor
		self.actor = Actor(self.state_dim, self.action_dim, self.net_width).to(device)
		
		# Load the pre-trained model weights for the Actor network
		self.actor.load_state_dict(torch.load("your model name")) # like "actor-206.pt"

		# Switch the Actor network to evaluation mode (
		self.actor.eval()
		
	def select_action(self, s):
		s = torch.from_numpy(s).float().to(device)
		with torch.no_grad():
			pi = self.actor.pi(s, softmax_dim=0)
			a = torch.argmax(pi).item()
			return a, None
			
def main():
	env = gym.make("LunarLander-v3", render_mode = "human")
	agent = PPO_agent()
	totel_step = 0
	while True:
		run_reward = 0
		done = False
		now_state = env.reset()[0]
		while not done :
			action, logprob = agent.select_action(now_state)
			next_state, reward, dw, tr, info = env.step(action)

			totel_step += 1
			run_reward += reward
			done = (dw or tr)
			now_state = next_state

		print(run_reward)
	

main()