from pathlib import Path

import gymnasium as gym
import torch
from brain_octavus.brain import Octavus_arms_brain
from brain_octavus.reward import reward as rw
from world_octavus.environment import OctopusEnv


env = OctopusEnv(reward_fn=rw)

brain = Octavus_arms_brain()
#continue from the last saved brain instead of starting from zero
if Path("octavus.pt").exists():
    brain.load_state_dict(torch.load("octavus.pt"))
    print("continuing from octavus.pt")
#weights adjuster. Adam its the pattern: adjust each weight in the rigth size alone
# lr = learning rate. size of the adjusts
optimizer = torch.optim.Adam(brain.parameters(), lr=3e-4)
n_steps = 2048 #steps before learn
n_laps = 300000 #time of training ctrl+c to stop (saves every 10 laps)
gamma = 0.99
#gamma = future grades discount
clip = 0.2 #PPO small step: an action's chance changes at most 20% per lap
minibatch_size = 64
x, _ = env.reset(seed=0)

for lap in range(n_laps):
    memory_x, memory_y, memory_log_prob, memory_reward, memory_done = [], [], [], [], []
    reached = 0

    for step in range(n_steps): # for each action from model
        x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            y, distribuition = brain.act(x)
            log_prob = distribuition.log_prob(y).sum(-1)
            next_x, reward, terminated, truncated, info = env.step(y.numpy())
            memory_x.append(x)
            memory_y.append(y)
            memory_log_prob.append(log_prob)
            memory_reward.append(reward)
            memory_done.append(terminated or truncated)
            reached += info["reached"]

            x = next_x

            if terminated or truncated:
                x, _ = env.reset()

    xs = torch.stack(memory_x)
    ys = torch.stack(memory_y)
    old_log_probs = torch.stack(memory_log_prob)
    with torch.no_grad():
        model_predict_grades = brain.critic(xs).squeeze(-1)
        future = brain.critic(torch.tensor(x, dtype=torch.float32)).item()

    real_grade = torch.zeros(n_steps)
    for t in reversed(range(n_steps)):
        if memory_done[t]:
            future = 0.0
        future = memory_reward[t] + gamma * future
        real_grade[t] = future

    advantages = real_grade - model_predict_grades
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for epoch in range(10):#how many times it reviews the same memory
        #shuffle the 2048 memories and cut them into pieces of 64: one adjust per piece, 32 per review
        for chunk in torch.randperm(n_steps).split(minibatch_size):
            predicted = brain.critic(xs[chunk]).squeeze(-1)
            critic_loss = ((predicted - real_grade[chunk]) ** 2).mean()
            
            _, distribuition = brain.act(xs[chunk])
            new_log_probs = distribuition.log_prob(ys[chunk]).sum(-1)
            #i need the prob so i can do it happen more, increasing the prob
            #like the chance of this right y was 10% lets do this 12%
            ratio = (new_log_probs - old_log_probs[chunk]).exp()#show how much octavus change his idea
            clipped_ratio = ratio.clamp(1 - clip, 1 + clip)#dont let the brain change more than 20 about
            #some action
            #mean turns all the error into a average
            arm_loss = -torch.min(ratio * advantages[chunk], clipped_ratio * advantages[chunk]).mean()
            pre_tanh = brain.pre_tanh(xs[chunk]) #every number before its last Tanh
            saturation = ((pre_tanh.abs() -2).clamp(min=0) ** 2).mean()#clamp: what goes below the min turns into the min, gives back a copy (clamp_ changes it in place)
            #loss is the amount of errors loss=error or amount of gradiant or fault
            loss = arm_loss + 0.5 * critic_loss  + 0.1 * saturation
            #the optimizer just try to get this small so when this is big
            #it will adjust the weights to get it small

            #clean the last turn grade
            optimizer.zero_grad()
            #calculate each weitgh fault in the bad predict
            loss.backward() # the gradient is stored in the onw weight
            #adjust the weights
            optimizer.step()
            brain.exploration.data.clamp_(max=0.0)#this clamp; max 1 again: with Hill's muscles the draw is cheap (at 0.5 it rarely reached a target by chance, so it never learned where to go)

    print(f"lap {lap}: reward {sum(memory_reward):+.1f} | reached {reached} | exploration {brain.exploration.exp().mean():.2f}", flush=True)
    if lap % 10 == 0:
        torch.save(brain.state_dict(), "octavus.pt")
