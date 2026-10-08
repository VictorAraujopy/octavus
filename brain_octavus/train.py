import math
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
from brain_octavus.brain import Octavus_arms_brain
from brain_octavus.reward import reward as rw
from world_octavus.environment import OctopusEnv


n_octopuses = 4 #octopuses playing at once, each on its own core: ~3x the steps per second (565 vs 176)
n_steps = 2048 #steps before learn, from all the octopuses together
steps_per_octopus = n_steps // n_octopuses #512 each
n_laps = 300000 #time of training ctrl+c to stop (saves every 10 laps)
gamma = 0.995
#gamma = future grades discount
#0.995: a reward 5 s (200 steps) later still counts 0.37 in an action's grade, 0.99 only reached ~2.5 s:
#the hard tilted jet looked good (2 s rising) and the flip at 3 s barely counted. 5 s = its whole jet breath
gae_lambda = 0.95 #GAE: how far the surprises of the coming steps reach into this step's grade (see below)
clip = 0.2 #PPO small step: an action's chance changes at most 20% per lap
minibatch_size = 64
#the draw's size (exploration) can't go past 0.5: with posture arms a draw of 1 threw every arm from end to end
#each step, and the brain hid in the limits (where the clip eats the draw), curled into a ball
exploration_cap = math.log(0.5) #exploration is stored as a log: exp(log(0.5)) = 0.5


def normalize(raw_x):
    #updates the running average and spread with these readings (one row per octopus), then hands them scaled by them
    raw_x = torch.as_tensor(raw_x, dtype=torch.float32)
    for reading in raw_x:
        senses["count"] += 1
        difference = reading - senses["mean"]
        senses["mean"] += difference / senses["count"]
        senses["var"] += (difference * (reading - senses["mean"]) - senses["var"]) / senses["count"]
    return ((raw_x - senses["mean"]) / (senses["var"] + 1e-8).sqrt()).clamp(-10, 10)


#each parallel octopus runs in a new Python process that reads this file again: only the first one may train
if __name__ == "__main__":
    #DISABLED: an octopus only starts a new episode when this loop says so (it needs to see how the last one ended)
    envs = gym.vector.AsyncVectorEnv([lambda: OctopusEnv(reward_fn=rw) for _ in range(n_octopuses)],
                                     autoreset_mode=gym.vector.AutoresetMode.DISABLED)

    brain = Octavus_arms_brain()
    #each sense's running average and spread: the brain gets every sense as "how far from normal" (mean 0, spread 1).
    #On a test robot (HalfCheetah) this, with GAE and the adjust cap below, took this trainer from 330 to 4134
    senses = {"mean": torch.zeros(envs.single_observation_space.shape), "var": torch.ones(envs.single_observation_space.shape), "count": 1e-4}
    #continue from the last saved brain instead of starting from zero
    if Path("octavus.pt").exists():
        saved = torch.load("octavus.pt")
        brain.load_state_dict(saved["brain"])
        senses = saved["senses"]
        print("continuing from octavus.pt")
    brain.exploration.data.clamp_(max=exploration_cap) #from the first lap, not only after the first adjust
    #weights adjuster. Adam its the pattern: adjust each weight in the rigth size alone
    # lr = learning rate. size of the adjusts
    optimizer = torch.optim.Adam(brain.parameters(), lr=3e-4)
    x = normalize(envs.reset(seed=0)[0])

    for lap in range(n_laps):
        memory_x, memory_y, memory_log_prob, memory_reward, memory_done, memory_cut_grade = [], [], [], [], [], []
        speed = 0.0 #meters per second in the episode's direction, summed over the lap's steps

        for step in range(steps_per_octopus): # for each action from model, all the octopuses at once (one row each)
            with torch.no_grad():
                y, distribuition = brain.act(x)
                log_prob = distribuition.log_prob(y).sum(-1)
                next_x, reward, terminated, truncated, info = envs.step(y.numpy())
                next_x = normalize(next_x)
                done = terminated | truncated
                memory_x.append(x)
                memory_y.append(y)
                memory_log_prob.append(log_prob)
                memory_reward.append(torch.tensor(reward, dtype=torch.float32))
                memory_done.append(torch.tensor(done))
                #the 20 s clock cut the episode, it isn't a real end: what the critic expects from there still counts
                #(counting it as 0 taught it the world ends at 20 s)
                cut = torch.tensor(truncated & ~terminated)
                memory_cut_grade.append(torch.where(cut, brain.critic(next_x).squeeze(-1), 0.0))
                speed += info["speed"].mean()

                x = next_x

                if done.any():
                    #a new episode only for the octopuses whose episode just ended
                    restarted_x, _ = envs.reset(options={"reset_mask": done})
                    x[torch.tensor(done)] = normalize(restarted_x[done])

        xs = torch.stack(memory_x) #steps x octopuses x senses
        ys = torch.stack(memory_y)
        old_log_probs = torch.stack(memory_log_prob)
        rewards = torch.stack(memory_reward)
        dones = torch.stack(memory_done)
        cut_grades = torch.stack(memory_cut_grade)
        with torch.no_grad():
            model_predict_grades = brain.critic(xs).squeeze(-1)
            future = brain.critic(x).squeeze(-1)

        #GAE: an action's advantage is its surprise (reward now + the critic's guess for the next step - its guess for
        #this one), plus the coming steps' surprises fading by gamma * gae_lambda. Summing the raw rewards to the end
        #was much noisier: on the test robot it peaked at 330 and collapsed. Each octopus has its own column
        advantages = torch.zeros(steps_per_octopus, n_octopuses)
        surprises_ahead = torch.zeros(n_octopuses)
        for t in reversed(range(steps_per_octopus)):
            #where an episode ended, what comes after is its end: 0, or the critic's guess if the clock cut it
            future = torch.where(dones[t], cut_grades[t], future)
            surprises_ahead = torch.where(dones[t], 0.0, surprises_ahead)
            surprise = rewards[t] + gamma * future - model_predict_grades[t]
            surprises_ahead = surprise + gamma * gae_lambda * surprises_ahead
            advantages[t] = surprises_ahead
            future = model_predict_grades[t]
        real_grade = advantages + model_predict_grades #what the critic should have predicted

        #from here on it doesn't matter which octopus: every step of every one is one memory
        xs, ys, old_log_probs = xs.flatten(0, 1), ys.flatten(0, 1), old_log_probs.flatten(0, 1)
        advantages, real_grade = advantages.flatten(), real_grade.flatten()
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
                #cap the size of each adjust: without it, one big jump threw away what it had learned
                nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
                #adjust the weights
                optimizer.step()
                brain.exploration.data.clamp_(max=exploration_cap)#keep the draw at 0.5 or less (the cap of 1 was for reaching a target by chance)

        print(f"lap {lap}: reward {rewards.sum():+.1f} | speed {speed / steps_per_octopus:+.3f} m/s | exploration {brain.exploration.exp().mean():.2f}", flush=True)
        if lap % 10 == 0:
            #the senses' average and spread go with the brain: it only understands senses scaled the same way
            torch.save({"brain": brain.state_dict(), "senses": senses}, "octavus.pt")
