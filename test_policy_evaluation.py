import gym
# import jbw
import argparse
import importlib
import time
import random
import numpy as np
import pickle
import tensorflow as tf
import torch
from GROUP_052.agent import Agent
import GROUP_052.sac.utils as utils
import os
from os import listdir, makedirs
from os.path import isfile, join
import wandb
from environments import JellyBeanEnv, MujocoEnv


def evaluate_agent(env, agent, video_recorder, num_eval_episodes, step):
    average_episode_reward = 0
    for episode in range(num_eval_episodes):
        obs = env.reset()
        agent.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            obs, reward, done, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward

        average_episode_reward += episode_reward
        video_recorder.save(f'{step}.mp4')
    average_episode_reward /= num_eval_episodes
    return average_episode_reward


def get_environment(env_type):
    '''Generates an environment specific to the agent type.'''
    if 'jellybean' in env_type:
        env = JellyBeanEnv(gym.make('JBW-COMP579-obj-v1'))
    elif 'mujoco' in env_type:
        env = MujocoEnv(gym.make('Hopper-v2'))
    else:
        raise Exception("ERROR: Please define your env_type to be either 'jellybean' or 'mujoco'!")
    return env


def save_checkpoint(state, filename="checkpoint.pth"):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent, exist_ok=True)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)


def train_agent(agent,
                env,
                env_eval,
                total_timesteps,
                evaluation_freq,
                n_episodes_to_evaluate,
                args,
                work_dir,
                video_recorder=None,
                replay_buffer=None,
                ):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tf.random.set_seed(args.seed)
    env.seed(args.seed)
    env_eval.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    step = 0
    episode, episode_reward, done = 0, 0, True
    _max_episode_steps = 1000
    array_of_mean_acc_rewards = []
    episode_reward_all = []
    current_best = 0
    while step < total_timesteps:
        if done:
            # evaluate agent periodically
            if step >= args.num_seed_steps and episode % 100 == 0:
                average_episode_reward = evaluate_agent(env, agent, video_recorder,
                                                        num_eval_episodes=n_episodes_to_evaluate, step=step)
                if args.use_wandb:
                    wandb.log({'iter': step, 'episode': episode, 'eval done average ep reward': average_episode_reward})

                # if average_episode_reward > current_best:
                #     current_best = average_episode_reward
                #     pickle_file = work_dir + f"/model_best.pt"
                #     with open(pickle_file, "wb") as fh:
                #         save_state = {'agent': agent,
                #                       'log_alpha_optimizer': agent.log_alpha_optimizer.state_dict(),
                #                       'actor_optimizer': agent.actor_optimizer.state_dict(),
                #                       'critic_optimizer': agent.critic_optimizer.state_dict(),
                #                       'step': step,
                #                       'episode': episode,
                #                       'episode_step': episode_step,
                #                       'episode_reward': average_episode_reward,
                #                       'done': done
                #                       }
                #         torch.save(save_state, fh)

            episode_reward_all.append(episode_reward)
            if args.use_wandb:
                wandb.log({'iter': step, 'episode': episode, 'train ep reward': episode_reward})

            obs = env.reset()
            agent.reset()

            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

        # sample action for data collection
        if step < args.num_seed_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=True)
        if step == args.num_seed_steps:
            print('starting updates')
        # run training update
        if step >= args.num_seed_steps:
            agent.update(replay_buffer, args.use_wandb, step, args.step_per_inter)

        next_obs, reward, done, _ = env.step(action)

        # # allow infinite bootstrap
        done = float(done)
        done_no_max = 0 if episode_step + 1 == _max_episode_steps else done
        episode_reward += reward

        # if (step <= args.reset_every) or ((step % args.reset_every) >= args.num_seed_steps):
        replay_buffer.add(obs, action, reward, next_obs, done,
                          done_no_max)

        obs = next_obs
        episode_step += 1
        step += 1

        if step % evaluation_freq == 0 and step >= args.num_seed_steps:
            # print(len(replay_buffer))
            avg_reward = evaluate_agent(env, agent, video_recorder, num_eval_episodes=n_episodes_to_evaluate, step=step)
            array_of_mean_acc_rewards.append(avg_reward)

            if args.use_wandb:
                wandb.log({'iter': step, 'episode': episode, 'eval average ep reward': avg_reward})
        # if (args.reset_every > 0) and (step % args.reset_every == 0):
        #     agent = Agent(env_specs=env_specs, device=args.device, hidden_dim=args.hidden_dim,
        #                   batch_size=args.batch_size)
            # agent.reinit_model()

        # if step % 100000 == 0:
        #     pickle_file = work_dir + f"/model_step_%d.pt" % step
        #     with open(pickle_file, "wb") as fh:
        #         save_state = {'agent': agent,
        #                       'log_alpha_optimizer': agent.log_alpha_optimizer.state_dict(),
        #                       'actor_optimizer': agent.actor_optimizer.state_dict(),
        #                       'critic_optimizer': agent.critic_optimizer.state_dict(),
        #                       'step': step,
        #                       'episode': episode,
        #                       'episode_step': episode_step,
        #                       'episode_reward': episode_reward,
        #                       'done': done
        #                       }
        #         torch.save(save_state, fh)

    # pickle_file = work_dir + f"/model_final.pt"
    # with open(pickle_file, "wb") as fh:
    #     save_state = {'agent': agent,
    #                   'log_alpha_optimizer': agent.log_alpha_optimizer.state_dict(),
    #                   'actor_optimizer': agent.actor_optimizer.state_dict(),
    #                   'critic_optimizer': agent.critic_optimizer.state_dict(),
    #                   'step': step,
    #                   'episode': episode,
    #                   'episode_step': episode_step,
    #                   'episode_reward': episode_reward,
    #                   'done': done
    #                   }
    #     torch.save(save_state, fh)
    return array_of_mean_acc_rewards


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--group', type=str, default='GROUP_052', help='group directory')
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--group_name", type=str, default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total_timesteps", type=int, default=2000000)
    parser.add_argument("--evaluation_freq", type=int, default=10000)
    parser.add_argument("--n_episodes_to_evaluate", type=int, default=20)
    parser.add_argument("--num_seed_steps", type=int, default=5000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--reset_every", type=int, default=200000)
    parser.add_argument("--step_per_inter", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--wandb_dir", type=str, default="/home/hattie/scratch/hopper")

    args = parser.parse_args()

    # set up wandb
    os.environ["WANDB_DIR"] = args.wandb_dir
    if args.use_wandb:
        group_vars = ['total_timesteps', 'seed', 'n_episodes_to_evaluate', 'num_seed_steps', 'hidden_dim',
                      'reset_every', 'step_per_inter', 'batch_size']

        group_name = ''
        for var in group_vars:
            group_name = group_name + '_' + var + str(getattr(args, var))
        wandb.init(project="comp579_sac",
                   group=args.group_name,
                   name=group_name)
        for var in group_vars:
            wandb.config.update({var: getattr(args, var)})

    path = './' + args.group + '/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    if ('agent.py' not in files) or ('env_info.txt' not in files):
        print("Your GROUP folder does not contain agent.py or env_info.txt!")
        exit()

    with open(path + 'env_info.txt') as f:
        lines = f.readlines()
    env_type = lines[0].lower()

    env = get_environment(env_type)
    env_eval = get_environment(env_type)
    if 'jellybean' in env_type:
        env_specs = {'scent_space': env.scent_space, 'vision_space': env.vision_space,
                     'feature_space': env.feature_space, 'action_space': env.action_space}
    if 'mujoco' in env_type:
        env_specs = {'observation_space': env.observation_space, 'action_space': env.action_space}
    # agent_module = importlib.import_module(args.group+'.agent')
    # agent = agent_module.Agent(env_specs)
    agent = Agent(env_specs=env_specs, device=args.device, hidden_dim=args.hidden_dim, batch_size=args.batch_size)
    work_dir = '/home/hattie/scratch/hopper/%s' % args.group_name
    replay_buffer = utils.ReplayBuffer(env_specs['observation_space'].shape,
                                       env_specs['action_space'].shape,
                                       args.total_timesteps,
                                       args.device)

    os.makedirs(work_dir + "/mujoco_results", exist_ok=True)
    video_recorder = utils.VideoRecorder(work_dir + "/mujoco_results")
    # Note these can be environment specific and you are free to experiment with what works best for you
    learning_curve = train_agent(agent, env, env_eval, total_timesteps=args.total_timesteps,
                                 evaluation_freq=args.evaluation_freq,
                                 n_episodes_to_evaluate=args.n_episodes_to_evaluate,
                                 args=args, work_dir=work_dir, video_recorder=video_recorder,
                                 replay_buffer=replay_buffer)

    pickle_file = work_dir + f"/learning_curve.pickle"
    with open(pickle_file, "wb") as fh:
        pickle.dump(learning_curve, fh)