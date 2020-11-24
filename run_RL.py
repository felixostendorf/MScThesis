def Mnorm(rho, M):
    norm = rho.T.dot(M).dot(rho)
    return np.sqrt(norm)


def evaluate10d():
    X_eval, Y_eval = [], []

    for _ in range(10_000):
        sample = tf.reshape(tf.random.uniform(
            shape=eval_env.observation_spec().shape, minval=-1, maxval=1), [-1, 10])
        ts = time_step.TimeStep(step_type=np.array([1]).astype(np.int32),
                                reward=np.array(
                                    [0.]).astype(np.float32),
                                discount=np.array(
                                    [1.]).astype(np.float32),
                                observation=sample)
        ac = agent_eval_policy.action(
            ts).action / hyperparams['scale']
        ac_norm = np.linalg.norm(ac.numpy().reshape(10), 1)
        norm_point = (
            hyperparams['wealth']**0.25) / hyperparams['scale'] * Mnorm(sample.numpy().reshape(10), M)

        if ac_norm < 0.01:
            Y_eval.append(0)
        else:
            Y_eval.append(ac_norm)
        X_eval.append(norm_point)

    sns.scatterplot(X_eval, Y_eval)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14, rotation=0)
    plt.savefig('./10dFigures/eval10d' + str(step)+'.png')
    plt.close()

    plt.plot(average_rewards)
    plt.savefig('./10dFigures/avgReward.png')
    plt.close()


def evaluate2d():
    df = pd.DataFrame(
        index=hyperparams['eval_grid'], columns=hyperparams['eval_grid'])
    df2 = pd.DataFrame(
        index=hyperparams['eval_grid'], columns=hyperparams['eval_grid'])
    for k in df.index:
        for j in df.columns:
            ts = time_step.TimeStep(step_type=np.array([1]).astype(np.int32),
                                    reward=np.array(
                                        [0.]).astype(np.float32),
                                    discount=np.array(
                                        [1.]).astype(np.float32),
                                    observation=np.array([[k/(hyperparams['wealth']/hyperparams['scale']),
                                                           j/(hyperparams['wealth']/hyperparams['scale'])]]).astype(np.float32))
            norm = np.linalg.norm(
                agent_eval_policy.action(ts).action.numpy(), 1) / hyperparams['scale']
            df2.loc[k, j] = norm
            if norm < 0.01:
                df.loc[k, j] = 0
            else:
                df.loc[k, j] = 1

    df.columns = df.columns/hyperparams['wealth']
    df.columns = df.columns.to_series().apply(lambda x: np.round(x, 2))
    df.index = df.index/hyperparams['wealth']
    df.index = df.index.to_series().apply(lambda x: np.round(x, 2))
    df = df.astype(float)
    ax = sns.heatmap(df, cbar=False, xticklabels=16, yticklabels=16)
    ax.invert_yaxis()
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14, rotation=0)
    plt.savefig('./plots_RL/' + str(step)+'.png')
    plt.close()

    df2.columns = df2.columns/hyperparams['wealth']
    df2.columns = df2.columns.to_series().apply(lambda x: np.round(x, 2))
    df2.index = df2.index/hyperparams['wealth']
    df2.index = df2.index.to_series().apply(lambda x: np.round(x, 2))
    df2 = df2.astype(float)
    ax = sns.heatmap(df2, xticklabels=16, yticklabels=16,
                     cmap=sns.cubehelix_palette(start=0, gamma=0.4, light=1, dark=0, reverse=True, as_cmap=True))
    ax.invert_yaxis()
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=14)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14, rotation=0)
    plt.savefig('./plots_RL/' + str(step)+'_2.png')
    plt.close()


if __name__ == '__main__':
    import tensorflow as tf
    from tf_agents.drivers import dynamic_step_driver
    from tf_agents.policies import random_tf_policy, policy_saver
    from tf_agents.environments import tf_py_environment
    from tf_agents.replay_buffers import tf_uniform_replay_buffer
    from tf_agents.trajectories import time_step
    from tf_agents.utils import common

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import sys
    import pickle

    from utils.valueFct import MertonValFct
    from utils.power_utility import PowerUtility
    from rl.environment import TransCostEnv
    from rl.agent import createAgent

    global_step = tf.compat.v1.train.get_or_create_global_step()

    hyperparams = dict(
        scale=10,  # 3, # 10
        reward_scale=1e20,  # 1e6, # 1e12
        wealth=5_000_000,  # 100, # 10_000

        num_steps=2_000_000,

        initial_collect_steps=10_000,
        gradient_steps=1,
        collect_steps_per_iteration=1,
        replay_buffer_capacity=1_000_000,

        batch_size=256,

        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        target_update_tau=0.005,
        target_update_period=2,
        reward_scale_factor=1,
        actor_update_period=2,

        actor_fc_layer_params=(128, 128),
        actor_dropout=None,
        critic_obs_fc_layer_params=None,
        critic_action_fc_layer_params=None,
        critic_joint_fc_layer_params=(128, 128),
        critic_joint_dropout=None,

        log_interval=1_000,
        eval_interval=50_000,
        plot_interval=4_000_000,
        checkpoint_dir='./rl_saver/checkpoint',
        policy_dir='./rl_saver/policy',
        # np.arange(-32, 33,1) #np.arange(-1024, 1040, 32)
        eval_grid=np.arange(-2**19, 2**19 + 2**14, 2**14)
    )
    # for testing 10d, load parameter settings from para file created in utils.fixed_cost.py
    # with open('para.pickle', 'rb') as f:
    #    market_para = pickle.load(f)
    #mu =  market_para['mu']
    #covM = market_para['Cov']
    #M = market_para['M']

    util = PowerUtility(3)
    mu = np.array([0.08, 0.08])
    covM = np.array([[0.4**2, 0.3*0.4**2], [0.3*0.4**2, 0.4**2]])
    ValFct = MertonValFct(utility=util, imp_rate=1,
                          int_rate=0.03, drift=mu,
                          covM=covM, dRiskyAssets=2)

    # environment to collect observations for training the agent
    train_env = tf_py_environment.TFPyEnvironment(
        TransCostEnv(wealth=hyperparams['wealth'], scale=hyperparams['scale'],
                     reward_scale=hyperparams['reward_scale'], steps=500,
                     dRiskyAsset=2, ValueFn=ValFct, PropCost=0.03))
    # environment to run evaluation policy
    eval_env = tf_py_environment.TFPyEnvironment(
        TransCostEnv(wealth=hyperparams['wealth'], scale=hyperparams['scale'],
                     reward_scale=hyperparams['reward_scale'], steps=1_000_000,
                     dRiskyAsset=2, ValueFn=ValFct, PropCost=0.03))

    agent = createAgent(train_env, global_step, hyperparams)
    agent.initialize()

    # evaluation policy
    agent_eval_policy = agent.policy
    # collection policy
    agent_collect_policy = agent.collect_policy
    # random policy for replay buffer initialization
    initial_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec())

    # replay buffer to store observation
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=hyperparams['replay_buffer_capacity'])

    # driver to initialize replay buffer
    init_agent_collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        initial_policy,
        observers=[replay_buffer.add_batch],
        num_steps=hyperparams['initial_collect_steps'])
    init_agent_collect_driver.run()

    # driver to collect observations for training the agent
    agent_collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent_collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=hyperparams['collect_steps_per_iteration'])

    # Dataset generates trajectories with shape [BatchSizex2x...]
    agent_dataset = replay_buffer.as_dataset(
        num_parallel_calls=4,
        sample_batch_size=hyperparams['batch_size'],
        num_steps=2).prefetch(3)
    agent_iterator = iter(agent_dataset)

    agent.train = common.function(agent.train)
    agent_collect_driver.run = common.function(
        agent_collect_driver.run)

    # Reset the train step
    agent.train_step_counter.assign(0)
    step = agent.train_step_counter.numpy()

    # for collecting results and losses
    critic_loss = []
    actor_loss = []
    average_rewards = [0.]
    eval_results = dict()
    eval_results['params'] = hyperparams
    train_checkpointer = common.Checkpointer(
        ckpt_dir=hyperparams['checkpoint_dir'],
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    tstep = train_env.reset()

    while step < hyperparams['num_steps']:

        eval_ts = eval_env.reset()

        agent_collect_driver.run()
        print(step,
              train_env.current_time_step().reward.numpy(),
              train_env.pyenv.envs[0]._avg_reward,
              eval_env.pyenv.envs[0]._avg_reward)
        eval_ts = eval_env._step(agent_eval_policy.action(eval_ts).action)

        for i in range(hyperparams['gradient_steps']):
            experience, _ = next(agent_iterator)
            train_loss = agent.train(experience)
            step = agent.train_step_counter.numpy()

            if step % hyperparams['log_interval'] == 0:
                actor_loss.append(train_loss[1][0])
                critic_loss.append(train_loss[1][1])
                average_rewards.append(eval_env.pyenv.envs[0]._avg_reward)

            if step % hyperparams['eval_interval'] == 0:
                # evaluate10d()

                # evaluate2d()

                train_checkpointer.save(global_step)
                tf_policy_saver.save(hyperparams['policy_dir'])

                eval_results['avg_reward'] = average_rewards
                eval_results['actor_loss'] = actor_loss
                eval_results['critic_loss'] = critic_loss

                with open('./results/TD3.pickle', 'wb') as f:
                    pickle.dump(eval_results, f)

            if step % hyperparams['plot_interval'] == 0:
                plt.plot(average_rewards)
                plt.show()
                plt.plot(actor_loss)
                plt.show()
                plt.plot(critic_loss)
                plt.show()

    print('Finish')
