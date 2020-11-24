from tf_agents.agents import Td3Agent
from tf_agents.agents.ddpg import actor_network, critic_network
import tensorflow as tf


def ACnetworks(environment, hyperparams) -> (actor_network, critic_network):
    observation_spec = environment.observation_spec()
    action_spec = environment.action_spec()

    actor_net = actor_network.ActorNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        fc_layer_params=hyperparams['actor_fc_layer_params'],
        dropout_layer_params=hyperparams['actor_dropout'],
        activation_fn=tf.nn.relu
    )

    critic_net = critic_network.CriticNetwork(
        input_tensor_spec=(observation_spec, action_spec),
        observation_fc_layer_params=hyperparams['critic_obs_fc_layer_params'],
        action_fc_layer_params=hyperparams['critic_action_fc_layer_params'],
        joint_fc_layer_params=hyperparams['critic_joint_fc_layer_params'],
        joint_dropout_layer_params=hyperparams['critic_joint_dropout'],
        activation_fn=tf.nn.relu
    )

    return (actor_net, critic_net)


def createAgent(environment, global_step, hyperparams) -> Td3Agent:
    actor_net, critic_net = ACnetworks(environment, hyperparams)

    tf_agent = Td3Agent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=hyperparams['actor_learning_rate']),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=hyperparams['critic_learning_rate']),
        exploration_noise_std=0.1,
        critic_network_2=None,
        target_actor_network=None,
        target_critic_network=None,
        target_critic_network_2=None,
        target_update_tau=hyperparams['target_update_tau'],
        target_update_period=hyperparams['target_update_period'],
        actor_update_period=hyperparams['actor_update_period'],
        dqda_clipping=None,
        td_errors_loss_fn=None,
        gamma=1.,
        reward_scale_factor=hyperparams['reward_scale_factor'],
        target_policy_noise=0.2,
        target_policy_noise_clip=0.5,
        gradient_clipping=None,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        train_step_counter=global_step,
        name=None)
    return tf_agent
