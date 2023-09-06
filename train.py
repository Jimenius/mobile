import argparse
import random

import gym
import d4rl

import numpy as np
import torch


from models.nets import MLP
from models.actor_critic import MAPLEActor, MAPLECritic
from models.dist import TanhDiagGaussian
from models.dynamics_model import EnsembleDynamicsModel
from models.rnn import GRU_Model
from dynamics import EnsembleDynamics
from utils.scaler import StandardScaler
from utils.termination_fns import get_termination_fn
from utils.load_dataset import qlearning_dataset, load_neorl_dataset, normalize_rewards
from utils.buffer import ReplayTrajBuffer
from utils.logger import Logger, make_log_dirs
from utils.policy_trainer import PolicyTrainer
from policies import MAPLEPolicy
from configs import loaded_args


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="maple")
    parser.add_argument("--task", type=str, default="walker2d-medium-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--context-hidden-dims", nargs="*", default=(16,))
    parser.add_argument("--recurrent-hidden-units", type=int, default=128)
    parser.add_argument("--context-update-freq", type=int, default=1)
    parser.add_argument("--uncertainty-mode", type=str, default="aleatoric")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    known_args, _ = parser.parse_known_args()
    default_args = loaded_args[known_args.task]
    for arg_key, default_value in default_args.items():
        parser.add_argument(f'--{arg_key}', default=default_value, type=type(default_value))

    return parser.parse_args()


def train(args=get_args()):
    print(args)
    # create env and dataset
    assert args.domain in ["gym", "adroit", "neorl"]
    if args.domain == "neorl":
        # task, version, data_type = tuple(args.task.split("-"))
        # env = neorl.make(task+'-'+version)
        # dataset = load_neorl_dataset(env, data_type)
        raise ValueError("NeoRL not supported yet")
    else:
        env = gym.make(args.task)
        dataset = qlearning_dataset(env)
    rewards = dataset["rewards"]
    data_size = len(rewards)
    if args.norm_reward:
        # dataset = normalize_rewards(dataset)
        r_mean, r_std = dataset["rewards"].mean(), dataset["rewards"].std()
        dataset["rewards"] = (dataset["rewards"] - r_mean) / (r_std + 1e-3)

    args.obs_shape = env.observation_space.shape
    args.obs_dim = np.prod(args.obs_shape)
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    device = args.device
    actor_context_extractor = GRU_Model(
        args.obs_dim,
        args.action_dim,
        args.recurrent_hidden_units,
        device=device,
    ).to(device)
    actor_backbone = MLP(
        input_dim=args.obs_dim + args.context_hidden_dims[-1],
        hidden_dims=args.hidden_dims,
    )
    preprocess_net = MLP(
        args.recurrent_hidden_units,
        hidden_dims=args.context_hidden_dims,
    )
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = MAPLEActor(actor_backbone, preprocess_net, dist, device).to(device)
    actor_optim = torch.optim.Adam(
        [*actor_context_extractor.parameters(), *actor.parameters()],
        lr=args.actor_lr
    )
    critic_context_extractor = GRU_Model(
        args.obs_dim,
        args.action_dim,
        args.recurrent_hidden_units,
        device=device,
    ).to(device)
    critics = []
    for i in range(args.num_q_ensemble):
        ### Concatenate before feed into nets
        preprocess_net = MLP(
            args.recurrent_hidden_units,
            hidden_dims=args.context_hidden_dims,
        )
        critic_backbone = MLP(
            input_dim=args.obs_dim + args.context_hidden_dims[-1] + args.action_dim,
            hidden_dims=args.hidden_dims
        )
        critics.append(MAPLECritic(critic_backbone, preprocess_net, device))
    critics = torch.nn.ModuleList(critics).to(device)
    critics_optim = torch.optim.Adam(
        [*critic_context_extractor.parameters(), *critics.parameters()],
        lr=args.critic_lr
    )

    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    if args.uncertainty_mode in ("aleatoric", "pairwise-diff"):
        model_penalty_coef = args.penalty_coef
    else:
        model_penalty_coef = 0.0
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        penalty_coef=model_penalty_coef,
        uncertainty_mode=args.uncertainty_mode,
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)

    # create policy
    if args.uncertainty_mode == "lcb":
        penalty_coef = args.penalty_coef
    else:
        penalty_coef = 0.0
    policy = MAPLEPolicy(
        dynamics,
        actor_context_extractor,
        actor,
        critic_context_extractor,
        critics,
        actor_optim,
        critics_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        penalty_coef=penalty_coef,
        num_samples=args.num_samples,
        deterministic_backup=args.deterministic_backup,
        max_q_backup=args.max_q_backup
    )

    # create buffer
    real_buffer = ReplayTrajBuffer(
        buffer_size=data_size,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        max_context_horizon=args.rollout_length,
        context_dim=args.recurrent_hidden_units,
        device=device
    )
    real_buffer.load_dataset(
        dataset,
        actor_context_extractor,
        critic_context_extractor
    )

    fake_buffer = ReplayTrajBuffer(
        buffer_size=args.rollout_batch_size*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        max_context_horizon=args.rollout_length,
        context_dim=args.recurrent_hidden_units,
        device=device
    )

    # log
    log_dirs = make_log_dirs(
        args.task, args.algo_name, args.seed, vars(args),
        record_params=["penalty_coef", "rollout_length"]
    )
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = PolicyTrainer(
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        context_update_freq=args.context_update_freq,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )

    # train
    if not load_dynamics_model:
        dynamics.train(
            dataset,
            logger,
            max_epochs_since_update=args.max_epochs_since_update,
            max_epochs=args.dynamics_max_epochs
        )
    
    performance = policy_trainer.train()
    performance_str = f"last 10 performance {performance['last_10_performace']}"
    print(performance_str)
    logger.log(performance_str)

if __name__ == "__main__":
    train()