default_args = {
    "domain": "gym",
    "actor_lr": 1e-4,
    "critic_lr": 3e-4,
    "hidden_dims": [256, 256],
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.1,
    "auto_alpha": True,
    "target_entropy": None,
    "alpha_lr": 1e-4,
    "deterministic_backup": False,
    "max_q_backup": False,
    "norm_reward": False,
    "num_q_ensemble": 2,

    "dynamics_lr": 1e-3,
    "dynamics_max_epochs": None,
    "max_epochs_since_update": 5,
    "dynamics_hidden_dims": [200, 200, 200, 200],
    "dynamics_weight_decay": [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4],
    "n_ensemble": 20,
    "n_elites": 13,
    "rollout_freq": 1000,
    "rollout_batch_size": 50000,
    "rollout_length": 10,
    "penalty_coef": 1.0,
    "num_samples": 10,
    "model_retain_epochs": 5,
    "real_ratio": 0.05,
    "load_dynamics_path": "",

    "epoch": 3000,
    "step_per_epoch": 1000,
    "eval_freq": 1,
    "eval_episodes": 10,
    "batch_size": 256,
    "lr_scheduler": True,
}