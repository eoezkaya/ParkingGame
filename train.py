import os
import sys
from statistics import fmean

# --- ALGORITHM TOGGLE (Simulated #ifdef) ---
USE_PPO = True  # Set to True for PPO, False for DQN
# -------------------------------------------

# 1. Fix the OpenMP duplicate error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt6.QtWidgets import QApplication
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from parking_env import ParkingEnv

# Explicit DQN Specification (12 parameters)
DV_SPEC_DQN = [
    ("learning_rate", float, 1e-5),
    ("exploration_fraction", float, 0.2),
    ("exploration_final_eps", float, 0.05),
    ("collision_penalty_magnitude", float, 20.0),
    ("distance_penalty_weight", float, 0.5),
    ("alignment_penalty_weight", float, 1.0),
    ("baseline_time_penalty_magnitude", float, 0.01),
    ("standing_reward_weight", float, 0.2),
    ("delta_reward_weight", float, 2.0),
    ("success_angle_threshold", float, 0.02),
    ("success_distance_threshold", float, 5.0),
    ("success_reward", float, 150.0),
]

# Explicit PPO Specification (10 parameters)
DV_SPEC_PPO = [
    ("learning_rate", float, 1e-5),
    ("collision_penalty_magnitude", float, 20.0),
    ("distance_penalty_weight", float, 0.5),
    ("alignment_penalty_weight", float, 1.0),
    ("baseline_time_penalty_magnitude", float, 0.01),
    ("standing_reward_weight", float, 0.2),
    ("delta_reward_weight", float, 2.0),
    ("success_angle_threshold", float, 0.02),
    ("success_distance_threshold", float, 5.0),
    ("success_reward", float, 150.0),
]


def load_hyperparams(file_path: str = "dv.dat") -> dict:
    spec = DV_SPEC_PPO if USE_PPO else DV_SPEC_DQN
    values = [default for _, _, default in spec]

    if not os.path.exists(file_path):
        algo_name = "PPO" if USE_PPO else "DQN"
        print(f"Warning: {file_path} not found. Using default {algo_name} hyperparameters.")
        return _build_hyperparams(values)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    expected = len(spec)
    if len(lines) < expected:
        algo_name = "PPO" if USE_PPO else "DQN"
        raise ValueError(
            f"{file_path} has {len(lines)} values, expected {expected} for {algo_name}. "
            "Please provide all required hyperparameters."
        )
    elif len(lines) > expected:
        print(f"Warning: {file_path} has {len(lines)} values, expected {expected}. Extra values will be ignored.")

    for idx, (name, cast_type, default) in enumerate(spec):
        if idx >= len(lines):
            continue

        raw_value = lines[idx]
        try:
            number = float(raw_value)
            values[idx] = int(number) if cast_type is int else float(number)
        except ValueError:
            print(
                f"Warning: invalid value '{raw_value}' for {name} in {file_path} line {idx + 1}. "
                f"Using default {default}."
            )

    return _build_hyperparams(values)


def _build_hyperparams(values: list) -> dict:
    if USE_PPO:
        return {
            "learning_rate": float(values[0]),
            "collision_penalty": -float(values[1]),
            "distance_penalty_weight": float(values[2]),
            "alignment_penalty_weight": float(values[3]),
            "baseline_time_penalty": -float(values[4]),
            "standing_reward_weight": float(values[5]),
            "delta_reward_weight": float(values[6]),
            "success_angle_threshold": float(values[7]),
            "success_distance_threshold": float(values[8]),
            "success_reward": float(values[9]),
        }
    else:
        return {
            "learning_rate": float(values[0]),
            "exploration_fraction": float(values[1]),
            "exploration_final_eps": float(values[2]),
            "collision_penalty": -float(values[3]),
            "distance_penalty_weight": float(values[4]),
            "alignment_penalty_weight": float(values[5]),
            "baseline_time_penalty": -float(values[6]),
            "standing_reward_weight": float(values[7]),
            "delta_reward_weight": float(values[8]),
            "success_angle_threshold": float(values[9]),
            "success_distance_threshold": float(values[10]),
            "success_reward": float(values[11]),
        }


class OverwriteModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_name = save_name

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_name)
            if self.verbose > 0:
                print(f"Checkpoint overwritten: {self.save_name}.zip")
        return True


def evaluate_model(model, env: ParkingEnv, app: QApplication, episodes: int = 1000) -> float:
    scores = []

    for episode_idx in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            
            if episode_idx % 10 == 0:
                app.processEvents()

        scores.append(env.calculate_final_score())
        print(f"Episode {episode_idx + 1}/{episodes} - Final Score: {scores[-1]:.2f}")

    return fmean(scores) if scores else 0.0

def train():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication(sys.argv)
    
    hyperparams = load_hyperparams("dv.dat")

    env = ParkingEnv(
        render_mode=None,
        enable_training_spawns=True,
        collision_penalty=hyperparams["collision_penalty"],
        distance_penalty_weight=hyperparams["distance_penalty_weight"],
        alignment_penalty_weight=hyperparams["alignment_penalty_weight"],
        baseline_time_penalty=hyperparams["baseline_time_penalty"],
        standing_reward_weight=hyperparams["standing_reward_weight"],
        delta_reward_weight=hyperparams["delta_reward_weight"],
        success_angle_threshold=hyperparams["success_angle_threshold"],
        success_distance_threshold=hyperparams["success_distance_threshold"],
        success_reward=hyperparams["success_reward"],
    )
    
    try:
        print("Checking environment compatibility...")
        check_env(env)
        policy_kwargs = dict(net_arch=[256, 256])
        
        algo_str = "ppo" if USE_PPO else "dqn"
        model_save_name = f"{algo_str}_parking_model"

        if USE_PPO:
            print("Initializing PPO model...")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                learning_rate=float(hyperparams["learning_rate"]),
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                device="auto"
            )
        else:
            print("Initializing DQN model...")
            model = DQN(
                "MlpPolicy",
                env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                learning_rate=float(hyperparams["learning_rate"]),
                buffer_size=200000,
                batch_size=64,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                target_update_interval=10000,
                exploration_fraction=float(hyperparams["exploration_fraction"]),
                exploration_final_eps=float(hyperparams["exploration_final_eps"]),
                device="auto"
            )

        checkpoint_callback = OverwriteModelCallback(
            save_freq=1000000,
            save_name=model_save_name,
            verbose=1
        )

        print(f"Starting training with {algo_str.upper()}...")
        model.learn(
            total_timesteps=10_000_000,
            callback=checkpoint_callback,
            progress_bar=True
        )

        model.save(model_save_name)
        print(f"Model saved as {model_save_name}.zip")

        # Evaluation
        env.set_training_spawns_disabled()
        print("Running post-training evaluation for 1000 episodes...")
        mean_score = -1.0 * evaluate_model(model, env, app, episodes=1000)
        
        with open("score.dat", "w", encoding="utf-8") as score_file:
            score_file.write(f"{mean_score:.6f}\n")
        print(f"Mean final score over 1000 episodes: {mean_score:.6f}")
        print("Saved mean score to score.dat")

    finally:
        print("Closing environment...")
        env.close()

if __name__ == "__main__":
    train()