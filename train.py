import os
import sys
from statistics import fmean

# 1. Fix the OpenMP duplicate error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt6.QtWidgets import QApplication
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from parking_env import ParkingEnv


DV_SPEC = [
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


def load_hyperparams(file_path: str = "dv.dat") -> dict:
    values = [default for _, _, default in DV_SPEC]

    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Using default hyperparameters.")
        return _build_hyperparams(values)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    expected = len(DV_SPEC)
    if len(lines) < expected:
        raise ValueError(
            f"{file_path} has {len(lines)} values, expected {expected}. "
            "Please provide all required hyperparameters."
        )
    elif len(lines) > expected:
        print(f"Warning: {file_path} has {len(lines)} values, expected {expected}. Extra values will be ignored.")

    for idx, (name, cast_type, default) in enumerate(DV_SPEC):
        if idx >= len(lines):
            continue

        raw_value = lines[idx]
        try:
            number = float(raw_value)
            if cast_type is int:
                values[idx] = int(number)
            else:
                values[idx] = float(number)
        except ValueError:
            print(
                f"Warning: invalid value '{raw_value}' for {name} in {file_path} line {idx + 1}. "
                f"Using default {default}."
            )

    return _build_hyperparams(values)


def _build_hyperparams(values: list) -> dict:
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


def evaluate_model(model: DQN, env: ParkingEnv, episodes: int = 1000) -> float:
    scores = []

    for episode_idx in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

        scores.append(env.calculate_final_score())
        print(f"Episode {episode_idx + 1}/{episodes} - Final Score: {scores[-1]:.2f}")

    return fmean(scores) if scores else 0.0

def train():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # 2. Initialize QApplication BEFORE the Environment
    app = QApplication(sys.argv)
    hyperparams = load_hyperparams("dv.dat")

    # 3. Create the environment
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
    
    print("Checking environment compatibility...")
    check_env(env)
    policy_kwargs = dict(net_arch=[256, 256])
    # 4. Define the DQN model
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

    # 5. Save checkpoints at constant intervals by overwriting one file
    checkpoint_callback = OverwriteModelCallback(
        save_freq=1000000,
        save_name="dqn_parking_model",
        verbose=1
    )

    # 6. Train
    print("Starting training...")
    model.learn(
        total_timesteps=10_000_000,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 7. Save final model
    model.save("dqn_parking_model")
    print("Model saved as dqn_parking_model.zip")

    
    # 8. Evaluate 1000 episodes and write mean score to score.dat
    env.set_training_spawns_disabled()
    print("Running post-training evaluation for 1000 episodes...")
    mean_score = evaluate_model(model, env, episodes=1000)
    with open("score.dat", "w", encoding="utf-8") as score_file:
        score_file.write(f"{mean_score:.6f}\n")
    print(f"Mean final score over 1000 episodes: {mean_score:.6f}")
    print("Saved mean score to score.dat")

if __name__ == "__main__":
    train()
