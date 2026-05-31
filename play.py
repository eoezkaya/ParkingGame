import os
import sys
import time
import numpy as np
import imageio.v2 as imageio

# Fix for macOS OpenMP duplicate initialization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage
from stable_baselines3 import DQN
from parking_env import ParkingEnv


def _capture_env_frame(env: ParkingEnv) -> np.ndarray:
    """Capture the current Qt widget frame as an RGB numpy array."""
    image = env.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    width = image.width()
    height = image.height()
    raw = image.bits().asstring(image.sizeInBytes())
    frame_rgba = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    return frame_rgba[:, :, :3].copy()

def play():
    # 1. Initialize QApplication
    app = QApplication(sys.argv)

    # 2. Load the environment in human mode
    env = ParkingEnv(render_mode="human")
    env.show_score_overlay = True
    env.show_lidar = False

    # Exit immediately via os._exit when the window is closed to avoid
    # a segfault caused by Qt destroying the C++ object before Python GC runs.
    app.lastWindowClosed.connect(lambda: os._exit(0))

    # 3. Load the trained model
    model_path = "dqn_parking_model.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run train.py first.")
        return

    print(f"Loading model: {model_path}")
    model = DQN.load(model_path)

    # 4. Play multiple episodes and record video
    num_episodes = 5
    video_path = "playback.mp4"
    fps = 30
    writer = imageio.get_writer(video_path, fps=fps)

    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0

            print(f"Starting Episode {episode + 1}/{num_episodes}")

            # Capture the initial frame of each episode.
            env.update()
            QApplication.processEvents()
            writer.append_data(_capture_env_frame(env))

            while not done:
                # Tell the model to predict the best action (deterministic=True is key for testing)
                action, _states = model.predict(obs, deterministic=True)

                # Take the step in the environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                done = terminated or truncated

                # Capture each rendered frame.
                writer.append_data(_capture_env_frame(env))

                # Brief sleep to make the movement look natural for human eyes
                # Increase this if the car moves too fast to see
                time.sleep(0.01)

            final_score = env.calculate_final_score()
            print(
                f"Episode {episode + 1} finished. Total Reward: {total_reward:.2f} | "
                f"Final Score: {final_score:.2f}/100"
            )
            env.update()
            QApplication.processEvents()
            # Let Qt process events for 1 second between episodes instead of blocking sleep
            end_time = time.time() + 1.0
            while time.time() < end_time:
                QApplication.processEvents()
                writer.append_data(_capture_env_frame(env))
    finally:
        writer.close()
        print(f"Saved playback video to {video_path}")

    print("Playback complete. Window stays open until closed.")
    sys.exit(app.exec())

if __name__ == "__main__":
    play()