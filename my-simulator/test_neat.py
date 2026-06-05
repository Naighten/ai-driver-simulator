import pickle

import neat
import numpy as np
import pygame

from CarGameEnv import CarGameEnv
from neatmodel.utils import CustomGenome


def load_genome_and_config(genome_path, config_path):
    """
    Load the best genome and NEAT config, build the neural network.
    """
    # Load the genome from disk
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Load NEAT configuration
    config = neat.Config(
        CustomGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create the FeedForwardNetwork
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net


def test_env(net, episodes=10, max_steps=1000):
    """
    Run the CarGameEnv for a number of episodes using the provided NEAT network.
    Prints per-episode rewards and returns the list of total rewards.
    """
    pygame.init()
    rewards = []

    for ep in range(1, episodes + 1):
        # Initialize environment
        env = CarGameEnv(track_name="newtrack7", render_mode="human")
        clock = pygame.time.Clock()
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            clock.tick(env.game.FPS)

            # Activate network to get action
            action = np.array(net.activate(obs), dtype=np.float32)

            # Step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                print(terminated, truncated)

            # Render the environment
            env.render()
            step += 1

        rewards.append(total_reward)
        print(f"Episode {ep}: Reward = {total_reward}")

        env.close()

    pygame.quit()

    avg_reward = np.mean(rewards)
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward}")
    return rewards


if __name__ == "__main__":
    # Paths to the stored genome and NEAT config
    GENOME_PATH = "neatmodel/best_genome.pkl"
    CONFIG_PATH = "neatmodel/config.txt"

    # Load network
    net = load_genome_and_config(GENOME_PATH, CONFIG_PATH)

    # Test the model
    test_env(net, episodes=10, max_steps=30000)
