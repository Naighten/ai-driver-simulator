"""Distributed SAC training script (Ape-X style via Queue).

Architecture:
  Workers (N processes)  ──push transitions──>  Queue  ──read──>  Learner
  Workers sync actor weights from Learner via Manager.dict

Usage:
    python -m agents.rl.train_sac --mode train
    python -m agents.rl.train_sac --mode resume --checkpoint <path>
"""

import argparse
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import yaml

from agents.rl.sac_agent import SACAgent
from agents.rl.shared_buffer import SharedReplayBuffer
from environment.torcs_env import TorcsEnv
from utils.logger import Logger
from utils.checkpoint import (
    save_sac_checkpoint,
    load_sac_checkpoint,
    find_latest_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_dir: str = "config") -> dict:
    cfg = {}
    for fname in ["base.yaml", "reward.yaml", "sac.yaml"]:
        fpath = os.path.join(config_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = yaml.safe_load(f)
                if data:
                    cfg.update(data)
    return cfg


def worker_process(
    worker_id: int,
    port: int,
    config: dict,
    transition_queue: mp.Queue,
    stop_event: mp.Event,
    actor_state_shared: dict,
) -> None:
    """Worker: collect experience, push transitions to queue, sync actor weights.

    Args:
        worker_id: worker index.
        port: TORCS UDP port.
        config: merged config dict.
        transition_queue: queue for sending (obs, action, reward, next_obs, done) tuples.
        stop_event: event to signal termination.
        actor_state_shared: multiprocessing manager dict with actor weights.
    """
    logger.info(f"Worker {worker_id} started on port {port}")

    sac_kwargs = config.get("sac", {})
    worker_agent = SACAgent(
        obs_dim=32,
        action_dim=3,
        hidden_dim=sac_kwargs.get("hidden_dim", 256),
        n_hidden=sac_kwargs.get("n_hidden", 2),
        lr=sac_kwargs.get("learning_rate", 3e-4),
        gamma=sac_kwargs.get("gamma", 0.99),
        tau=sac_kwargs.get("tau", 0.005),
        alpha=sac_kwargs.get("alpha", "auto"),
        policy_freq=sac_kwargs.get("policy_freq", 2),
        device="cpu",
    )

    torcs_cfg = config.get("torcs", {})
    reward_weights = config.get("reward", {})

    env = TorcsEnv(
        host="localhost",
        port=port,
        timeout=torcs_cfg.get("timeout", 1.0),
        max_steps=torcs_cfg.get("max_steps", 50000),
        grace_steps=torcs_cfg.get("grace_steps", 100),
        offtrack_limit=torcs_cfg.get("offtrack_limit", 1.5),
        offtrack_patience=torcs_cfg.get("offtrack_patience", 5),
        stuck_speed_threshold=torcs_cfg.get("stuck_speed_threshold", 2.0),
        stuck_steps=torcs_cfg.get("stuck_steps", 100),
        smoothing_alpha=config.get("action", {}).get("smoothing_alpha", 0.6),
        action_repeat=config.get("action", {}).get("repeat", 4),
        min_accel=config.get("action", {}).get("min_accel", 0.2),
        reward_weights=reward_weights,
    )

    sync_interval = sac_kwargs.get("sync_actor_every", 200)
    step = 0
    episode_reward = 0.0
    episode_step = 0

    try:
        obs, info = env.reset()
        while not stop_event.is_set():
            # Sync actor weights from manager dict
            if step > 0 and step % sync_interval == 0:
                if "actor_state" in actor_state_shared:
                    worker_agent.set_actor_weights(actor_state_shared["actor_state"])

            # Select action
            obs_tensor = torch.from_numpy(obs).float()
            action = worker_agent.actor.get_action(obs_tensor, deterministic=False)
            action[1] = max(0.0, min(1.0, action[1]))
            action[2] = max(0.0, min(1.0, action[2]))

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_step += 1

            # Push to queue (picklable numpy arrays)
            transition_queue.put((obs, action, reward, next_obs, done))

            if done:
                logger.info(
                    f"Worker {worker_id}: episode reward={episode_reward:.1f}, "
                    f"steps={episode_step}"
                )
                episode_reward = 0.0
                episode_step = 0
                obs, info = env.reset()
            else:
                obs = next_obs

            step += 1

    except KeyboardInterrupt:
        logger.info(f"Worker {worker_id} interrupted")
    finally:
        env.close()
        logger.info(f"Worker {worker_id} stopped after {step} steps")


def train_sac(config: dict, checkpoint_path: Optional[str] = None) -> None:
    """Main SAC training loop.

    Learner runs in main process, reads transitions from queue,
    stores in local buffer, and performs gradient updates.
    Workers are spawned child processes.

    Args:
        config: merged configuration dict.
        checkpoint_path: optional path to resume from checkpoint.
    """
    sac_kwargs = config.get("sac", {})
    torcs_cfg = config.get("torcs", {})
    ports = torcs_cfg.get("ports", [3001])
    log_dir = config.get("logging", {}).get("log_dir", "runs/sac")
    checkpoint_dir = config.get("checkpoint", {}).get("dir", "checkpoints")
    learning_starts = sac_kwargs.get("learning_starts", 10_000)
    max_steps = 1_000_000
    buffer_capacity = sac_kwargs.get("buffer_size", 1_000_000)

    # Learner local buffer (not shared — filled from queue)
    buffer = SharedReplayBuffer(
        obs_dim=32,
        action_dim=3,
        capacity=buffer_capacity,
    )

    # Learner SAC agent
    learner = SACAgent(
        obs_dim=32,
        action_dim=3,
        hidden_dim=sac_kwargs.get("hidden_dim", 256),
        n_hidden=sac_kwargs.get("n_hidden", 2),
        lr=sac_kwargs.get("learning_rate", 3e-4),
        gamma=sac_kwargs.get("gamma", 0.99),
        tau=sac_kwargs.get("tau", 0.005),
        alpha=sac_kwargs.get("alpha", "auto"),
        policy_freq=sac_kwargs.get("policy_freq", 2),
        batch_size=sac_kwargs.get("batch_size", 256),
        buffer_capacity=buffer_capacity,
    )
    learner.set_shared_buffer(buffer)

    # Resume from checkpoint
    start_step = 0
    if checkpoint_path:
        data = load_sac_checkpoint(checkpoint_path)
        learner.load_checkpoint_data(data)
        start_step = data["step"]
        logger.info(f"Resumed SAC from checkpoint at step {start_step}")

    # Logger
    logger_writer = Logger(log_dir=log_dir, tag="sac")

    # IPC primitives
    mp_ctx = mp.get_context("spawn")
    manager = mp_ctx.Manager()
    transition_queue = mp_ctx.Queue(maxsize=10000)
    stop_event = mp_ctx.Event()
    actor_state_shared = manager.dict()

    # Start worker processes
    workers = []
    for i, port in enumerate(ports):
        p = mp_ctx.Process(
            target=worker_process,
            args=(i, port, config, transition_queue, stop_event, actor_state_shared),
        )
        p.start()
        workers.append(p)
        logger.info(f"Started worker {i} on port {port}")

    # Signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Training loop: read from queue, fill buffer, update learner
    step = start_step
    try:
        while step < max_steps and not stop_event.is_set():
            # Drain queue into buffer (non-blocking)
            drained = 0
            while not transition_queue.empty() and drained < 100:
                try:
                    obs, action, reward, next_obs, done = transition_queue.get_nowait()
                    buffer.push(obs, action, reward, next_obs, done)
                    drained += 1
                except Exception:
                    break

            if drained > 0 and buffer.ready(learning_starts):
                for _ in range(min(drained, 10)):
                    metrics = learner.update()
                    step += 1

                # Publish actor weights for workers
                if step % sac_kwargs.get("sync_actor_every", 200) == 0:
                    actor_state_shared["actor_state"] = learner.get_actor_state()

                # Logging
                if step % 100 == 0 and step > start_step:
                    logger_writer.log_scalars(step, metrics, prefix="sac")
                    logger.info(
                        f"Step {step}: critic_loss={metrics['critic_loss']:.4f}, "
                        f"actor_loss={metrics.get('actor_loss', 0):.4f}, "
                        f"alpha={metrics['alpha']:.4f}, "
                        f"buffer_size={len(buffer)}"
                    )

                # Checkpoint
                if step % 5000 == 0 and step > start_step:
                    ckpt_data = learner.get_checkpoint_data()
                    filepath = save_sac_checkpoint(
                        checkpoint_dir, **ckpt_data, metadata={"config": config}
                    )
                    logger.info(f"Saved checkpoint: {filepath}")
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

        # Final checkpoint
        if step > 0:
            ckpt_data = learner.get_checkpoint_data()
            filepath = save_sac_checkpoint(
                checkpoint_dir,
                **ckpt_data,
                metadata={"config": config, "final": True},
            )
            logger.info(f"Final checkpoint saved: {filepath}")
        logger_writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train SAC agent for TORCS")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "resume"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = load_config()

    if args.mode == "resume":
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = find_latest_checkpoint(
                config.get("checkpoint", {}).get("dir", "checkpoints")
            )
        if checkpoint_path is None:
            logger.error("No checkpoint found for resume mode")
            sys.exit(1)
        train_sac(config, checkpoint_path)
    else:
        train_sac(config)


if __name__ == "__main__":
    main()
