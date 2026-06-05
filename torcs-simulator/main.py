"""Main entry point for the TORCS agents training framework.

Simplifies launching training or evaluation for both SAC and NEAT agents.

Usage:
    python main.py --agent sac --mode train
    python main.py --agent neat --mode train
    python main.py --agent sac --mode resume --checkpoint <path>
    python main.py --eval --agent sac --checkpoint <path>
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="TORCS RL & NEAT Training Framework")
    parser.add_argument(
        "--agent", type=str, choices=["sac", "neat"], required=True,
        help="Agent type to train or evaluate"
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "resume"],
        help="Training mode (train from scratch or resume from checkpoint)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint for resume or eval mode"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation instead of training"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of evaluation episodes (default: 3)"
    )

    args = parser.parse_args()

    if args.eval:
        from eval import main as eval_main
        # Forward arguments to eval.py
        eval_sys_argv = [
            "eval.py",
            "--agent", args.agent,
            "--checkpoint", args.checkpoint or "",
            "--episodes", str(args.episodes),
        ]
        sys.argv = eval_sys_argv
        eval_main()
    else:
        if args.agent == "sac":
            from agents.rl.train_sac import main as sac_main
            sac_sys_argv = ["train_sac.py", "--mode", args.mode or "train"]
            if args.checkpoint:
                sac_sys_argv.extend(["--checkpoint", args.checkpoint])
            sys.argv = sac_sys_argv
            sac_main()
        elif args.agent == "neat":
            from agents.neat.train_neat import main as neat_main
            neat_sys_argv = ["train_neat.py", "--mode", args.mode or "train"]
            if args.checkpoint:
                neat_sys_argv.extend(["--checkpoint", args.checkpoint])
            sys.argv = neat_sys_argv
            neat_main()


if __name__ == "__main__":
    main()
