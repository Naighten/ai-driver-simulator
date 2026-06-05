import argparse

from CarGameEnv import CarGameEnv
from neatmodel.utils import run_neat
from rl.agent import RLAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["neat", "rl"], required=True)
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["newtrack2", "newtrack3", "newtrack5", "newtrack6"],
        help="Список треков для обучения подряд",
    )
    parser.add_argument("--checkpoint", help="Path to NEAT checkpoint")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.algo == "neat":
        ckpt = args.checkpoint
        for _ in range(50):
            for track_name in args.tracks:
                print(f"\n=== Training on track: {track_name} ===")
                env = CarGameEnv(
                    track_name=track_name, render_mode="human" if args.render else None
                )

                winner, ckpt = run_neat(env, ckpt)
                print(
                    f"Finished track {track_name}, best genome saved, next checkpoint = {ckpt}"
                )
    elif args.algo == "rl":
        env = CarGameEnv(render_mode="human" if args.render else None)
        agent = RLAgent(env)
        if args.checkpoint:
            agent.load_neat_checkpoint(args.checkpoint)
        agent.train()


if __name__ == "__main__":
    main()
