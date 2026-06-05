import logging
import os
import pickle
from logging import FileHandler

import neat
import numpy as np


class TimeoutException(Exception):
    pass


class ResettingFileHandler(FileHandler):
    def __init__(self, filename, max_lines=10000, mode="a", encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.max_lines = max_lines
        self.line_count = 0
        self.check_file_size()

    def emit(self, record):
        super().emit(record)
        self.line_count += 1
        if self.line_count >= self.max_lines:
            self.reset_log()

    def reset_log(self):
        self.close()
        open(self.baseFilename, "w", encoding=self.encoding).close()
        self.line_count = 0
        self.stream = self._open()

    def check_file_size(self):
        if os.path.exists(self.baseFilename):
            with open(self.baseFilename, "r", encoding=self.encoding) as f:
                self.line_count = sum(1 for _ in f)


class CustomGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.node_counter = max(self.nodes.keys()) if self.nodes else 0

    def mutate_add_node(self, config):
        for _ in range(100):  # 100 попыток найти уникальный ID
            new_id = config.get_new_node_key(self.nodes)
            if new_id not in self.nodes:
                super().mutate_add_node(config)
                return
        logging.warning("Не удалось добавить узел после 100 попыток")


def run_neat(env, checkpoint_path=None):
    # Настройка конфигурации NEAT
    config_path = "neatmodel/config.txt"
    config = neat.Config(
        CustomGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Настройка логирования с кастомным обработчиком
    log_dir = "neatmodel"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "evaluation.log")

    handler = ResettingFileHandler(log_file, max_lines=10000, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

    # Функция оценки геномов
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            try:
                logging.info(f"Evaluating genome {genome_id}")
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                total_reward = 0
                obs, _ = env.reset()

                stagnant_steps = 0
                max_steps = 10000
                for _ in range(max_steps):
                    obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
                    action = np.array(net.activate(obs))
                    obs, reward, done, _, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break

                    if reward < 0.01:
                        stagnant_steps += 1
                    else:
                        stagnant_steps = 0

                    if stagnant_steps > 200:
                        break

                complexity_penalty = (
                        len(genome.nodes) * 0.01 + len(genome.connections) * 0.005
                )

                genome.fitness = (
                                         total_reward / env.track_max_reward
                                 ) - complexity_penalty

                logging.info(
                    f"Genome {genome_id} has {len(genome.nodes)} nodes and {len(genome.connections)} connections"
                )
                logging.info(
                    f"Genome {genome_id} fitness: {genome.fitness:.5f} (reward: {total_reward:.2f} penalty: {complexity_penalty:.5f})"
                )
            except Exception as e:
                logging.exception(f"Error evaluating genome {genome_id}")
                genome.fitness = -100  # наказание

    # Загрузка популяции из чекпоинта или новая
    if checkpoint_path:
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        p = neat.Population(config)

    # Репортеры (в консоль, статистика, чекпоинты)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Префикс для чекпоинтов
    cp_prefix = "neatmodel/checkpoints/neat-checkpoint-"

    # Собственный чекпоинтер
    cp = neat.Checkpointer(
        generation_interval=10, filename_prefix=cp_prefix, time_interval_seconds=300
    )
    p.add_reporter(cp)

    last_cp = checkpoint_path  # из аргумента при ресторте

    # Запуск NEAT
    try:
        winner = p.run(eval_genomes, 20)
    except (KeyboardInterrupt, TimeoutException) as e:
        logging.info(f"Interrupted: {e}")
        # Сохраняем последний чекпоинт вручную
        p.generation += 0  # текущий номер поколения внутри p
        cp.save_checkpoint(config, p.population, p.species_set, p.generation)
        last_cp = f"{cp_prefix}{p.generation}"
        stats = [r for r in p.reporters if isinstance(r, neat.StatisticsReporter)][0]
        best = stats.best_genome()
        with open("neatmodel/best_genome.pkl", "wb") as f:
            pickle.dump(best, f)
        return best, last_cp
    finally:
        logging.shutdown()

    # При нормальном финише тоже сохраняем финальный чекпоинт
    cp.save_checkpoint(config, p.population, p.species, p.generation)
    last_cp = f"{cp_prefix}{p.generation}"

    with open("neatmodel/stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    # Сохранение лучшего генома
    with open("neatmodel/best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    return winner, last_cp
