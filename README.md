<div align="center">
    <h1>
        <b>AI Driver Simulator</b>
    </h1>
    <h3>
        Обучение агентов RL и NEAT для управления гоночными автомобилями
    </h3>
</div>

Проект состоит из двух симуляторов.

---

## 1. my-simulator — 2D трековый симулятор

Собственная среда на Pygame с 8 треками.
Наблюдения: 12 чисел (позиция, угол, скорость, 8 сенсоров-лучей).
Действия: непрерывные `[steering, throttle]` в диапазоне `[-1, 1]`.

**Обучение:**
| Алгоритм | Команда |
|----------|---------|
| NEAT | `python my-simulator/main.py --algo neat` |
| PPO (stable-baselines3) | `python my-simulator/main.py --algo rl` |
| PPO (последовательные этапы) | `python my-simulator/train.py` |

**Тестирование:**
| Команда | Описание |
|---------|----------|
| `python my-simulator/test_neat.py` | Запуск обученной NEAT модели на треке |
| `python my-simulator/play.py` | Ручное управление (стрелки) |

---

## 2. torcs-simulator — интеграция с TORCS

Обёртка над гоночным симулятором TORCS (`gym.Env`).
Наблюдения: 32 числа (скорость, обороты, датчики дистанции и т.д.).
Действия: `[steering, acceleration, brake]`.

**Обучение:**
| Алгоритм | Команда |
|----------|---------|
| SAC | `python torcs-simulator/main.py --agent sac --mode train` |
| SAC (с чекпоинта) | `python torcs-simulator/main.py --agent sac --mode resume --checkpoint <path>` |
| NEAT | `python torcs-simulator/main.py --agent neat --mode train` |

**Тестирование:**
| Команда | Описание |
|---------|----------|
| `python torcs-simulator/main.py --eval --agent sac --checkpoint <path>` | Оценка SAC модели |
| `python torcs-simulator/main.py --eval --agent neat --checkpoint <path>` | Оценка NEAT модели |
| `python torcs-simulator/eval.py --agent sac --checkpoint <path>` | Запуск eval напрямую |

---

## Зависимости

Python **3.14**, управление через `uv`.

Основные библиотеки: `torch`, `gymnasium`, `numpy`, `neat-python`, `pygame`, `opencv-python`, `stable-baselines3`, `tensorboard`, `pyyaml`.

```bash
uv sync
