"""RPM-based heuristic for automatic gear shifting.

Provides a gear recommendation based on current RPM and speed.
This is used by the action builder and is not part of the action space.
"""


# Upshift and downshift RPM thresholds for each gear.
# Index 0 is for neutral (gear=0), gear ranges from 1 to 6.
_UP_SHIFT_RPM = [0, 7000, 7000, 7000, 7000, 7000, 7000]
_DOWN_SHIFT_RPM = [0, 3500, 3500, 4000, 4000, 4500, 4500]


def recommend_gear(rpm: float, current_gear: int, speed_x: float) -> int:
    """Recommend a gear based on RPM and speed.

    Uses a simple RPM-threshold heuristic:
      - Upshift when RPM exceeds the upshift threshold for current gear.
      - Downshift when RPM falls below the downshift threshold.
      - Never shift above gear 6 or below gear 1.
      - At very low speed (< 5 km/h), force gear 1.

    Args:
        rpm: current engine RPM.
        current_gear: current gear (1-6).
        speed_x: longitudinal speed in km/h.

    Returns:
        int: recommended gear (1-6).
    """
    if speed_x < 5.0:
        return 1

    new_gear = current_gear

    if rpm > _UP_SHIFT_RPM[new_gear] and new_gear < 6:
        new_gear += 1
    elif rpm < _DOWN_SHIFT_RPM[new_gear] and new_gear > 1:
        new_gear -= 1

    return new_gear
