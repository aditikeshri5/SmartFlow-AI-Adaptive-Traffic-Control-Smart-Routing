
import numpy as np


class FuzzyTrafficController:
    """
    Mamdani Fuzzy Logic Controller for adaptive traffic signals.

    Inputs:
        density: Traffic occupancy/congestion score (0-100%)
        waiting_time: Queue waiting time (0-120 seconds)

    Output:
        Green signal duration (10-60 seconds)
    """

    def __init__(self):
        self.density_range = np.arange(0, 101, 1)
        self.waiting_range = np.arange(0, 121, 1)
        self.green_range = np.arange(10, 61, 1)

    @staticmethod
    def trimf(x, points):
        """Triangular membership function."""
        a, b, c = points
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)

        if a == b:
            result = np.where(x == a, 1.0, result)
        else:
            result = np.maximum(
                result, np.minimum((x - a) / (b - a), 1.0)
            )

        if b == c:
            result = np.where(x == c, 1.0, result)
        else:
            result = np.minimum(
                result, np.maximum((c - x) / (c - b), 0.0)
            )

        return np.clip(result, 0.0, 1.0)

    def density_membership(self, value):
        return {
            "Low": self.trimf(value, [0, 0, 50]),
            "Medium": self.trimf(value, [20, 50, 80]),
            "High": self.trimf(value, [50, 100, 100]),
        }

    def waiting_membership(self, value):
        return {
            "Short": self.trimf(value, [0, 0, 60]),
            "Medium": self.trimf(value, [20, 60, 100]),
            "Long": self.trimf(value, [60, 120, 120]),
        }

    def green_membership(self):
        return {
            "Short": self.trimf(self.green_range, [10, 10, 30]),
            "Medium": self.trimf(self.green_range, [20, 35, 50]),
            "Long": self.trimf(self.green_range, [40, 60, 60]),
        }

    def calculate_green_time(self, density, waiting_time):
        """Apply fuzzy rules and centroid defuzzification."""

        density = float(np.clip(density, 0, 100))
        waiting_time = float(np.clip(waiting_time, 0, 120))

        d = self.density_membership(density)
        w = self.waiting_membership(waiting_time)
        outputs = self.green_membership()

        # Mamdani rule base:
        #             Waiting time
        # Density     Short    Medium    Long
        # Low         Short    Short     Medium
        # Medium      Short    Medium    Long
        # High        Medium   Long      Long

        rules = [
            ("Low", "Short", "Short"),
            ("Low", "Medium", "Short"),
            ("Low", "Long", "Medium"),
            ("Medium", "Short", "Short"),
            ("Medium", "Medium", "Medium"),
            ("Medium", "Long", "Long"),
            ("High", "Short", "Medium"),
            ("High", "Medium", "Long"),
            ("High", "Long", "Long"),
        ]

        aggregated = np.zeros_like(self.green_range, dtype=float)
        fired_rules = []

        for density_label, waiting_label, output_label in rules:
            strength = min(
                d[density_label],
                w[waiting_label]
            )

            if strength > 0:
                clipped_output = np.minimum(
                    strength, outputs[output_label]
                )
                aggregated = np.maximum(
                    aggregated, clipped_output
                )

                fired_rules.append({
                    "density": density_label,
                    "waiting": waiting_label,
                    "green": output_label,
                    "strength": round(float(strength), 3),
                })

        # Centroid defuzzification
        area = np.sum(aggregated)

        if area == 0:
            green_time = 30.0
        else:
            green_time = float(
                np.sum(self.green_range * aggregated) / area
            )

        return {
            "green_time": round(green_time, 2),
            "density_membership": d,
            "waiting_membership": w,
            "aggregated_output": aggregated,
            "green_range": self.green_range,
            "fired_rules": fired_rules,
        }
