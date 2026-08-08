"""
=========================================
Power Calculator
=========================================
"""

from AI_Street_Sweeper import config


class PowerCalculator:

    def __init__(self):

        self.max_power = config.MAX_POWER

    # ------------------------------------------

    def calculate(self, brush_rpm, fan_rpm):

        brush_ratio = (
            brush_rpm /
            config.MAX_BRUSH_RPM
        )

        fan_ratio = (
            fan_rpm /
            config.MAX_FAN_RPM
        )

        adaptive_power = (
            (0.30 * brush_ratio) +
            (0.70 * fan_ratio)
        ) * self.max_power

        saving = (
            (self.max_power - adaptive_power)
            / self.max_power
        ) * 100

        return {

            "power": round(adaptive_power, 2),

            "saving": round(saving, 2)

        }