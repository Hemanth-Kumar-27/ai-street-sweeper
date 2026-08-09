"""
=========================================
Power Calculator
=========================================
"""

import config


class PowerCalculator:

    def __init__(self):

        self.max_power = config.MAX_POWER
        self.max_brush_power = config.MAX_BRUSH_POWER
        self.max_fan_power = config.MAX_FAN_POWER
        self.normal_power = config.NORMAL_POWER

    # ------------------------------------------

    def calculate(self, brush_rpm, fan_rpm):

        brush_ratio = brush_rpm / config.MAX_BRUSH_RPM
        fan_ratio = fan_rpm / config.MAX_FAN_RPM

        brush_power = self.max_brush_power * brush_ratio
        fan_power = self.max_fan_power * (fan_ratio ** 3)

        adaptive_power = brush_power + fan_power

        saving = (
            (self.normal_power - adaptive_power)
            / self.normal_power
        ) * 100

        return {
            "power": round(adaptive_power, 2),
            "saving": round(saving, 2),
        }