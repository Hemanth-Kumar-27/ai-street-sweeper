"""
=========================================
Fuzzy Logic Controller
=========================================
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class Controller:

    def __init__(self):

        # -----------------------------
        # Inputs
        # -----------------------------

        self.debris = ctrl.Antecedent(
            np.arange(0, 4, 1),
            "debris"
        )

        # -----------------------------
        # Outputs
        # -----------------------------

        self.brush = ctrl.Consequent(
            np.arange(30, 201, 1),
            "brush"
        )

        self.fan = ctrl.Consequent(
            np.arange(1000, 2501, 1),
            "fan"
        )

        # -----------------------------
        # Membership Functions
        # (Gaussian, not triangular -- chosen so membership degree
        # changes smoothly across debris levels instead of the sharp
        # piecewise-linear edges of a triangular MF, which reduces
        # abrupt swings in the defuzzified brush/fan target before the
        # step-limiter in _step_toward_target even sees it)
        # -----------------------------

        self.debris["clean"] = fuzz.gaussmf(
            self.debris.universe,
            0, 0.3
        )

        self.debris["low"] = fuzz.gaussmf(
            self.debris.universe,
            1, 0.3
        )

        self.debris["medium"] = fuzz.gaussmf(
            self.debris.universe,
            2, 0.3
        )

        self.debris["high"] = fuzz.gaussmf(
            self.debris.universe,
            3, 0.3
        )

        # -----------------------------

        self.brush["clean"] = fuzz.gaussmf(
            self.brush.universe,
            30, 6
        )

        self.brush["low"] = fuzz.gaussmf(
            self.brush.universe,
            70, 6
        )

        self.brush["medium"] = fuzz.gaussmf(
            self.brush.universe,
            120, 6
        )

        self.brush["high"] = fuzz.gaussmf(
            self.brush.universe,
            200, 6
        )

        # -----------------------------

        self.fan["clean"] = fuzz.gaussmf(
            self.fan.universe,
            1000, 40
        )

        self.fan["low"] = fuzz.gaussmf(
            self.fan.universe,
            1300, 40
        )

        self.fan["medium"] = fuzz.gaussmf(
            self.fan.universe,
            1800, 40
        )

        self.fan["high"] = fuzz.gaussmf(
            self.fan.universe,
            2500, 40
        )

        # -----------------------------
        # Rules
        # -----------------------------

        rules = [

            ctrl.Rule(
                self.debris["clean"],
                (self.brush["clean"], self.fan["clean"])
            ),

            ctrl.Rule(
                self.debris["low"],
                (self.brush["low"], self.fan["low"])
            ),

            ctrl.Rule(
                self.debris["medium"],
                (self.brush["medium"], self.fan["medium"])
            ),

            ctrl.Rule(
                self.debris["high"],
                (self.brush["high"], self.fan["high"])
            )

        ]

        system = ctrl.ControlSystem(rules)

        self.simulation = ctrl.ControlSystemSimulation(
            system
        )

        self.last_brush_rpm = None
        self.last_fan_rpm = None

    # -----------------------------------------

    def _quantize_output(
        self,
        value,
        step,
        minimum,
        maximum
    ):

        quantized_value = round(value / step) * step
        return int(np.clip(quantized_value, minimum, maximum))

    # -----------------------------------------

    def _step_toward_target(
        self,
        current_value,
        target_value,
        step,
        minimum,
        maximum
    ):

        if current_value is None:
            return int(np.clip(target_value, minimum, maximum))

        if target_value == current_value:
            return current_value

        delta = target_value - current_value

        if abs(delta) <= step:
            return int(np.clip(target_value, minimum, maximum))

        next_value = current_value + (step if delta > 0 else -step)
        return int(np.clip(next_value, minimum, maximum))

    # -----------------------------------------

    def get_settings(
        self,
        prediction,
        coverage = None
    ):

        mapping = {

            "clean": 0,
            "low": 1,
            "medium": 2,
            "high": 3

        }

        self.simulation.input["debris"] = mapping[prediction]

        # self.simulation.input["coverage"] = coverage

        self.simulation.compute()

        target_brush = self._quantize_output(
            self.simulation.output["brush"],
            25,
            30,
            200
        )

        target_fan = self._quantize_output(
            self.simulation.output["fan"],
            50,
            1000,
            2500
        )

        brush_rpm = self._step_toward_target(
            self.last_brush_rpm,
            target_brush,
            25,
            30,
            200
        )

        fan_rpm = self._step_toward_target(
            self.last_fan_rpm,
            target_fan,
            50,
            1000,
            2500
        )

        self.last_brush_rpm = brush_rpm
        self.last_fan_rpm = fan_rpm

        return {
            "brush": brush_rpm,
            "fan": fan_rpm
        }