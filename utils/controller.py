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

        self.coverage = ctrl.Antecedent(
            np.arange(0, 101, 1),
            "coverage"
        )

        # -----------------------------
        # Outputs
        # -----------------------------

        self.brush = ctrl.Consequent(
            np.arange(100, 341, 1),
            "brush"
        )

        self.fan = ctrl.Consequent(
            np.arange(1200, 3001, 1),
            "fan"
        )

        # -----------------------------
        # Membership Functions
        # -----------------------------

        self.debris["clean"] = fuzz.trimf(
            self.debris.universe,
            [0, 0, 1]
        )

        self.debris["low"] = fuzz.trimf(
            self.debris.universe,
            [0, 1, 2]
        )

        self.debris["medium"] = fuzz.trimf(
            self.debris.universe,
            [1, 2, 3]
        )

        self.debris["high"] = fuzz.trimf(
            self.debris.universe,
            [2, 3, 3]
        )

        # -----------------------------

        self.coverage["low"] = fuzz.trimf(
            self.coverage.universe,
            [0, 0, 40]
        )

        self.coverage["medium"] = fuzz.trimf(
            self.coverage.universe,
            [20, 50, 80]
        )

        self.coverage["high"] = fuzz.trimf(
            self.coverage.universe,
            [60, 100, 100]
        )

        # -----------------------------

        self.brush["slow"] = fuzz.trimf(
            self.brush.universe,
            [100, 140, 180]
        )

        self.brush["medium"] = fuzz.trimf(
            self.brush.universe,
            [170, 240, 290]
        )

        self.brush["fast"] = fuzz.trimf(
            self.brush.universe,
            [270, 320, 340]
        )

        # -----------------------------

        self.fan["low"] = fuzz.trimf(
            self.fan.universe,
            [1200, 1500, 1800]
        )

        self.fan["medium"] = fuzz.trimf(
            self.fan.universe,
            [1700, 2200, 2600]
        )

        self.fan["high"] = fuzz.trimf(
            self.fan.universe,
            [2500, 2800, 3000]
        )

        # -----------------------------
        # Rules
        # -----------------------------

        rules = [

            ctrl.Rule(
                self.debris["clean"],
                (self.brush["slow"], self.fan["low"])
            ),

            ctrl.Rule(
                self.debris["low"],
                (self.brush["medium"], self.fan["medium"])
            ),

            ctrl.Rule(
                self.debris["medium"],
                (self.brush["fast"], self.fan["high"])
            ),

            ctrl.Rule(
                self.debris["high"],
                (self.brush["fast"], self.fan["high"])
            )

        ]

        system = ctrl.ControlSystem(rules)

        self.simulation = ctrl.ControlSystemSimulation(
            system
        )

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

        return {

            "brush": int(
                self.simulation.output["brush"]
            ),

            "fan": int(
                self.simulation.output["fan"]
            )

        }