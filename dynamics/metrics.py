import numpy as np


def metric2objective(metric, objective):
    if objective == "disturbance":
        return {
            "disturbance_resistance_score":
                metric["disturbance_resistance_score"],
        }
    elif objective == "contact":
        return {
            "num_contacts":
                metric["num_contacts"],
        }
    elif objective == "disturbance_contact":
        return {
            "disturbance_resistance_score":
                metric["disturbance_resistance_score"],

            "num_contacts":
                metric["num_contacts"],

            "combined_score":
                metric["disturbance_resistance_score"]
                + 0.1 * metric["num_contacts"],
        }
    else:
        raise ValueError("objective not supported")