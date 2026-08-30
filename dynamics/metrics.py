import numpy as np


def metric2objective(metric, objective):
    if objective == "disturbance":
        return {
            "disturbance_resistance_score": metric["disturbance_resistance_score"],
        }

    elif objective == "contact":
        return {
            "num_contacts": metric["num_contacts"],
        }

    elif objective == "angular_span":
        return {
            "angular_span": metric["angular_span"],
        }

    elif objective == "disturbance_contact":
        return {
            "disturbance_resistance_score": metric["disturbance_resistance_score"],
            "num_contacts": metric["num_contacts"],
            "combined_score": (
                metric["disturbance_resistance_score"]
                + 0.1 * metric["num_contacts"]
            ),
        }

    elif objective == "disturbance_span":
        return {
            "disturbance_resistance_score": metric["disturbance_resistance_score"],
            "angular_span": metric["angular_span"],
            "combined_score": (
                metric["disturbance_resistance_score"]
                + 0.5 * metric["angular_span"]
            ),
        }

    elif objective == "disturbance_contact_span":
        return {
            "disturbance_resistance_score": metric["disturbance_resistance_score"],
            "num_contacts": metric["num_contacts"],
            "angular_span": metric["angular_span"],
            "combined_score": (
                metric["disturbance_resistance_score"]
                + 0.1 * metric["num_contacts"]
                + 0.5 * metric["angular_span"]
            ),
        }
    else:
        raise ValueError(f"objective not supported: {objective}")
