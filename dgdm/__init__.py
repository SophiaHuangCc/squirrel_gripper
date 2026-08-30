"""Faithful DGDM-style design generation for the Squirrel gripper."""

from .data import PROFILE_CHANNELS, InteractionProfileDataset, UnconditionalDesignDataset
from .guidance import ProfileTarget, ScenarioBatch
from .models import InteractionProfileModel

__all__ = [
    "InteractionProfileDataset",
    "InteractionProfileModel",
    "PROFILE_CHANNELS",
    "ProfileTarget",
    "ScenarioBatch",
    "UnconditionalDesignDataset",
]
