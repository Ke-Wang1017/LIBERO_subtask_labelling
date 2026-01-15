from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda

try:
    from robosuite.robots.single_arm import SingleArm
except ImportError:
    from robosuite.robots import FixedBaseRobot as SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
    }
)
