from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda import Panda


@register_agent()
class PandaWristCamIROM(Panda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_wristcam_irom"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_irom.urdf"


@register_agent()
class PandaWristCamPeg(Panda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_wristcam_peg"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"
