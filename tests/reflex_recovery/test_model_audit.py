from pathlib import Path

import mujoco

from musclemimic.research.reflex_recovery.model_audit import audit_model


def test_model_audit_exports_expected_artifacts(tmp_path: Path):
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody><body name='toe'><joint name='mtp' type='hinge'/><geom name='toe_contact' type='sphere' size='.1'/></body></worldbody></mujoco>"""
    )
    summary = audit_model(model, tmp_path)
    assert summary["toe_body_count"] == 1
    assert summary["toe_joint_count"] == 1
    for filename in ("bodies.csv", "joints.csv", "actuators.csv", "sites.csv", "sensors.csv", "contacts.csv", "toe_audit.md", "action_path.md", "summary.json"):
        assert (tmp_path / filename).exists()
