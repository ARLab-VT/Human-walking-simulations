"""Export reproducible tables from the exact compiled MuJoCo model."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from musclemimic.research.reflex_recovery.muscle_groups import build_lower_body_groups, export_group_map


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or f"unnamed_{index}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_model(model: mujoco.MjModel, output_dir: str | Path) -> dict[str, object]:
    """Export model tables and return a small JSON-serializable summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bodies = [
        {"body_id": index, "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, index), "parent_id": int(model.body_parentid[index])}
        for index in range(model.nbody)
    ]
    joints = []
    for index in range(model.njnt):
        joints.append(
            {
                "joint_id": index,
                "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, index),
                "type": int(model.jnt_type[index]),
                "qpos_address": int(model.jnt_qposadr[index]),
                "dof_address": int(model.jnt_dofadr[index]),
                "limited": bool(model.jnt_limited[index]),
                "range_min_rad": float(model.jnt_range[index, 0]),
                "range_max_rad": float(model.jnt_range[index, 1]),
                "stiffness_nm_rad": float(model.jnt_stiffness[index]),
                "damping_nms_rad": float(model.dof_damping[model.jnt_dofadr[index]]),
                "armature_kgm2": float(model.dof_armature[model.jnt_dofadr[index]]),
            }
        )
    actuators = []
    for index in range(model.nu):
        actuators.append(
            {
                "actuator_id": index,
                "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index),
                "transmission_type": int(model.actuator_trntype[index]),
                "transmission_id": int(model.actuator_trnid[index, 0]),
                "control_limited": bool(model.actuator_ctrllimited[index]),
                "control_min": float(model.actuator_ctrlrange[index, 0]),
                "control_max": float(model.actuator_ctrlrange[index, 1]),
                "force_limited": bool(model.actuator_forcelimited[index]),
                "force_min_n": float(model.actuator_forcerange[index, 0]),
                "force_max_n": float(model.actuator_forcerange[index, 1]),
                "dynamics_type": int(model.actuator_dyntype[index]),
            }
        )
    sites = [
        {"site_id": index, "name": _name(model, mujoco.mjtObj.mjOBJ_SITE, index), "body_id": int(model.site_bodyid[index])}
        for index in range(model.nsite)
    ]
    sensors = [
        {
            "sensor_id": index,
            "name": _name(model, mujoco.mjtObj.mjOBJ_SENSOR, index),
            "type": int(model.sensor_type[index]),
            "object_type": int(model.sensor_objtype[index]),
            "object_id": int(model.sensor_objid[index]),
            "dimension": int(model.sensor_dim[index]),
        }
        for index in range(model.nsensor)
    ]
    geoms = [
        {
            "geom_id": index,
            "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, index),
            "body_id": int(model.geom_bodyid[index]),
            "contact_type": int(model.geom_contype[index]),
            "contact_affinity": int(model.geom_conaffinity[index]),
        }
        for index in range(model.ngeom)
    ]
    for filename, rows in (("bodies.csv", bodies), ("joints.csv", joints), ("actuators.csv", actuators), ("sites.csv", sites), ("sensors.csv", sensors), ("contacts.csv", geoms)):
        _write_csv(output / filename, rows)

    actuator_name_tuple = tuple(str(row["name"]) for row in actuators)
    groups = (
        build_lower_body_groups(actuator_name_tuple)
        if {"iliacus_r", "iliacus_l"}.issubset(actuator_name_tuple)
        else ()
    )
    if groups:
        export_group_map(groups, output)

    toe_bodies = [row for row in bodies if "toe" in str(row["name"]).lower()]
    toe_joints = [row for row in joints if any(token in str(row["name"]).lower() for token in ("toe", "mtp"))]
    toe_geoms = [row for row in geoms if any(token in str(row["name"]).lower() for token in ("toe", "foot", "heel"))]
    toe_report = [
        "# Toe/MTP Audit",
        "",
        f"- Separate toe-named bodies: {len(toe_bodies)}",
        f"- Toe/MTP-named joints: {len(toe_joints)}",
        f"- Heel/foot/toe contact geometries: {len(toe_geoms)}",
        "",
        "The CSV tables are authoritative. Active/passive status and center-of-pressure progression require inspection of these compiled-model rows and a contact rollout.",
    ]
    (output / "toe_audit.md").write_text("\n".join(toe_report) + "\n", encoding="utf-8")
    (output / "action_path.md").write_text(
        "# Action Path\n\nThe policy emits normalized actions. Residuals are composed and clipped in normalized action space before `DefaultControl.generate_action`, which maps them to compiled actuator control ranges and writes them to `data.ctrl`. External joint torque disturbances remain separate in `data.qfrc_applied`.\n",
        encoding="utf-8",
    )
    summary = {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "njnt": model.njnt,
        "nsite": model.nsite,
        "nsensor": model.nsensor,
        "timestep_s": float(model.opt.timestep),
        "toe_body_count": len(toe_bodies),
        "toe_joint_count": len(toe_joints),
        "functional_muscle_group_count": len(groups),
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    (output / "summary.json").write_text(payload + "\n", encoding="utf-8")
    summary["audit_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    return summary
