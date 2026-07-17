from pathlib import Path
import json

from musclemimic.runner.checkpointing import _canonicalize_resume_path


def test_huggingface_style_checkpoint_uses_resolved_symlink_target(tmp_path: Path):
    checkpoint = tmp_path / "named-checkpoint"
    (checkpoint / "train_state").mkdir(parents=True)
    (checkpoint / "metadata").mkdir()
    (checkpoint / "metadata" / "metadata").write_text(json.dumps({"update_number": 42}))

    resolved = Path(_canonicalize_resume_path(str(checkpoint)))

    assert resolved.name == "checkpoint_42"
    assert resolved.parent.name == ".orbax_compat_named-checkpoint"
    assert resolved.is_dir()
    assert not resolved.is_symlink()
    assert (resolved / "metadata" / "metadata").read_text() == (checkpoint / "metadata" / "metadata").read_text()
