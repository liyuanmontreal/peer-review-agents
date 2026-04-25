"""Tests for reva.foreground — foreground (in-terminal) agent runner."""
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from reva.foreground import run_foreground


def _make_agent_dir(tmp_path):
    agent_dir = tmp_path / "agents" / "foo"
    agent_dir.mkdir(parents=True)
    (agent_dir / ".api_key").write_text("test-key\n")
    return agent_dir


_SIMPLE_SCRIPT = "#!/usr/bin/env bash\necho hello\n"


def test_run_foreground_calls_bash_with_launch_script(tmp_path):
    agent_dir = _make_agent_dir(tmp_path)
    with patch("reva.foreground.shutil.which", return_value="/usr/bin/bash"), \
         patch("reva.foreground.subprocess.run") as mock_run:
        run_foreground("foo", str(agent_dir), _SIMPLE_SCRIPT)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/bin/bash"
    assert cmd[1].endswith(".reva_launch.sh")


def test_run_foreground_raises_when_bash_not_found(tmp_path):
    agent_dir = _make_agent_dir(tmp_path)
    with patch("reva.foreground.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="bash not found"):
            run_foreground("foo", str(agent_dir), _SIMPLE_SCRIPT)


def test_run_foreground_sets_cwd_to_working_dir(tmp_path):
    agent_dir = _make_agent_dir(tmp_path)
    with patch("reva.foreground.shutil.which", return_value="/usr/bin/bash"), \
         patch("reva.foreground.subprocess.run") as mock_run:
        run_foreground("foo", str(agent_dir), _SIMPLE_SCRIPT)

    kwargs = mock_run.call_args[1]
    assert kwargs["cwd"] == str(agent_dir.resolve())


def test_run_foreground_does_not_capture_output(tmp_path):
    """Foreground mode must let stdin/stdout/stderr flow to the terminal."""
    agent_dir = _make_agent_dir(tmp_path)
    with patch("reva.foreground.shutil.which", return_value="/usr/bin/bash"), \
         patch("reva.foreground.subprocess.run") as mock_run:
        run_foreground("foo", str(agent_dir), _SIMPLE_SCRIPT)

    kwargs = mock_run.call_args[1]
    assert not kwargs.get("capture_output", False)
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs


def test_run_foreground_writes_launch_files(tmp_path):
    """run_foreground writes .reva_launch.sh, same as create_session."""
    agent_dir = _make_agent_dir(tmp_path)
    with patch("reva.foreground.shutil.which", return_value="/usr/bin/bash"), \
         patch("reva.foreground.subprocess.run"):
        run_foreground("foo", str(agent_dir), _SIMPLE_SCRIPT)

    assert (agent_dir / ".reva_launch.sh").exists()


def test_run_foreground_uses_resolved_absolute_path(tmp_path):
    """Script path passed to bash is absolute, not relative."""
    agent_dir = _make_agent_dir(tmp_path)
    with patch("reva.foreground.shutil.which", return_value="/usr/bin/bash"), \
         patch("reva.foreground.subprocess.run") as mock_run:
        run_foreground("foo", str(agent_dir), _SIMPLE_SCRIPT)

    cmd = mock_run.call_args[0][0]
    script_path = Path(cmd[1])
    assert script_path.is_absolute()


def test_run_foreground_error_message_mentions_git(tmp_path):
    """Error for missing bash should guide Windows users."""
    agent_dir = _make_agent_dir(tmp_path)
    with patch("reva.foreground.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="Git"):
            run_foreground("foo", str(agent_dir), _SIMPLE_SCRIPT)
