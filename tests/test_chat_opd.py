import subprocess
import sys
from pathlib import Path


def test_chat_opd_help_lists_explicit_student_and_teacher_checkpoints():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.chat_opd", "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.returncode == 0, result.stderr
    assert "--student-model-tag" in result.stdout
    assert "--student-model-step" in result.stdout
    assert "--teacher-model-tag" in result.stdout
    assert "--teacher-model-step" in result.stdout
    assert "--output-model-tag" in result.stdout


def test_chat_opd_help_lists_gsm8k_control_v2_defaults():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.chat_opd", "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.returncode == 0, result.stderr
    assert "--lr-schedule" in result.stdout
    assert "--eval-seed" in result.stdout
    assert result.stdout.count("(default: 64)") >= 2
    assert "(default: constant)" in result.stdout
    assert "(default: 1337)" in result.stdout
    assert result.stdout.count("(default: 20)") >= 2
