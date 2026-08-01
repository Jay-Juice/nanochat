import json

import pytest

from scripts.chat_rl_diagnostics import (
    compare_result_files,
    make_sampling_seed,
    write_result,
)


def test_sampling_seed_is_reproducible_and_unique_across_passes():
    seeds = [
        make_sampling_seed(1337, question_idx=9, sampling_pass=i, num_sampling_passes=2)
        for i in range(2)
    ]

    assert seeds == [1355, 1356]
    assert len(set(seeds)) == 2


def test_result_writer_refuses_to_overwrite_by_default(tmp_path):
    output_path = tmp_path / "diagnostics.json"
    write_result(output_path, {"first": True})

    with pytest.raises(FileExistsError, match="already exists"):
        write_result(output_path, {"second": True})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"first": True}


def test_result_file_comparison_uses_saved_records_and_labels(tmp_path):
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps({
        "label": "d24-sft",
        "records": [
            {"idx": 0, "rewards": [1, 0]},
            {"idx": 1, "rewards": [0, 0]},
        ],
    }), encoding="utf-8")
    right_path.write_text(json.dumps({
        "label": "d32-sft",
        "records": [
            {"idx": 0, "rewards": [0, 1]},
            {"idx": 1, "rewards": [1, 0]},
        ],
    }), encoding="utf-8")

    result = compare_result_files(left_path, right_path)

    assert result["left_label"] == "d24-sft"
    assert result["right_label"] == "d32-sft"
    assert result["comparison"]["intersection"] == 1
    assert result["comparison"]["right_only"] == 1
