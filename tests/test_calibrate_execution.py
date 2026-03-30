from pathlib import Path
import json
import subprocess


def test_calibrate_execution_script(tmp_path: Path):
    state = {
        "fills": [
            {"is_maker": True, "missed": False, "fee": 1.2, "slippage": 2.0},
            {"is_maker": False, "missed": True, "fee": 0.0, "slippage": 0.0},
        ]
    }
    state_path = tmp_path / "paper_state.json"
    state_path.write_text(json.dumps(state))
    out = tmp_path / "calib.json"
    cmd = [
        "python3",
        "scripts/calibrate_execution.py",
        "--paper-state",
        str(state_path),
        "--out-json",
        str(out),
    ]
    cp = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert cp.returncode == 0
    payload = json.loads(out.read_text())
    assert "recommended_execution" in payload
    assert payload["recommended_execution"]["maker_fill_prob"] > 0

