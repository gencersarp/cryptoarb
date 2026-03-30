from pathlib import Path
import json
import subprocess


def test_go_no_go_script(tmp_path: Path):
    payload = {
        "best_eval": {
            "wf_results": [
                {"asset": "BTC", "pass_fail": {"checks": {"a": True, "b": True}}},
                {"asset": "ETH", "pass_fail": {"checks": {"a": True}}},
            ]
        }
    }
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(payload))
    cmd = ["python3", "scripts/run_go_no_go.py", "--input-json", str(p)]
    cp = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert cp.returncode == 0
    assert "GO:" in cp.stdout


def test_go_no_go_missing_file_returns_no_go(tmp_path: Path):
    missing = tmp_path / "missing.json"
    cmd = ["python3", "scripts/run_go_no_go.py", "--input-json", str(missing), "--allow-missing"]
    cp = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert cp.returncode == 1
    assert "NO-GO" in cp.stdout
