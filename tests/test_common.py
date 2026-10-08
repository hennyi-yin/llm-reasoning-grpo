from src.common import load_config


def test_json_scientific_notation_retains_numeric_types(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"learning_rate": 1e-05, "beta": 0.0}', encoding="utf-8")
    assert load_config(path)["learning_rate"] == 1e-5
