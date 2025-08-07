import json

import pandas as pd
import pytest

from utils.logging import JsonLogger, parse_json_lines, read_json_log


@pytest.fixture
def log_path(tmp_path):
    """Provides a temporary path for a log file."""
    return tmp_path / "test_log.json"


def test_logger_as_context_manager(log_path):
    """Tests that the logger creates a file and can be used as a context manager."""
    assert not log_path.exists()
    with JsonLogger(str(log_path)) as logger:
        assert logger._file is not None
        assert not logger._file.closed
        logger.log({"epoch": 0, "loss": 0.5})
    assert log_path.exists()
    assert logger._file is None


def test_logger_start_stop(log_path):
    """Tests manual start and stop of the logger."""
    logger = JsonLogger(str(log_path))
    with pytest.raises(IOError):
        logger.log({"epoch": 0, "loss": 0.5})  # Should fail if not started

    logger.start()
    assert logger._file is not None
    logger.log({"epoch": 1, "loss": 0.4})
    logger.stop()
    assert logger._file is None

    with open(log_path, "r") as f:
        content = f.read()
    assert '{"epoch":1,"loss":0.4}' in content


def test_logger_appends_data(log_path):
    """Tests that subsequent logs are appended to the file."""
    with JsonLogger(str(log_path)) as logger:
        logger.log({"epoch": 0, "loss": 0.5})

    with JsonLogger(str(log_path)) as logger:
        logger.log({"epoch": 1, "loss": 0.4})

    with open(log_path, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2
    assert '{"epoch":0,"loss":0.5}' in lines[0]
    assert '{"epoch":1,"loss":0.4}' in lines[1]


def test_logger_default_filter(log_path):
    """Tests that the default filter only logs numeric values."""
    with JsonLogger(str(log_path)) as logger:
        logger.log({"step": 1, "metric": 0.99, "comment": "test", "valid": True})

    with open(log_path, "r") as f:
        data = json.loads(f.read())

    # Asserting against a set of items makes the test robust to key order
    assert set(data.items()) == {("step", 1), ("metric", 0.99)}


def test_logger_no_filter(log_path):
    """Tests the logger with filtering disabled."""
    with JsonLogger(str(log_path), filter_fn=None) as logger:
        logger.log({"step": 1, "comment": "test", "valid": True})

    with open(log_path, "r") as f:
        data = json.loads(f.read())

    assert data == {"step": 1, "comment": "test", "valid": True}


def test_logger_custom_filter(log_path):
    """Tests using a custom filter function."""
    # A filter that only allows string values
    str_filter = lambda k, v: isinstance(v, str)
    with JsonLogger(str(log_path), filter_fn=str_filter) as logger:
        logger.log({"name": "run-1", "id": 123, "status": "finished"})

    with open(log_path, "r") as f:
        data = json.loads(f.read())

    assert data == {"name": "run-1", "status": "finished"}


# --- Tests for Reading Functions ---


def test_read_non_existent_log(caplog):
    """Tests reading a log file that does not exist."""
    df = read_json_log("non_existent_file.json")
    assert df.empty
    assert "Log file not found" in caplog.text


def test_read_empty_log(log_path):
    """Tests reading an empty log file."""
    log_path.touch()
    df = read_json_log(str(log_path))
    assert df.empty


def test_read_and_parse_malformed_log(log_path, caplog):
    """Tests that malformed JSON lines are skipped during reading."""
    log_content = [
        '{"epoch": 0, "loss": 0.5}\n',
        "this is not json\n",
        '{"epoch": 1, "loss": 0.4, "lr": 1e-4}\n',
        '{"epoch": 2, "loss": 0.3, "lr": 1e-4',  # Incomplete line
    ]
    log_path.write_text("".join(log_content))

    # Test the generator directly
    records = list(parse_json_lines(str(log_path)))
    assert len(records) == 2
    assert records[0] == {"epoch": 0, "loss": 0.5}
    assert records[1] == {"epoch": 1, "loss": 0.4, "lr": 1e-4}
    assert "Skipping malformed JSON on line 2" in caplog.text
    assert "Skipping malformed JSON on line 4" in caplog.text

    # Test the DataFrame reader
    df = read_json_log(str(log_path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["epoch"].tolist() == [0, 1]


def test_read_log_with_required_keys(log_path):
    """Tests filtering log entries by required keys."""
    log_content = [
        '{"epoch": 0, "loss": 0.5}\n',
        '{"epoch": 1, "loss": 0.4, "accuracy": 0.91}\n',
        '{"epoch": 2, "accuracy": 0.92}\n',
    ]
    log_path.write_text("".join(log_content))

    df = read_json_log(str(log_path), required_keys=["accuracy"])
    assert len(df) == 2
    assert df["epoch"].tolist() == [1, 2]

    df_all = read_json_log(str(log_path))
    assert len(df_all) == 3
