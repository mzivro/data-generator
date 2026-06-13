import io
from unittest.mock import MagicMock, patch

import pandas as pd

API_KEY = "sk-test"
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0


def decode_csv(data):
    return pd.read_csv(io.BytesIO(data))


def stub_llm_results(rows):
    results = []
    for row in rows:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = row
        results.append(mock_result)
    return results


def patch_generator(rows):
    mock_generator = MagicMock()
    mock_generator.generate.return_value = stub_llm_results(rows)
    return patch(
        "engine.create_openai_data_generator", return_value=mock_generator
    ), mock_generator
