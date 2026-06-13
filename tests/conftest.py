from unittest.mock import patch

import pandas as pd
import pytest

from engine import Engine
from helpers import API_KEY, MODEL, TEMPERATURE


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob"],
            "age": [30, 25],
            "score": [9.5, 8.0],
            "active": [True, False],
        }
    )


@pytest.fixture
def engine():
    with patch("engine.ChatOpenAI") as mock_chat_openai:
        instance = Engine(API_KEY, MODEL, TEMPERATURE)
        instance._mock_chat_openai = mock_chat_openai
        yield instance
