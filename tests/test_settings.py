import pytest
from pydantic import ValidationError

from settings import Settings


@pytest.fixture
def valid_settings_kwargs():
    return {"api_key": "sk-test", "model": "gpt-4.1-mini", "temperature": 0.0}


def test_defaults(valid_settings_kwargs):
    settings = Settings(**valid_settings_kwargs)
    assert settings.model == "gpt-4.1-mini"
    assert settings.temperature == 0.0


def test_custom_values():
    settings = Settings(api_key="sk-test", model="gpt-4o", temperature=1.5)
    assert settings.model == "gpt-4o"
    assert settings.temperature == 1.5


@pytest.mark.parametrize("api_key,expected", [("  sk-test  ", "sk-test"), ("key", "key")])
def test_api_key_is_stripped(api_key, expected):
    settings = Settings(api_key=api_key)
    assert settings.api_key == expected


@pytest.mark.parametrize("api_key", ["", "   "])
def test_empty_api_key_raises(api_key):
    with pytest.raises(ValidationError, match="No OpenAI API key provided"):
        Settings(api_key=api_key)


@pytest.mark.parametrize("model,expected", [("  gpt-4o  ", "gpt-4o"), ("gpt-4.1-mini", "gpt-4.1-mini")])
def test_model_is_stripped(model, expected):
    settings = Settings(api_key="sk-test", model=model)
    assert settings.model == expected


@pytest.mark.parametrize("model", ["", "   "])
def test_empty_model_raises(model):
    with pytest.raises(ValidationError, match="Model cannot be empty"):
        Settings(api_key="sk-test", model=model)
