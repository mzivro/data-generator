import io
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import ValidationError

from engine import Engine
from helpers import API_KEY, MODEL, TEMPERATURE, decode_csv, patch_generator


def test_init_passes_settings_to_chat_openai(engine):
    engine._mock_chat_openai.assert_called_once_with(
        api_key=API_KEY,
        model=MODEL,
        temperature=TEMPERATURE,
    )


def test_generate_pydantic_model_maps_column_types(engine, sample_df):
    model = engine._generate_pydantic_model(sample_df)
    fields = model.model_fields
    assert fields["name"].annotation is str
    assert fields["age"].annotation is int
    assert fields["score"].annotation is float
    assert fields["active"].annotation is bool


def test_generate_pydantic_model_unknown_dtype_defaults_to_str(engine):
    df = pd.DataFrame({"col": pd.to_datetime(["2024-01-01"])})
    model = engine._generate_pydantic_model(df)
    assert model.model_fields["col"].annotation is str


def test_generate_pydantic_model_custom_name(engine, sample_df):
    model = engine._generate_pydantic_model(sample_df, model_name="CustomRow")
    assert model.__name__ == "CustomRow"


def test_row_to_prompt(engine):
    assert engine._row_to_prompt({"name": "Alice", "age": 30}) == "name: Alice\nage: 30"


def test_generate_example_dicts(engine, sample_df):
    examples = engine._generate_example_dicts(sample_df)
    assert len(examples) == len(sample_df)
    assert examples[0] == {
        "example": "name: Alice\nage: 30\nscore: 9.5\nactive: True"
    }


def test_generate_file_csv(engine, sample_df):
    data, mime = engine._generate_file("output.csv", sample_df)
    assert mime == "text/csv"
    pd.testing.assert_frame_equal(decode_csv(data), sample_df, check_dtype=False)


def test_generate_file_xlsx(engine, sample_df):
    data, mime = engine._generate_file("output.xlsx", sample_df)
    assert mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    roundtrip = pd.read_excel(io.BytesIO(data))
    pd.testing.assert_frame_equal(roundtrip, sample_df, check_dtype=False)


def test_call_exports_generated_rows(engine, sample_df):
    generated = {"name": "Carol", "age": 40, "score": 7.5, "active": True}
    patch_ctx, mock_generator = patch_generator([generated])

    with patch_ctx:
        data, mime = engine(
            sample_data_df=sample_df,
            subject="people",
            extra="random values",
            runs=1,
            file_name="out.csv",
        )

    assert mime == "text/csv"
    assert decode_csv(data).iloc[0].to_dict() == generated

    mock_generator.generate.assert_called_once()
    kwargs = mock_generator.generate.call_args.kwargs
    assert kwargs["subject"] == "people"
    assert kwargs["runs"] == 1
    assert kwargs["extra"].startswith("random values")
    assert "Return ONLY a valid JSON object." in kwargs["extra"]


def test_call_extends_rows_when_model_dump_returns_list(engine, sample_df):
    patch_ctx, _ = patch_generator(
        [
            [
                {"name": "Dan", "age": 22, "score": 6.0, "active": False},
                {"name": "Eve", "age": 33, "score": 8.5, "active": True},
            ]
        ]
    )

    with patch_ctx:
        data, _ = engine(sample_df, "people", "", runs=2, file_name="out.csv")

    assert decode_csv(data)["name"].tolist() == ["Dan", "Eve"]


def test_call_prepends_append_data(engine, sample_df):
    generated = {"name": "Frank", "age": 50, "score": 5.0, "active": False}
    patch_ctx, _ = patch_generator([generated])
    engine.append_data = sample_df.iloc[[0]]

    with patch_ctx:
        data, _ = engine(sample_df, "people", "", runs=1, file_name="out.csv")

    result = decode_csv(data)
    assert result.iloc[0]["name"] == "Alice"
    assert result.iloc[1]["name"] == "Frank"


def test_init_rejects_empty_api_key():
    with patch("engine.ChatOpenAI"):
        with pytest.raises(ValidationError, match="No OpenAI API key provided"):
            Engine("", MODEL, TEMPERATURE)
