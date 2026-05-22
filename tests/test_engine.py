import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from engine import Engine


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
    with patch("engine.ChatOpenAI"):
        return Engine()


class TestEngineInit:
    @patch("engine.ChatOpenAI")
    def test_init_configures_llm(self, mock_chat_openai):
        Engine()
        mock_chat_openai.assert_called_once()


class TestGeneratePydanticModel:
    def test_maps_column_types(self, engine, sample_df):
        model = engine._generate_pydantic_model(sample_df)
        fields = model.model_fields
        assert fields["name"].annotation is str
        assert fields["age"].annotation is int
        assert fields["score"].annotation is float
        assert fields["active"].annotation is bool

    def test_unknown_dtype_defaults_to_str(self, engine):
        df = pd.DataFrame({"col": pd.to_datetime(["2024-01-01"])})
        model = engine._generate_pydantic_model(df)
        assert model.model_fields["col"].annotation is str

    def test_custom_model_name(self, engine, sample_df):
        model = engine._generate_pydantic_model(sample_df, model_name="CustomRow")
        assert model.__name__ == "CustomRow"


class TestRowToPrompt:
    def test_formats_row_as_key_value_lines(self, engine):
        row = {"name": "Alice", "age": 30}
        assert engine._row_to_prompt(row) == "name: Alice\nage: 30"


class TestGenerateExampleDicts:
    def test_returns_few_shot_examples(self, engine, sample_df):
        examples = engine._generate_example_dicts(sample_df)
        assert len(examples) == len(sample_df)
        assert all("example" in ex for ex in examples)
        assert "name: Alice" in examples[0]["example"]
        assert "age: 30" in examples[0]["example"]


class TestGenerateFile:
    def test_csv_output(self, engine, sample_df):
        data, mime = engine._generate_file("output.csv", sample_df)
        assert mime == "text/csv"
        content = data.decode("utf-8")
        assert "name,age,score,active" in content
        assert "Alice" in content

    def test_xlsx_output(self, engine, sample_df):
        data, mime = engine._generate_file("output.xlsx", sample_df)
        assert mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert isinstance(data, bytes)
        assert len(data) > 0
        roundtrip = pd.read_excel(io.BytesIO(data))
        pd.testing.assert_frame_equal(
            roundtrip, sample_df, check_dtype=False
        )


class TestCall:
    @patch("engine.create_openai_data_generator")
    def test_generates_csv_with_mocked_llm(
        self, mock_create_generator, engine, sample_df
    ):
        mock_row = MagicMock()
        mock_row.model_dump.return_value = {
            "name": "Carol",
            "age": 40,
            "score": 7.5,
            "active": True,
        }
        mock_generator = MagicMock()
        mock_generator.generate.return_value = [mock_row]
        mock_create_generator.return_value = mock_generator

        data, mime = engine(
            sample_data_df=sample_df,
            subject="people",
            extra="random values",
            runs=1,
            file_name="out.csv",
        )

        assert mime == "text/csv"
        assert "Carol" in data.decode("utf-8")
        mock_generator.generate.assert_called_once()
        call_kwargs = mock_generator.generate.call_args.kwargs
        assert call_kwargs["subject"] == "people"
        assert "Return ONLY a valid JSON object." in call_kwargs["extra"]

    @patch("engine.create_openai_data_generator")
    def test_handles_list_model_dump(
        self, mock_create_generator, engine, sample_df
    ):
        mock_row = MagicMock()
        mock_row.model_dump.return_value = [
            {"name": "Dan", "age": 22, "score": 6.0, "active": False},
            {"name": "Eve", "age": 33, "score": 8.5, "active": True},
        ]
        mock_generator = MagicMock()
        mock_generator.generate.return_value = [mock_row]
        mock_create_generator.return_value = mock_generator

        data, mime = engine(
            sample_data_df=sample_df,
            subject="people",
            extra="",
            runs=2,
            file_name="out.csv",
        )

        content = data.decode("utf-8")
        assert "Dan" in content
        assert "Eve" in content

    @patch("engine.create_openai_data_generator")
    def test_appends_existing_data(
        self, mock_create_generator, engine, sample_df
    ):
        mock_row = MagicMock()
        mock_row.model_dump.return_value = {
            "name": "Frank",
            "age": 50,
            "score": 5.0,
            "active": False,
        }
        mock_generator = MagicMock()
        mock_generator.generate.return_value = [mock_row]
        mock_create_generator.return_value = mock_generator

        engine.append_data = sample_df.iloc[[0]]
        data, _ = engine(
            sample_data_df=sample_df,
            subject="people",
            extra="",
            runs=1,
            file_name="out.csv",
        )

        content = data.decode("utf-8")
        assert content.count("Alice") == 1
        assert "Frank" in content
