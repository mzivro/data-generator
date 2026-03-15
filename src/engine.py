from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.utils.pydantic import create_model
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_SUFFIX,
    SYNTHETIC_FEW_SHOT_PREFIX,
)

from langchain_experimental.tabular_synthetic_data.openai import (
    create_openai_data_generator,
)

import pandas as pd
import numpy as np
import csv
import io
import os


class Engine:
    """
    Core engine responsible for synthetic tabular data generation.

    This class integrates a large language model with LangChain's
    synthetic data utilities to generate structured tabular data
    based on example rows.

    Attributes
    ----------
    llm : ChatOpenAI
        Language model used for generating synthetic rows.
    append_data : pandas.DataFrame or None
        Optional dataset appended before generated rows in the
        final output file.

    Notes
    -----
    Environment variables are loaded using ``python-dotenv``.
    The OpenAI model can be configured via the ``OPENAI_MODEL``
    environment variable.
    """

    def __init__(self):
        """
        Initialize the synthetic data engine.

        Loads environment variables and initializes the OpenAI
        chat model used for data generation.

        Notes
        -----
        Default model:
        ``gpt-4.1-mini`` if the ``OPENAI_MODEL`` environment
        variable is not defined.
        """
        load_dotenv()

        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        self.llm = ChatOpenAI(model=model)
        self.append_data = None

    def _generate_pydantic_model(self, df, model_name="DynamicModel"):
        """
        Create a dynamic Pydantic model based on a DataFrame schema.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame used to infer column names and data types.
        model_name : str, default="DynamicModel"
            Name of the generated Pydantic model.

        Returns
        -------
        pydantic.BaseModel
            Dynamically generated model describing the structure
            of rows in the dataset.

        Notes
        -----
        NumPy dtypes are mapped to native Python types:

        - ``int64`` → ``int``
        - ``float64`` → ``float``
        - ``object`` → ``str``
        - ``bool`` → ``bool``

        Unknown types default to ``str``.
        """
        type_map = {np.int64: int, np.float64: float, np.object_: str, np.bool_: bool}

        fields = {
            col: (type_map.get(df[col].dtype.type, str), ...) for col in df.columns
        }

        return create_model(model_name, **fields)

    def _row_to_prompt(self, row):
        """
        Convert a row dictionary into a prompt-friendly string.

        Parameters
        ----------
        row : dict
            Dictionary representing a single dataset row.

        Returns
        -------
        str
            Multiline string representation of the row in the format:

            ``column: value``

        Notes
        -----
        The resulting string is used as a few-shot example
        for the language model.
        """

        return "\n".join(f"{k}: {v}" for k, v in row.items())

    def _generate_example_dicts(self, df):
        """
        Convert a DataFrame into few-shot example dictionaries.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing sample rows.

        Returns
        -------
        list of dict
            List of dictionaries formatted for LangChain
            ``FewShotPromptTemplate``.
        """
        rows = df.to_dict(orient="records")

        sample_data_dict = [{"example": self._row_to_prompt(row)} for row in rows]

        return sample_data_dict

    def _generate_file(self, file_name, df):
        """
        Generate an output file from a DataFrame.

        Parameters
        ----------
        file_name : str
            Target file name including extension (``.csv`` or ``.xlsx``).
        df : pandas.DataFrame
            DataFrame to be written to the file.

        Returns
        -------
        tuple
            Tuple containing:

            - bytes
                File content ready for download.
            - str
                MIME type corresponding to the file format.

        Raises
        ------
        ValueError
            If an unsupported file extension is provided.
        """
        if file_name.endswith(".csv"):
            output = io.StringIO()
            df.to_csv(output, index=False)

            return output.getvalue().encode("utf-8"), "text/csv"

        elif file_name.endswith(".xlsx"):
            output = io.BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)

            return (
                output.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    def run(self, sample_data_df, subject, extra, runs, file_name):
        """
        Generate synthetic tabular data using an LLM.

        Parameters
        ----------
        sample_data_df : pandas.DataFrame
            Sample rows used as few-shot examples for the model.
        subject : str
            Description of the dataset topic used to guide generation.
        extra : str
            Additional instructions for the language model.
        runs : int
            Number of synthetic rows to generate.
        file_name : str
            Output file name including extension.

        Returns
        -------
        tuple
            Tuple containing:

            - bytes
                Generated file content.
            - str
                MIME type of the generated file.

        Notes
        -----
        Workflow:

        1. Infer a schema from the sample data.
        2. Construct a few-shot prompt template.
        3. Generate synthetic rows using the LLM.
        4. Convert results to a DataFrame.
        5. Optionally append existing data.
        6. Export the result as CSV or XLSX.
        """
        dynamic_model = self._generate_pydantic_model(sample_data_df)
        examples = self._generate_example_dicts(sample_data_df)

        openai_template = PromptTemplate(
            input_variables=["examples"], template="{example}"
        )

        prompt_template = FewShotPromptTemplate(
            prefix=SYNTHETIC_FEW_SHOT_PREFIX,
            examples=examples,
            suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
            input_variables=["subject", "extra"],
            example_prompt=openai_template,
        )

        generator = create_openai_data_generator(
            output_schema=dynamic_model, llm=self.llm, prompt=prompt_template
        )

        results = generator.generate(subject=subject, extra=extra, runs=runs)

        generated_rows = [r.model_dump() for r in results]
        generated_df = pd.DataFrame(generated_rows)
        generated_df = generated_df[sample_data_df.columns]

        if self.append_data is not None:
            generated_df = pd.concat(
                [self.append_data, generated_df], ignore_index=True
            )

        data, mime = self._generate_file(file_name, generated_df)

        return data, mime
