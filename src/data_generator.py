import streamlit as st
import pandas as pd

from engine import Engine


class DataGenerator:
    """
    Streamlit-based user interface for synthetic tabular data generation.

    The class manages the application layout, file uploads, user inputs,
    and interaction with the :class:`Engine` responsible for generating
    synthetic data.

    Notes
    -----
    The instance of :class:`Engine` is stored in ``st.session_state`` to
    persist across Streamlit reruns.
    """

    def __init__(self):
        """
        Initialize the Streamlit application.

        Sets up the page configuration, application title, and ensures
        that an :class:`Engine` instance exists in the Streamlit session
        state.

        Notes
        -----
        The engine instance is created only once per user session.
        """
        st.set_page_config(page_title="Data Generator", layout="centered")
        st.title("Synthetic Data Generator")

        if "engine" not in st.session_state:
            st.session_state.engine = Engine()

    def run(self):
        """
        Execute the main Streamlit application workflow.

        Handles the following steps:

        - Uploading a sample dataset (CSV or XLSX)
        - Selecting sample rows used as examples
        - Collecting generation parameters
        - Triggering synthetic data generation
        - Providing the generated file for download

        Raises
        ------
        Exception
            Displays an error message in the UI if file loading or
            generation fails.

        Notes
        -----
        The selected sample rows are passed to the :class:`Engine`
        which uses them as few-shot examples for the LLM-based
        synthetic data generator.
        """
        uploaded_file = st.file_uploader("Load sample data", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)

                st.session_state.engine.whole_file = df

                col_1, col_2 = st.columns(2)

                with col_1:
                    start_point = st.number_input(
                        "Start point", min_value=0, value=0, step=1
                    )

                with col_2:
                    steps = st.number_input("Steps", min_value=1, value=5, step=1)

                st.subheader("Sample data")

                sample_data_df = df.iloc[start_point : start_point + steps]

                st.dataframe(sample_data_df)

                subject = st.text_input(
                    "Enter data subject",
                    value="example data",
                    help="A topic about the sample data, it might improve the results",
                )

                extra = st.text_input(
                    "Enter extra prompt",
                    value="choose all values at random",
                    help="Extra prompt for LLM during data generation",
                )

                runs = st.number_input(
                    "Choose runs count",
                    min_value=1,
                    value=5,
                    step=1,
                    help="Count of data rows which LLM must generate",
                )

                file_name = st.text_input("Enter file name", value="generated")
                file_format = st.radio(
                    "Choose a file format", [".csv", ".xlsx"], horizontal=True
                )

                append_mode = st.radio(
                    "What do you want to append?",
                    [
                        "Do not append anything",
                        "Append sample data",
                        "Append whole data",
                    ],
                    captions=[
                        "Does not append anything to the file",
                        "Appends sample data ahead of the file",
                        "Appends whole data ahead of the file",
                    ],
                    horizontal=True,
                )

                if append_mode == "Append sample data":
                    st.session_state.engine.append_data = sample_data_df
                elif append_mode == "Append whole data":
                    st.session_state.engine.append_data = df
                else:
                    st.session_state.engine.append_data = None

                output_file_name = file_name + file_format

                st.subheader(f"Output file name: {output_file_name}")

                if st.button("Generate", width="stretch"):
                    try:
                        data, mime = st.session_state.engine.run(
                            sample_data_df, subject, extra, runs, output_file_name
                        )

                        st.download_button(
                            label="Success! Click here to download the file",
                            data=data,
                            file_name=output_file_name,
                            mime=mime,
                            width="stretch",
                        )
                    except Exception as e:
                        st.error(f"An error has occured: {e}")
            except Exception as e:
                st.error(f"An error has occured: {e}")
