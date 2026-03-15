# Synthetic Data Generator

A **Streamlit-based application** for generating synthetic tabular datasets using a Large Language Model (LLM).
The tool allows users to upload sample data and automatically generate new rows that follow the same schema and patterns.

The generation process uses **few-shot prompting** with example rows from the dataset to guide the model in producing realistic structured data.

---

# Features

* Upload **CSV** or **Excel** datasets
* Select **sample rows** to guide generation
* Provide **contextual prompts** to improve generated data quality
* Generate **synthetic rows using an LLM**
* Export results as **CSV or XLSX**
* Optionally **append generated data to existing datasets**

---

# Architecture

The project consists of three main components:

```
app.py                  # Entry point
│
├── data_generator.py   # Streamlit UI frontend
│
└── engine.py           # LLM-powered backend
```

### Application Flow

1. User uploads a dataset.
2. A subset of rows is selected as **few-shot examples**.
3. A **dynamic schema** is inferred from the dataset.
4. The LLM generates new rows based on examples and prompts.
5. Results are exported to a downloadable file.

---

# Technologies

* Python
* Streamlit
* Pandas
* NumPy
* LangChain
* OpenAI API
* Pydantic

---

# Installation

Clone the repository:

```bash
git clone https://github.com/mzivro/data-generator.git
cd data-generator
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux / MacOS
venv\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root.

Example:

```
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
```

If `OPENAI_MODEL` is not provided, the default model is used (gpt-4.1-mini).

---

# Running the Application

Start the Streamlit application:

```bash
streamlit run src/app.py
```

Then open the browser at:

```
http://localhost:8501
```

---

# Usage

1. Upload a dataset (`.csv` or `.xlsx`)
2. Select:

   * **Start point** (row index)
   * **Number of steps** (sample rows)
3. Provide optional prompts:

   * **Subject** - dataset topic
   * **Extra prompt** - generation instructions
4. Choose:

   * Number of rows to generate
   * Output format
5. Click **Generate**
6. Download the generated dataset.

---

# Example Use Cases

* Creating synthetic datasets for **machine learning**
* Expanding **small datasets**
* Generating **test data for applications**
* Producing **privacy-safe data replacements**

---

# Limitations

* Generated data quality depends on the **sample examples and prompts**
* Large datasets may require careful sampling
* LLM outputs may still require **validation or post-processing**

---

# License

MIT License. Feel free to use, modify, and build upon this project.
