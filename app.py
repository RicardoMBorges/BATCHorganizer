# app.py  --- Streamlit interface for CENABIO / generic extract database
# ------------------------------------------------------------
# Requirements (no pyarrow anywhere):
#   streamlit
#   pandas
#   plotly
#   seaborn (optional, only if you add heatmaps)
#
# Run with:
#   streamlit run app.py
# ------------------------------------------------------------

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List
from PIL import Image

import streamlit as st
import pandas as pd
import plotly.express as px

# Make sure we can import your existing helper module
# (collections2batches.py should be one level above this app by default)
sys.path.append(os.path.abspath(".."))
import collections2batches as c2b   # your existing module


# -----------------------------
# 1. Basic configuration
# -----------------------------
st.set_page_config(
    page_title="Plant Extracts – Batch Organizer",
    layout="wide",
)

# ---- session state keys ----
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "cenabiodb_merged" not in st.session_state:
    st.session_state.cenabiodb_merged = None
if "cenabiodb_merged2_filt" not in st.session_state:
    st.session_state.cenabiodb_merged2_filt = None
if "cenabiodb_merged2_clean" not in st.session_state:
    st.session_state.cenabiodb_merged2_clean = None
if "batches" not in st.session_state:
    st.session_state.batches = None
if "summary_df" not in st.session_state:
    st.session_state.summary_df = None
if "clean_csv_path" not in st.session_state:
    st.session_state.clean_csv_path = None
if "summary_output_path" not in st.session_state:
    st.session_state.summary_output_path = None
if "batches_output_path" not in st.session_state:
    st.session_state.batches_output_path = None
if "params" not in st.session_state:
    st.session_state.params = None

# ============================================================
# Load logos
# ============================================================
STATIC_DIR = Path(__file__).parent / "static"
LOGO_PATH = STATIC_DIR / "LAABio.png"
BATCH_LOGO_PATH = STATIC_DIR / "BatchOrganizer.png"

try:
    logo = Image.open(LOGO_PATH)
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at static/LAABio.png")

try:
    batch_logo_img = Image.open(BATCH_LOGO_PATH)
    st.sidebar.image(batch_logo_img, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("logo_BatchOrganizer not found at static/BatchOrganizer.png")

#st.sidebar.markdown("""---""")

# PayPal donate button
st.sidebar.markdown("""
<hr>
<center>
<p>To support the app development:</p>
<a href="https://www.paypal.com/donate/?business=2FYTFNDV4F2D4&no_recurring=0&item_name=Support+with+%245+→+Send+receipt+to+tlc2chrom.app@gmail.com+with+your+login+email+→+Access+within+24h!&currency_code=USD" target="_blank">
    <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_SM.gif" alt="Donate with PayPal button" border="0">
</a>
</center>
""", unsafe_allow_html=True)

st.sidebar.markdown("""---""")

TUTORIAL_URL = "https://github.com/RicardoMBorges/BATCHorganizer/blob/main/README.md"
try:
    st.sidebar.link_button("📘 Tutorial", TUTORIAL_URL)
except Exception:
    st.sidebar.markdown(
        f'<a href="{TUTORIAL_URL}" target="_blank">'
        '<button style="padding:0.6rem 1rem; border-radius:8px; border:1px solid #ddd; cursor:pointer;">📘 Tutorial</button>'
        '</a>',
        unsafe_allow_html=True,
    )


MockData_URL = "https://github.com/RicardoMBorges/BATCHorganizer/tree/main/Mock%20Data"
try:
    st.sidebar.link_button("Mock Data", MockData_URL)
except Exception:
    st.sidebar.markdown(
        f'<a href="{MockData_URL}" target="_blank">'
        '<button style="padding:0.6rem 1rem; border-radius:8px; border:1px solid #ddd; cursor:pointer;">Mock Data</button>'
        '</a>',
        unsafe_allow_html=True,
    )
    
VIDEO_URL = "https://youtu.be/2Ou160tkJ5c"
try:
    st.sidebar.link_button("Video", VIDEO_URL)
except Exception:
    st.sidebar.markdown(
        f'<a href="{VIDEO_URL}" target="_blank">'
        '<button style="padding:0.6rem 1rem; border-radius:8px; border:1px solid #ddd; cursor:pointer;">📘 Tutorial</button>'
        '</a>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("""---""")


st.title("Batch Organizer")

st.markdown(
    """
This app loads extract collection metadata,  
generates taxonomic summaries (family / genus / species),  
and creates hierarchical acquisition batches (Family → Genus → Species) with QCs.
"""
)


# -----------------------------
# 2. Helpers to load and merge (CENABIO mode)
# -----------------------------
def load_raw_tables(folder_path: str) -> Dict[str, pd.DataFrame]:
    """Load the four TXT files from the 'downloaded' folder."""
    folder = Path(folder_path)

    tables = {}

    files_expected = {
        "appSampleTPlant": "appSampleTPlant_1384151563.txt",
        "appSampleTrack": "appSampleTrack_1384157081.txt",
        "appSampleCompound": "appSampleCompound_1384157249.txt",
        "appTPlant": "appTPlant_1384151712.txt",
    }

    for key, fname in files_expected.items():
        fpath = folder / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing file: {fpath}")
        if key == "appSampleTrack":
            tables[key] = pd.read_csv(fpath, sep="\t", on_bad_lines="skip")
        else:
            tables[key] = pd.read_csv(fpath, sep="\t")

    return tables


def merge_cenabio_tables(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reproduce your merge logic to obtain:
      - cenabiodb_merged
      - cenabiodb_merged2_filt
      - cenabiodb_merged2_clean (final)
    """

    appSampleTPlant = tables["appSampleTPlant"].copy()
    appSampleTrack = tables["appSampleTrack"].copy()
    appSampleCompound = tables["appSampleCompound"].copy()  # currently unused, kept for completeness
    appTPlant = tables["appTPlant"].copy()

    # Ensure plant registration codes are strings
    appSampleTPlant["Reg Planta"] = appSampleTPlant["Reg Planta"].astype(str)
    appTPlant["Reg TPlanta"] = appTPlant["Reg TPlanta"].astype(str)

    # 1) Join sample info with botanical info
    cenabiodb_merged = pd.merge(
        appSampleTPlant,
        appTPlant,
        left_on="Reg Planta",
        right_on="Reg TPlanta",
        how="right",
    )

    # Filter a first “simple” subset
    cenabiodb_merged_filt = cenabiodb_merged[
        [
            "Reg Amostra",
            "Reg Planta",
            "Família",
            "Gênero",
            "Espécies",
            "Comum",
            "Nome do Herbário",
        ]
    ]

    # 2) Join with the physical tracking table (position etc.)
    cenabiodb_merged2 = pd.merge(
        cenabiodb_merged,
        appSampleTrack,
        left_on="Reg Amostra",
        right_on="Reg Amostra",
        how="right",
    )

    cenabiodb_merged2_filt = cenabiodb_merged2[
        [
            "Reg Amostra",
            "Reg Planta",
            "Família",
            "Gênero",
            "Espécies",
            "Comum",
            "Nome do Herbário",
            "Câmara Fria",
            "Armário",
            "Gaveta",
            "Coluna",
            "Linha",
            "Cor da tampa",
        ]
    ]

    # Clean: drop rows without sample code and without family
    cenabiodb_merged2_clean = cenabiodb_merged2_filt.dropna(subset=["Reg Amostra"])
    cenabiodb_merged2_clean = cenabiodb_merged2_clean.dropna(subset=["Família"])

    return cenabiodb_merged, cenabiodb_merged2_filt, cenabiodb_merged2_clean


def save_clean_table(df: pd.DataFrame, base_folder: str | None) -> str:
    """
    Save the clean table as CSV (semicolon-separated) and return the path.

    - If base_folder is not None: save alongside that folder (CENABIO mode).
    - If base_folder is None: save to current working directory (generic mode).
    """
    if base_folder:
        base_dir = Path(base_folder).parent
        out_path = base_dir / "cenabiodb_merged2_clean.csv"
    else:
        out_path = Path("cenabiodb_merged2_clean.csv")

    df.to_csv(out_path, sep=";", index=False)
    return str(out_path)


# -----------------------------
# 3. Plotting helpers (inside app)
# -----------------------------
def plot_family_bar(df: pd.DataFrame):
    """Interactive bar plot: number of samples per family."""
    family_counts = df["Família"].value_counts().sort_values(ascending=False)
    plot_df = family_counts.reset_index()
    plot_df.columns = ["Família", "N_amostras"]
    fig = px.bar(
        plot_df,
        x="Família",
        y="N_amostras",
        title="Distribution by Family (only rows with full taxonomy)",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def plot_genus_bar(df: pd.DataFrame):
    """Interactive bar plot: number of samples per genus."""
    genus_counts = df["Gênero"].value_counts().sort_values(ascending=False)
    plot_df = genus_counts.reset_index()
    plot_df.columns = ["Gênero", "N_amostras"]
    fig = px.bar(
        plot_df,
        x="Gênero",
        y="N_amostras",
        title="Distribution by Genus",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def plot_global_sunburst(df: pd.DataFrame):
    """Global sunburst: Family → Genus → Species."""
    df_plot = df.dropna(subset=["Família", "Gênero", "Espécies"]).copy()
    df_plot["count"] = 1

    fig = px.sunburst(
        df_plot,
        path=["Família", "Gênero", "Espécies"],
        values="count",
        title="Sunburst – Family, Genus, Species (global)",
        width=800,
        height=800,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_batch_sunburst(df_batch: pd.DataFrame, batch_name: str):
    """Sunburst for a single batch (only Amostra rows)."""
    df_samples = df_batch.copy()
    # Robust filter for sample type
    if "Tipo" in df_samples.columns:
        tipo_norm = df_samples["Tipo"].astype(str).str.strip().str.lower()
        df_samples = df_samples[tipo_norm.eq("amostra")]
    # else: if there is no 'Tipo' column, assume everything is a sample

    if df_samples.empty:
        st.info(f"No sample rows found for {batch_name}.")
        return

    # Fill missing taxonomy labels
    for col in ["Família", "Gênero", "Espécies"]:
        if col in df_samples.columns:
            df_samples[col] = (
                df_samples[col]
                .replace("", None)
                .fillna("Not specified")
            )
        else:
            df_samples[col] = "Not specified"

    fig = px.sunburst(
        df_samples,
        path=["Família", "Gênero", "Espécies"],
        title=f"Sunburst – {batch_name}",
        width=700,
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# 4. Sidebar – configuration
# -----------------------------
st.sidebar.header("Metadata source")

data_source = st.sidebar.radio(
    "Choose the metadata source",
    ("CENABIO TXT tables", "Generic CSV/Excel metadata"),
    index=0,
)

DEFAULT_FOLDER = r"C:\Users\borge\Documents\ex_extracta\downloaded"

folder_path = None
uploaded_file = None

if data_source == "CENABIO TXT tables":
    folder_path = st.sidebar.text_input(
        "Folder with CENABIO TXT files",
        value=DEFAULT_FOLDER,
    )
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload metadata file (.csv, .xls, .xlsx)",
        type=["csv", "xls", "xlsx"],
    )

st.sidebar.header("Batch configuration")

samples_per_batch = st.sidebar.number_input(
    "Samples per batch (excluding QCs)",
    min_value=10,
    max_value=200,
    value=80,
    step=1,
)

batch_structure_raw = st.sidebar.text_input(
    "Batch structure (block sizes, comma-separated)",
    value="24,24,32",  # must sum to samples_per_batch
)

qc_structure_raw = st.sidebar.text_input(
    "QC structure (number of QC blocks per block in batch_structure)",
    value="3,3,2",
)

qc_samples_str = st.sidebar.text_input(
    "QC labels (comma-separated)",
    value="Blank,QC_Inter_Batch,QC_Intra_Batch",
)

run_pipeline = st.sidebar.button("Run / recompute pipeline")

st.sidebar.markdown("""---""")
st.sidebar.markdown("""
### Citation

Please cite:

[DOI](https://doi.org/10.1038/s41592-025-02660-z)
""")


# -----------------------------
# 5. Main logic – heavy pipeline only on button click
# -----------------------------
if run_pipeline or not st.session_state.pipeline_ready:
    # 5.1 parse structures and QC labels
    try:
        batch_structure_list = [
            int(x.strip()) for x in batch_structure_raw.split(",") if x.strip()
        ]
        qc_structure_list = [
            int(x.strip()) for x in qc_structure_raw.split(",") if x.strip()
        ]
        qc_samples = tuple(
            x.strip() for x in qc_samples_str.split(",") if x.strip()
        )
    except Exception as exc:
        st.error(f"Error parsing batch/QC structures: {exc}")
        st.stop()

    if sum(batch_structure_list) != samples_per_batch:
        st.error(
            f"The sum of batch_structure ({sum(batch_structure_list)}) "
            f"must equal samples_per_batch ({samples_per_batch})."
        )
        st.stop()

    if len(batch_structure_list) != len(qc_structure_list):
        st.error("batch_structure and qc_structure must have the same length.")
        st.stop()

    if not qc_samples:
        st.error("You must provide at least one QC label.")
        st.stop()

    # 5.2 load and prepare data depending on source
    st.subheader("Data loading and merging")

    if data_source == "CENABIO TXT tables":
        # ----- original TXT workflow -----
        if not folder_path:
            st.error("Please provide the folder path containing the CENABIO TXT files.")
            st.stop()

        try:
            raw_tables = load_raw_tables(folder_path)
        except Exception as exc:
            st.error(f"Error loading TXT tables: {exc}")
            st.stop()

        st.success("TXT tables loaded successfully.")

        with st.expander("Show raw tables (heads)", expanded=False):
            for name, df_raw in raw_tables.items():
                st.markdown(f"**{name}** – shape: {df_raw.shape}")
                st.dataframe(df_raw.head())

        cenabiodb_merged, cenabiodb_merged2_filt, cenabiodb_merged2_clean = merge_cenabio_tables(raw_tables)

        st.markdown(
            f"""
- **cenabiodb_merged** shape: `{cenabiodb_merged.shape}`  
- **cenabiodb_merged2_filt** shape: `{cenabiodb_merged2_filt.shape}`  
- **cenabiodb_merged2_clean** (no NaN in Reg Amostra & Família) shape: `{cenabiodb_merged2_clean.shape}`
"""
        )

        with st.expander("Preview of clean table (cenabiodb_merged2_clean)", expanded=True):
            st.dataframe(cenabiodb_merged2_clean.head(50))

        clean_csv_path = save_clean_table(cenabiodb_merged2_clean, base_folder=folder_path)
        st.success(f"Clean table saved to: `{clean_csv_path}`")

        base_dir = Path(folder_path).parent

        # columns are already 'Família', 'Gênero', 'Espécies'
        family_col_source = "Família"
        genus_col_source = "Gênero"
        species_col_source = "Espécies"

    else:
        # ----- generic CSV/Excel workflow -----
        if uploaded_file is None:
            st.error("Please upload a metadata CSV or Excel file in the sidebar.")
            st.stop()

        fname = uploaded_file.name.lower()
        try:
            if fname.endswith(".csv"):
                # Use python engine and sep=None to autodetect delimiter (comma, semicolon, tab, etc.)
                try:
                    meta_df = pd.read_csv(uploaded_file, sep=None, engine="python")
                except Exception as exc:
                    st.error(f"Error reading CSV file with automatic delimiter detection: {exc}")
                    st.stop()
            else:
                try:
                    meta_df = pd.read_excel(uploaded_file)
                except Exception as exc:
                    st.error(f"Error reading Excel file: {exc}")
                    st.stop()

        except Exception as exc:
            st.error(f"Error reading uploaded file: {exc}")
            st.stop()

        st.markdown(f"**Uploaded metadata** – shape: `{meta_df.shape}`")
        with st.expander("Preview of uploaded metadata", expanded=True):
            st.dataframe(meta_df.head(50))

        cols = list(meta_df.columns)
        if len(cols) < 3:
            st.error("Metadata must have at least three columns (for Family, Genus, Species).")
            st.stop()

        st.markdown("### Column mapping for taxonomy")

        family_col_source = st.selectbox(
            "Column for Family (Família)",
            cols,
            key="family_col_select",
        )
        genus_col_source = st.selectbox(
            "Column for Genus (Gênero)",
            cols,
            key="genus_col_select",
        )
        species_col_source = st.selectbox(
            "Column for Species (Espécies)",
            cols,
            key="species_col_select",
        )

        # Build a dataframe equivalent to cenabiodb_merged2_clean
        # Keep ALL original columns, just add canonical ones
        cenabiodb_merged2_clean = meta_df.copy()
        cenabiodb_merged2_clean["Família"] = cenabiodb_merged2_clean[family_col_source]
        cenabiodb_merged2_clean["Gênero"] = cenabiodb_merged2_clean[genus_col_source]
        cenabiodb_merged2_clean["Espécies"] = cenabiodb_merged2_clean[species_col_source]

        # For compatibility, just mirror
        cenabiodb_merged = cenabiodb_merged2_clean
        cenabiodb_merged2_filt = cenabiodb_merged2_clean

        clean_csv_path = save_clean_table(cenabiodb_merged2_clean, base_folder=None)
        st.success(
            f"Generic metadata loaded. Family={family_col_source}, "
            f"Genus={genus_col_source}, Species={species_col_source}.  "
            f"Clean table saved to: `{clean_csv_path}`"
        )

        base_dir = Path.cwd()

    # 5.3 batch creation (common for both modes)
    st.subheader("Hierarchical batch creation (Family → Genus → Species)")

    batches_output_path = base_dir / "batches_hierarchical"
    batches_output_path.mkdir(parents=True, exist_ok=True)

    st.markdown(
        f"Batches will be written as CSV files in: `{batches_output_path}`"
    )

    try:
        batches = c2b.criar_batches_hierarquico_fam_gen_esp(
            data=cenabiodb_merged2_clean,
            output_path=str(batches_output_path),
            samples_per_batch=int(samples_per_batch),
            qc_samples=qc_samples,
            qc_structure=tuple(qc_structure_list),
            batch_structure=tuple(batch_structure_list),
            family_col="Família",
            genero_col="Gênero",
            especie_col="Espécies",
            random_state=42,
            keep_na_labels=True,
        )
        st.success(f"{len(batches)} hierarchical batches created.")
    except Exception as exc:
        st.error(f"Error while creating batches: {exc}")
        st.stop()

    # 5.4 batch composition summary
    st.subheader("Family/Genus composition per batch")

    summary_output_path = base_dir / "batches_hierarchical_summary"
    summary_output_path.mkdir(parents=True, exist_ok=True)

    try:
        summary_df = c2b.gerar_resumo_composicao(
            batches,
            output_path=str(summary_output_path),
        )
    except Exception as exc:
        st.error(f"Error while generating composition summary: {exc}")
        st.stop()

    # ---- store everything in session_state ----
    st.session_state.pipeline_ready = True
    st.session_state.cenabiodb_merged = cenabiodb_merged
    st.session_state.cenabiodb_merged2_filt = cenabiodb_merged2_filt
    st.session_state.cenabiodb_merged2_clean = cenabiodb_merged2_clean
    st.session_state.batches = batches
    st.session_state.summary_df = summary_df
    st.session_state.clean_csv_path = clean_csv_path
    st.session_state.summary_output_path = str(summary_output_path)
    st.session_state.batches_output_path = str(batches_output_path)

    # also store the parameters used
    if data_source == "CENABIO TXT tables":
        folder_for_params = folder_path
    else:
        folder_for_params = "N/A (generic file)"

    st.session_state.params = {
        "data_source": data_source,
        "folder_path": folder_for_params,
        "samples_per_batch": int(samples_per_batch),
        "batch_structure": batch_structure_list,
        "qc_structure": qc_structure_list,
        "qc_samples": list(qc_samples),
        "family_col_source": family_col_source,
        "genus_col_source": genus_col_source,
        "species_col_source": species_col_source,
    }

else:
    # no new run, just reuse what we already computed
    if not st.session_state.pipeline_ready:
        st.info(
            "Set the metadata source and batch parameters on the left, "
            "then click **Run / recompute pipeline**."
        )
        st.stop()

# from here on, always use objects from session_state
cenabiodb_merged2_clean = st.session_state.cenabiodb_merged2_clean
batches = st.session_state.batches
summary_df = st.session_state.summary_df
clean_csv_path = st.session_state.clean_csv_path
summary_output_path = Path(st.session_state.summary_output_path)
batches_output_path = Path(st.session_state.batches_output_path)
params = st.session_state.params

# --- show cached parameters ---
if params is not None:
    st.info(
        f"**Cached pipeline parameters**  \n"
        f"- Data source: `{params['data_source']}`  \n"
        f"- Folder: `{params['folder_path']}`  \n"
        f"- Samples per batch: `{params['samples_per_batch']}`  \n"
        f"- Batch structure: `{params['batch_structure']}`  \n"
        f"- QC structure: `{params['qc_structure']}`  \n"
        f"- QC labels: `{params['qc_samples']}`  \n"
        f"- Family column (source): `{params['family_col_source']}`  \n"
        f"- Genus column (source): `{params['genus_col_source']}`  \n"
        f"- Species column (source): `{params['species_col_source']}`"
    )

# --- show clean table info / path ---
st.subheader("Data loading and merging (cached)")
st.markdown(
    f"""
- **cenabiodb_merged2_clean** shape: `{cenabiodb_merged2_clean.shape}`  
- Clean table saved to: `{clean_csv_path}`
"""
)

# Taxonomic distributions
st.subheader("Taxonomic distributions")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Number of samples per family**")
    plot_family_bar(cenabiodb_merged2_clean)

with col2:
    st.markdown("**Number of samples per genus**")
    plot_genus_bar(cenabiodb_merged2_clean)

st.markdown("---")
st.markdown("**Global Family–Genus–Species sunburst**")
plot_global_sunburst(cenabiodb_merged2_clean)

# summary already computed
st.subheader("Family/Genus composition per batch (cached)")
st.markdown(
    f"Summary CSV saved to: `{summary_output_path / 'resumo_familia_genero_por_batch.csv'}`"
)
with st.expander("Show Summary CSV ", expanded=False):
    st.dataframe(summary_df.head(100))

summary_csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download summary CSV",
    data=summary_csv,
    file_name="resumo_familia_genero_por_batch.csv",
    mime="text/csv",
)

# -----------------------------
# 5.5 Per-batch sunburst plots
# -----------------------------
st.subheader("Per-batch sunburst explorer")

if batches is None or len(batches) == 0:
    st.info("No batches available. Run / recompute the pipeline first.")
else:
    batch_names = [
        df_b["Batch"].iloc[0] for df_b in batches if "Batch" in df_b.columns
    ]
    if not batch_names:
        st.info("No batch names found.")
    else:
        selected_batch_name = st.selectbox("Select a batch", batch_names)
        selected_batch_df = None
        for df_b in batches:
            if "Batch" in df_b.columns and df_b["Batch"].iloc[0] == selected_batch_name:
                selected_batch_df = df_b
                break

        if selected_batch_df is not None:
            plot_batch_sunburst(selected_batch_df, selected_batch_name)
            with st.expander("Show full batch table", expanded=False):
                st.dataframe(selected_batch_df)
        else:
            st.warning("Could not find the selected batch in memory.")

st.markdown(
    """
    Developed by **Ricardo M Borges** and **LAABio-IPPN-UFRJ**  
    contact: ricardo_mborges@yahoo.com.br  

    🔗 Details: [GitHub repository](https://github.com/RicardoMBorges/DBsimilarity_st)

    Check also: [DAFdiscovery](https://dafdiscovery.streamlit.app/)
    
    Check also: [TLC2Chrom](https://tlc2chrom.streamlit.app/)
    """
)