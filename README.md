# **Batch Organizer – Taxonomy-Driven Extract Collection Manager**

A Streamlit application for organizing biological collections into **acquisition batches** using a hierarchical strategy (Family → Genus → Species + QC samples).
This tool was created to support large biological collections, enabling streamlined HPLC/LC-MS acquisition planning with taxonomic awareness.

The app accepts **CSV or Excel metadata files**, allows you to select custom column names (e.g., *Family*, *Família*, *Gênero*), and provides rich visualizations including:

* ✔️ Barplots (samples per Family / Genus)
* ✔️ Global Sunburst (Family → Genus → Species)
* ✔️ Per-batch Sunburst explorers
* ✔️ Automatic batch generation with QC structure
* ✔️ Downloadable batch CSVs and summary tables

The app can be used locally or deployed directly on **Streamlit Cloud**.

---

## **Features**

### **Metadata Input**

* Upload **CSV** (`.csv`) or **Excel** (`.xlsx`, `.xls`) metadata files
* Automatic **delimiter detection** (`,` `;` `\t`)
* Select your own taxonomy headers:

  * Family (Família)
  * Genus (Gênero)
  * Species (Espécies)

### **Visualization**

* Interactive bar charts (Plotly)
* Global hierarchical sunburst
* Per-batch sunburst plots
* Batch composition summary

### **Batch Creation**

* Configurable:

  * Samples per batch
  * Block structure (e.g., 24,24,32)
  * QC labels (e.g., Blank, QC Inter-Batch, QC Intra-Batch)
  * QC block distribution
* Ensures families and genera stay together
* Automatically combines small families to fill batches
* Exports all batches to CSV

---

# **Installation**

### **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/batch-organizer.git
cd batch-organizer
```

### **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### **Install dependencies**

```bash
pip install -r requirements.txt
```

---

# **Running the App Locally**

```bash
streamlit run app.py
```

Your browser will open automatically at:

```
http://localhost:8501
```

---

# **Running Online (Streamlit Cloud)**

1. Push the repo to GitHub
2. Open: [https://share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select the repo, branch, and file: `app.py`
5. Deploy

No extra configuration needed—no pyarrow dependencies.

---

# **Input File Format**

You may upload **any CSV or Excel metadata file**, simple or complex.

### **Minimum required columns**

You must designate:

| Purpose | Example Column Names  |
| ------- | --------------------- |
| Family  | `Family`, `Família`   |
| Genus   | `Genus`, `Gênero`     |
| Species | `Species`, `Espécies` |

Everything else is optional and will be preserved in batches.

### **CSV Delimiters supported**

* `,` (comma)
* `;` (semicolon)
* `\t` (tab)

Automatic detection is built-in.

### **Example Minimal CSV**

```csv
SampleID;Family;Genus;Species;Location
A001;Myrtaceae;Eugenia;uniflora;RJ
A002;Fabaceae;Vachellia;farnesiana;MG
A003;Rubiaceae;Psychotria;nuda;RJ
```

---

# **Full Tutorial**

This tutorial walks you through the full workflow.

---

## **1️Upload Metadata**

1. In the sidebar (left), click **Upload CSV/Excel metadata**
2. Upload your `.csv` or `.xlsx` file
3. The app automatically detects delimiter and shows a preview

📌 *If your file uses semicolon (`;`), no problem—fully supported.*

---

## **2️Map Taxonomy Columns**

You will see dropdown menus:

* Select Family column
* Select Genus column
* Select Species column
* (Optional) map extra metadata columns

Choose from the column names of your uploaded file
(Works with Portuguese, English, or custom headers.)

---

## **3️Configure Batch Parameters**

In the sidebar:

### **Samples per batch**

Default: **80**

### **Batch structure**

Defines block sizes inside each batch
Example:

```
24,24,32
```

Total must equal *Samples per batch*.

### **QC structure**

How many QC injections follow each block
Example:

```
1,1,1
```

### **QC Labels**

```
Blank,QC_Inter_Batch,QC_Intra_Batch
```

---

## **4️Generate Batches**

Click:

### **Run / recompute pipeline**

The app will:

✔️ read metadata
✔️ clean table
✔️ visualize taxonomy
✔️ create hierarchical batches
✔️ generate batch summary
✔️ store results in memory

---

## **Explore Taxonomy**

### **Bar plot – Families**

Number of samples per family

### **Bar plot – Genera**

Number of samples per genus

### **Sunburst – Global**

Family → Genus → Species
Interactive and zoomable

---

## **Download Results**

### **a) Batch summary**

```
Download summary CSV
```

### **b) Individual batch tables**

Under **Per-batch sunburst explorer**, use:

* Select batch
* View table
* Download via Streamlit’s top-right menu

All batches are also saved locally when running offline.

---

# **Tips for Best Results**

* Ensure all taxonomy rows are filled
* Group rare families manually if needed
* Use stable sample identifiers: `SampleID`, `Reg Amostra`, etc.
* Keep QC labels short to avoid clutter in RT sequences

---

# **Troubleshooting**

### **❗“Error tokenizing data”**

Your CSV probably uses `;` instead of `,`.
The app now automatically detects delimiters.

### **❗ Column not found**

Check that your selected headers match exactly (case-insensitive).

### **❗ Batches splitting families**

Increase samples per batch
OR enable the option “combine small families” (soon).

---

# **Citation**

If you use this tool in research, please cite:
** To be added**

---

# **📄 License**

MIT License
You are free to use, modify, and distribute with attribution.
