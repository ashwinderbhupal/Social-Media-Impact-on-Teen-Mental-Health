# 📱 Social Media Impact on Teen Mental Health — Deep Analysis

> A complete end-to-end R Markdown project analyzing how social media habits,
> sleep patterns, academic performance, and psychosocial factors relate to
> **depression** in teenagers.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Analysis Sections](#-analysis-sections)
- [Machine Learning Models](#-machine-learning-models)
- [Key Findings](#-key-findings)
- [Known Issues & Fixes](#-known-issues--fixes)
- [Output Files](#-output-files)
- [Limitations](#-limitations)

---

## 🔍 Project Overview

This project performs a **comprehensive data science analysis** on teenager mental
health data, covering:

- Exploratory Data Analysis (EDA) with 15+ visualizations
- Deep statistical testing (T-tests, ANOVA, Chi-square, Spearman)
- Feature engineering (sleep deficit, screen load, mental health score)
- 4 machine learning classifiers with cross-validation
- Model explainability via SHAP values and variable importance
- Actionable recommendations for teens, parents, schools, and policymakers

The entire analysis is contained in a **single self-knitting R Markdown file**
that produces a polished, interactive HTML report.

---

## 📊 Dataset

**Source:** [Kaggle — Social Media Impact on Teen Mental Health](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy/data)

**File name expected:** `data/teen_mental_health.csv`

### Columns (13 total)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `age` | Numeric | Teen's age (typically 13–19) |
| 2 | `gender` | Categorical | Male / Female / Other |
| 3 | `daily_social_media_hours` | Numeric | Hours spent on social media per day |
| 4 | `platform_usage` | Categorical | Primary platform (Instagram / TikTok / Both / etc.) |
| 5 | `sleep_hours` | Numeric | Average nightly sleep hours |
| 6 | `screen_time_before_sleep` | Numeric | Hours of screen use immediately before bed |
| 7 | `academic_performance` | Numeric | Academic performance score |
| 8 | `physical_activity` | Numeric | Physical activity hours per week |
| 9 | `social_interaction_level` | Numeric / Categorical | Level of offline social interaction (Low/Medium/High or 0/1/2) |
| 10 | `stress_level` | Numeric | Self-reported stress score |
| 11 | `anxiety_level` | Numeric | Self-reported anxiety score |
| 12 | `addiction_level` | Numeric | Social media addiction score |
| 13 | `depression_label` | Binary Target | 0 = No Depression / 1 = Depression |

---

## 📁 Project Structure

```
project/
│
├── data/
│   └── teen_mental_health.csv          ← Place your dataset here
│
├── teen_mental_health_deep_analysis.Rmd  ← Main analysis file (knit this)
├── README.md                             ← This file
│
└── output/                             ← Auto-generated after knitting
    └── teen_mental_health_deep_analysis.html
```

> ⚠️ **Important:** The CSV must be placed inside a `data/` subfolder
> in the same directory as the `.Rmd` file. The file must be named
> exactly `teen_mental_health.csv`.

---

## ⚙️ Requirements

### R Version
- **R ≥ 4.1.0** recommended
- **RStudio ≥ 2022.07** recommended (for knitting support)

### R Packages

The `.Rmd` file **auto-installs** all missing packages on first run.
Here is the full list:

| Package | Purpose |
|---------|---------|
| `tidyverse` | Data wrangling and ggplot2 visualizations |
| `scales` | Axis formatting (percent, comma) |
| `corrplot` | Correlation heatmap |
| `patchwork` | Combining multiple ggplot panels |
| `plotly` | Interactive charts |
| `caret` | Train/test split, cross-validation folds |
| `randomForest` | Random Forest classifier |
| `xgboost` | XGBoost classifier |
| `e1071` | Support Vector Machine |
| `glmnet` | Logistic Regression (Ridge/Lasso) |
| `pROC` | ROC curves and AUC |
| `vip` | Variable importance plots |
| `GGally` | Pairplot (ggpairs) |
| `ggridges` | Ridge density plots |
| `viridis` | Colour palettes |
| `kableExtra` | Styled tables in HTML output |
| `DT` | Interactive data tables |
| `gridExtra` | Grid layout for base plots |

---

## 🚀 Installation & Setup

### Step 1 — Download the Dataset

1. Go to: https://www.kaggle.com/datasets/algozee/teenager-menthal-healy/data
2. Click **Download** (you need a free Kaggle account)
3. Extract the ZIP and locate the CSV file
4. Rename it to `teen_mental_health.csv` if it has a different name

### Step 2 — Set Up the Project Folder

```
mkdir project
cd project
mkdir data
# Copy teen_mental_health.csv into the data/ folder
```

### Step 3 — Open in RStudio

1. Open RStudio
2. Go to **File → Open File**
3. Select `teen_mental_health_deep_analysis.Rmd`
4. RStudio will detect the working directory automatically

### Step 4 — Install R (if not installed)

Download from: https://cran.r-project.org/

Install RStudio from: https://posit.co/download/rstudio-desktop/

---

## ▶️ How to Run

### Option A — RStudio (Recommended)

1. Open `teen_mental_health_deep_analysis.Rmd` in RStudio
2. Press **`Ctrl + Shift + K`** (Windows/Linux) or **`Cmd + Shift + K`** (Mac)
3. Wait for knitting to complete (~2–5 minutes depending on machine)
4. The HTML report opens automatically in your browser

### Option B — R Console

```r
rmarkdown::render("teen_mental_health_deep_analysis.Rmd")
```

### Option C — Run Sections Individually

Click **Run → Run All** in RStudio, or run each chunk with
**`Ctrl + Shift + Enter`** to step through section by section.

---

## 📂 Analysis Sections

### Section 0 — Setup
- Auto-installs and loads all 17 required packages
- Defines column lists, global ggplot theme, random seed
- Loads and validates the CSV (checks all 13 columns exist)
- Prints shape, data types, first 10 rows, and descriptive statistics

### Section 1 — Data Cleaning & Preprocessing
- **Missing value heatmap** — visualizes which rows/columns have NAs
- **Auto-imputation** — numeric columns → median, categorical → mode
- **Duplicate removal** — reports how many rows were dropped
- **Target encoding** — normalizes `depression_label` from any format
  (0/1, yes/no, true/false, strings) into a clean `No Depression / Depression` factor
- **Categorical encoding** — label-encodes gender, platform_usage,
  social_interaction_level for numeric analysis

### Section 2 — Exploratory Data Analysis
- **2×5 distribution grid** — histogram + KDE for all 10 numeric features
- **Countplots** — gender, platform_usage, depression_label with counts
- **Pearson correlation heatmap** — hierarchically clustered, annotated
- **Pairplot** — 5 key mental health variables colored by depression label
- **Boxplots** — 8 features split by depression label with jittered points
- **Ridge density plots** — overlapping distributions reveal class separation

### Section 3 — Deep Statistical Analysis
- **IQR outlier detection** — count and visualize outliers per feature
- **8 Welch T-tests** — depressed vs non-depressed for every key feature,
  with significance table and -log10(p) bar chart
- **5 ANOVA tests** — mental health outcomes by platform_usage,
  with Tukey HSD post-hoc test for stress_level
- **Chi-square test** — gender × depression_label independence test
  with proportion bar chart
- **9 Spearman correlations** — all features vs encoded depression label,
  sorted by |ρ| with direction bar chart

### Section 4 — Feature Engineering
Creates 5 new features:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `sleep_deficit` | `8 - sleep_hours` | Measures under-sleeping |
| `screen_load` | `daily_social_media_hours + screen_time_before_sleep` | Total screen burden |
| `mental_health_score` | MinMax-normalised sum of stress + anxiety + addiction | Composite mental burden |
| `usage_category` | Bins: Light (<2h) / Moderate (2-4h) / Heavy (>4h) | Categorical screen time |
| `age_group` | Bins: Early (13-15) / Mid (16-17) / Late (18-19) Teen | Age cohort |

Also builds the final **scaled model matrix** (17 features) with NA imputation.

### Section 5 — Advanced Visualizations
- **Interactive Plotly scatter** — social media hours vs anxiety, sized by addiction
- **Interactive Plotly box** — stress and anxiety grouped by platform
- **3 Violin plots** — sleep, addiction, stress by depression label
- **2 Pivot heatmaps** — anxiety by age × usage, stress by gender × platform
- **2 Stacked bar charts** — depression proportion by gender and usage category
- **Line plot** — mean stress/anxiety/addiction trend across usage bins
- **2 Regression scatters** — screen time vs sleep, academic vs mental health score

### Section 6 — Machine Learning
Trains and evaluates 4 classifiers on an 80/20 stratified split:

See [Machine Learning Models](#-machine-learning-models) section below.

### Section 7 — Model Explainability
- **Random Forest importance** — MeanDecreaseAccuracy bar chart
- **XGBoost importance** — Gain bar chart + `xgb.ggplot.importance`
- **SHAP mean absolute importance** — global feature ranking
- **SHAP beeswarm plot** — direction and magnitude for every test observation
- **SHAP dependence plots** — functional relationship for top 2 features

### Section 8 — Key Insights & Conclusions
- Platform risk summary table (depression rate + all mental health metrics per platform)
- Age group breakdown table
- Final best model summary
- 8 numbered data-driven findings
- Recommendations for teens, parents, schools, policymakers
- Dataset limitations and future work suggestions

---

## 🤖 Machine Learning Models

All models use the same 17-feature scaled matrix and 80/20 stratified split.

| Model | Implementation | Key Hyperparameters |
|-------|---------------|-------------------|
| Logistic Regression | `glmnet::cv.glmnet` | Ridge (α=0), 5-fold CV lambda tuning |
| Random Forest | `randomForest` | 200 trees, importance=TRUE |
| XGBoost | `xgboost::xgb.train` | η=0.1, max_depth=6, subsample=0.8, 150 rounds |
| SVM (RBF) | `e1071::svm` | Radial kernel, probability=TRUE |

### Evaluation Metrics per Model
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap
- ROC curve (all 4 on one chart with AUC in legend)
- 5-fold cross-validation accuracy (mean ± std) — RF and XGBoost
- Macro F1 score (custom implementation, no package dependency)
- ROC-AUC score

### Custom Macro F1 Implementation
The notebook uses a custom `macro_f1()` function instead of `MLmetrics::F1_Score()`
to avoid the `unused argument (average = "macro")` error:

```r
macro_f1 <- function(pred, actual) {
  lvls <- levels(actual)
  f1s  <- vapply(lvls, function(lv) {
    tp <- sum(pred == lv & actual == lv)
    fp <- sum(pred == lv & actual != lv)
    fn <- sum(pred != lv & actual == lv)
    if ((tp + fp) == 0 || (tp + fn) == 0) return(0)
    prec <- tp / (tp + fp)
    rec  <- tp / (tp + fn)
    if ((prec + rec) == 0) return(0)
    2 * prec * rec / (prec + rec)
  }, numeric(1))
  mean(f1s)
}
```

---

## 🔑 Key Findings

1. **Dose-response relationship** — stress, anxiety, and addiction scores rise
   consistently from Light → Moderate → Heavy social media users.

2. **Sleep pathway** — screen time before sleep negatively correlates with
   sleep hours; sleep deficit is one of the strongest depression predictors.

3. **Symptom cluster** — stress, anxiety, and addiction are highly
   inter-correlated and together form the most discriminative feature set.

4. **Platform differences** — ANOVA reveals significant variation in mental
   health outcomes across different platforms.

5. **Academic performance** — negatively correlates with depression; students
   with higher scores show lower depression rates.

6. **Physical activity buffer** — negative Spearman correlation with depression
   suggests physical activity is a protective factor.

7. **Gender association** — chi-square test reveals whether depression is
   distributed differently by gender (see report for specific p-value).

8. **Composite score validity** — the engineered `mental_health_score` ranks
   among top SHAP features, validating the multi-symptom approach.

---

## 🐛 Known Issues & Fixes

### Error 1 — `F1_Score() unused argument (average = "macro")`
**Cause:** `MLmetrics` package does not support `average` parameter.
**Fix:** The notebook uses a custom `macro_f1()` function — no external
dependency needed. This is already implemented in the current version.

---

### Error 2 — `cor(): no complete element pairs`
**Cause:** Hidden factor/character columns or fully-NA columns after
`transmute()` break `cor(..., use = "complete.obs")`.

**Fix:** Replace the correlation data preparation with:
```r
corr_data <- corr_data |>
  mutate(across(everything(),
    ~ suppressWarnings(as.numeric(as.character(.))))) |>
  select(where(~ sum(!is.na(.)) > 1))

corr_mat <- tryCatch(
  cor(corr_data, use = "complete.obs",   method = "pearson"),
  error = function(e)
  cor(corr_data, use = "pairwise.complete.obs", method = "pearson")
)
```

---

### Error 3 — `glmnet(): x has missing values`
**Cause:** Factor columns (usage_cat_enc, age_group_enc) coerce to NA
when passed to `as.matrix()`, and `scale()` silently preserves them.

**Fix:** Force numeric coercion and median-impute before scaling:
```r
X_raw <- df_enc |>
  select(all_of(feature_cols)) |>
  mutate(across(everything(),
    ~ suppressWarnings(as.numeric(as.character(.))))) |>
  as.matrix()

for (j in seq_len(ncol(X_raw))) {
  bad <- !is.finite(X_raw[, j])
  if (any(bad)) X_raw[bad, j] <- median(X_raw[!bad, j], na.rm = TRUE)
}
X_scaled <- scale(X_raw)
```

---

### Error 4 — CSV file not found
**Cause:** Dataset placed in wrong directory or wrong filename.
**Fix:** Ensure the file is at `data/teen_mental_health.csv` relative
to the `.Rmd` file. Check with:
```r
file.exists("data/teen_mental_health.csv")  # must return TRUE
```

---

### Error 5 — Package not found
**Cause:** Auto-install failed (network issue or permission error).
**Fix:** Manually install:
```r
install.packages(c(
  "tidyverse", "scales", "corrplot", "patchwork", "plotly",
  "caret", "randomForest", "xgboost", "e1071", "glmnet",
  "pROC", "vip", "GGally", "ggridges", "viridis",
  "kableExtra", "DT", "gridExtra"
))
```

---

## 📤 Output Files

After knitting, the following are generated:

| File | Description |
|------|-------------|
| `teen_mental_health_deep_analysis.html` | Full interactive HTML report |
| Inline plots | All 30+ visualizations embedded in the HTML |
| Interactive charts | Plotly scatter and box plots (zoomable, hoverable) |

---

## ⚠️ Limitations

| Limitation | Details |
|-----------|---------|
| Cross-sectional data | Cannot establish causal relationships |
| Self-reported measures | Stress, sleep, and usage hours may be biased |
| Binary target | `depression_label` (0/1) loses clinical severity nuance |
| Platform grouping | Ignores within-platform behavior differences |
| Sample size | Results depend on the size and representativeness of the dataset |
| No temporal data | Cannot track changes in mental health over time |

---

## 📜 License

This project is for educational and research purposes.
Dataset credit: [Kaggle — algozee](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy/data)

---

## 🙋 FAQ

**Q: Can I use a different CSV filename?**
A: Yes — change `csv_path <- "data/teen_mental_health.csv"` in Section 0
to your actual file path.

**Q: The HTML report is very large — is that normal?**
A: Yes. Interactive Plotly charts add ~2–4 MB to the HTML file. This is expected.

**Q: Can I run this on Posit Cloud / RStudio Cloud?**
A: Yes. Upload the `.Rmd` and the `data/` folder, then knit normally.
Note that XGBoost and Random Forest CV may be slower on free-tier instances.

**Q: Why does `social_interaction_level` need special handling?**
A: In some versions of the dataset it is stored as text (Low/Medium/High)
and in others as a numeric (0/1/2). The notebook handles both automatically.

**Q: The pairplot takes a long time — can I skip it?**
A: Yes. Set the chunk option to `eval=FALSE`:
````
```{r pairplot, eval=FALSE}
````

---

*Generated for the Social Media Impact on Teen Mental Health Deep Analysis Project*
