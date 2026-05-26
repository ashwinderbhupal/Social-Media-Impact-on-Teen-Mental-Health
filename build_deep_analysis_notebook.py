"""Generate teen_mental_health_deep_analysis.ipynb."""
import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "teen_mental_health_deep_analysis.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": source.splitlines(keepends=True), "outputs": [], "execution_count": None}


cells = []

cells.append(md(
    "# Social Media Impact on Teen Mental Health — Deep Analysis\n\n"
    "This notebook analyzes **13 features** from `teen_mental_health.csv` to explore how "
    "social media use, sleep, academics, and psychosocial factors relate to **depression_label** "
    "(main target). Run all cells top-to-bottom for the full pipeline."
))

# SECTION 0
cells.append(md("## Section 0 — Setup\n\nImport libraries and load the dataset."))
cells.append(code(
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import plotly.express as px\n"
    "import plotly.graph_objects as go  # noqa: F401\n"
    "from scipy import stats\n"
    "from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler\n"
    "from sklearn.model_selection import train_test_split, cross_val_score\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # noqa: F401\n"
    "from sklearn.svm import SVC\n"
    "from sklearn.metrics import (\n"
    "    classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score\n"
    ")\n"
    "from xgboost import XGBClassifier\n"
    "import shap\n"
    "from IPython.display import display\n\n"
    "sns.set_theme(style='whitegrid', palette='muted')\n"
    "plt.rcParams['figure.figsize'] = (10, 6)\n"
    "RANDOM_STATE = 42\n\n"
    "COLUMNS = [\n"
    "    'age', 'gender', 'daily_social_media_hours', 'platform_usage', 'sleep_hours',\n"
    "    'screen_time_before_sleep', 'academic_performance', 'physical_activity',\n"
    "    'social_interaction_level', 'stress_level', 'anxiety_level', 'addiction_level',\n"
    "    'depression_label'\n"
    "]\n"
    "NUMERIC_COLS = [\n"
    "    'age', 'daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep',\n"
    "    'academic_performance', 'physical_activity', 'social_interaction_level',\n"
    "    'stress_level', 'anxiety_level', 'addiction_level'\n"
    "]\n"
    "CAT_COLS = ['gender', 'platform_usage', 'social_interaction_level', 'depression_label']\n\n"
    "def numeric_col(name):\n"
    "  return 'social_interaction_level_encoded' if name == 'social_interaction_level' else name\n"
))
cells.append(code(
    "df = pd.read_csv('data/teen_mental_health.csv')\n"
    "df = df[COLUMNS].copy()\n\n"
    "print('Shape:', df.shape)\n"
    "print('\\nDtypes:')\n"
    "print(df.dtypes)\n"
    "print('\\nHead (5):')\n"
    "display(df.head(5))\n"
    "print('\\nDescribe:')\n"
    "display(df.describe(include='all'))"
))

# SECTION 1
cells.append(md("## Section 1 — Data Cleaning & Preprocessing\n\nCheck quality, remove duplicates, and encode categorical fields."))
cells.append(code(
    "missing = df.isnull().sum()\n"
    "print('Missing values per column:')\n"
    "print(missing)\n\n"
    "plt.figure(figsize=(10, 4))\n"
    "sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')\n"
    "plt.title('Missing Values Heatmap')\n"
    "plt.xlabel('Columns')\n"
    "plt.ylabel('Rows')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "n_before = len(df)\n"
    "df = df.drop_duplicates()\n"
    "removed = n_before - len(df)\n"
    "print(f'Duplicate rows removed: {removed}')"
))
cells.append(code(
    "for col in ['gender', 'platform_usage', 'depression_label']:\n"
    "    print(f'\\n{col} value_counts:')\n"
    "    print(df[col].value_counts())"
))
cells.append(code(
    "# Map binary depression to readable labels, then encode target\n"
    "dep_map = {0: 'No Depression', 1: 'Depression'}\n"
    "if df['depression_label'].dtype != object:\n"
    "    df['depression_label'] = df['depression_label'].map(dep_map)\n\n"
    "le_dep = LabelEncoder()\n"
    "df['depression_label_encoded'] = le_dep.fit_transform(df['depression_label'])\n"
    "print('Depression encoding:', dict(zip(le_dep.classes_, le_dep.transform(le_dep.classes_))))\n\n"
    "le_gender = LabelEncoder()\n"
    "df['gender_encoded'] = le_gender.fit_transform(df['gender'])\n"
    "le_platform = LabelEncoder()\n"
    "df['platform_usage_encoded'] = le_platform.fit_transform(df['platform_usage'])\n"
    "le_social = LabelEncoder()\n"
    "df['social_interaction_level_encoded'] = le_social.fit_transform(df['social_interaction_level'])\n"
    "print('Gender encoding:', dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))\n"
    "print('Platform encoding:', dict(zip(le_platform.classes_, le_platform.transform(le_platform.classes_))))"
))

# SECTION 2
cells.append(md("## Section 2 — Exploratory Data Analysis (EDA)\n\nVisualize distributions, categories, correlations, and group differences."))
cells.append(code(
    "fig, axes = plt.subplots(2, 5, figsize=(20, 8))\n"
    "axes = axes.ravel()\n"
    "plot_cols = NUMERIC_COLS + ['social_interaction_level']\n"
    "plot_df = df.copy()\n"
    "plot_df['social_interaction_level'] = plot_df['social_interaction_level_encoded']\n"
    "for ax, col in zip(axes, plot_cols):\n"
    "    sns.histplot(plot_df[col], kde=True, ax=ax, color='steelblue')\n"
    "    ax.set_title(col)\n"
    "    ax.set_xlabel(col)\n"
    "    ax.set_ylabel('Count')\n"
    "plt.suptitle('Distributions of Numeric Features', y=1.02, fontsize=14)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
    "for ax, col in zip(axes, ['gender', 'platform_usage', 'depression_label']):\n"
    "    sns.countplot(data=df, x=col, ax=ax, hue=col, legend=False, palette='Set2')\n"
    "    ax.set_title(f'Countplot: {col}')\n"
    "    ax.set_xlabel(col)\n"
    "    ax.set_ylabel('Count')\n"
    "    ax.tick_params(axis='x', rotation=20)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "corr_cols = [numeric_col(c) for c in NUMERIC_COLS] + ['depression_label_encoded']\n"
    "corr = df[corr_cols].corr(method='pearson')\n"
    "corr.index = NUMERIC_COLS + ['depression_label']\n"
    "corr.columns = NUMERIC_COLS + ['depression_label']\n"
    "plt.figure(figsize=(12, 10))\n"
    "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)\n"
    "plt.title('Pearson Correlation Heatmap (Numeric Features)')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "pair_cols = ['daily_social_media_hours', 'sleep_hours', 'stress_level', 'anxiety_level', 'addiction_level']\n"
    "sns.pairplot(df, vars=pair_cols, hue='depression_label', palette='Set1', diag_kind='kde', height=2.2)\n"
    "plt.suptitle('Pairplot by Depression Label', y=1.02)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "box_cols = ['stress_level', 'anxiety_level', 'addiction_level', 'sleep_hours', 'daily_social_media_hours']\n"
    "fig, axes = plt.subplots(2, 3, figsize=(16, 9))\n"
    "axes = axes.ravel()\n"
    "for ax, col in zip(axes, box_cols):\n"
    "    sns.boxplot(data=df, x='depression_label', y=col, ax=ax, palette='pastel')\n"
    "    ax.set_title(f'{col} by Depression Label')\n"
    "    ax.set_xlabel('Depression Label')\n"
    "    ax.set_ylabel(col)\n"
    "axes[-1].axis('off')\n"
    "plt.suptitle('Numeric Features vs Depression Label', y=1.02)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# SECTION 3
cells.append(md("## Section 3 — Deep Statistical Analysis\n\nHypothesis tests and rank correlations with the encoded target."))
cells.append(code(
    "def iqr_outliers(series):\n"
    "    q1, q3 = series.quantile(0.25), series.quantile(0.75)\n"
    "    iqr = q3 - q1\n"
    "    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr\n"
    "    return ((series < low) | (series > high)).sum()\n\n"
    "outlier_counts = {c: iqr_outliers(df[numeric_col(c)]) for c in NUMERIC_COLS}\n"
    "print('IQR outlier counts:', outlier_counts)\n\n"
    "fig, axes = plt.subplots(2, 5, figsize=(20, 8))\n"
    "axes = axes.ravel()\n"
    "for ax, col in zip(axes, NUMERIC_COLS):\n"
    "    sns.boxplot(y=df[numeric_col(col)], ax=ax, color='skyblue')\n"
    "    ax.set_title(col)\n"
    "    ax.set_ylabel(col)\n"
    "plt.suptitle('IQR Outlier Detection — Boxplots', y=1.02)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

def ttest_cell(var_name, title):
    cells.append(code(
        f"dep = df['depression_label']\n"
        f"group0 = df.loc[dep == 'No Depression', '{var_name}']\n"
        f"group1 = df.loc[dep == 'Depression', '{var_name}']\n"
        f"t_stat, p_val = stats.ttest_ind(group0, group1, equal_var=False)\n"
        f"print('Variable: {var_name}')\n"
        f"print(f't-statistic: {{t_stat:.4f}}')\n"
        f"print(f'p-value: {{p_val:.4e}}')\n"
        f"print('Mean (No Depression):', group0.mean())\n"
        f"print('Mean (Depression):', group1.mean())"
    ))
    cells.append(md(
        f"**Interpretation ({title}):** "
        f"If p < 0.05, average `{var_name}` differs significantly between depressed and non-depressed teens; "
        f"compare the printed group means to see direction of the effect."
    ))

for v, t in [
    ('daily_social_media_hours', 'Social media hours'),
    ('sleep_hours', 'Sleep hours'),
    ('anxiety_level', 'Anxiety level'),
    ('addiction_level', 'Addiction level'),
]:
    ttest_cell(v, t)

cells.append(code(
    "groups = [df.loc[df['platform_usage'] == p, 'stress_level'] for p in df['platform_usage'].unique()]\n"
    "f_stat, p_val = stats.f_oneway(*groups)\n"
    "print('ANOVA: stress_level across platform_usage')\n"
    "print(f'F-statistic: {f_stat:.4f}')\n"
    "print(f'p-value: {p_val:.4e}')"
))
cells.append(md(
    "**Interpretation (ANOVA — stress by platform):** A significant p-value means mean stress_level "
    "differs across at least one pair of platform_usage groups (Instagram, TikTok, Both)."
))
cells.append(code(
    "groups = [df.loc[df['platform_usage'] == p, 'anxiety_level'] for p in df['platform_usage'].unique()]\n"
    "f_stat, p_val = stats.f_oneway(*groups)\n"
    "print('ANOVA: anxiety_level across platform_usage')\n"
    "print(f'F-statistic: {f_stat:.4f}')\n"
    "print(f'p-value: {p_val:.4e}')"
))
cells.append(md(
    "**Interpretation (ANOVA — anxiety by platform):** Significant results imply platform choice is "
    "associated with different average anxiety levels."
))
cells.append(code(
    "ct = pd.crosstab(df['gender'], df['depression_label'])\n"
    "chi2, p_val, dof, expected = stats.chi2_contingency(ct)\n"
    "print('Chi-square: gender vs depression_label')\n"
    "print(ct)\n"
    "print(f'chi2: {chi2:.4f}, p-value: {p_val:.4e}, dof: {dof}')"
))
cells.append(md(
    "**Interpretation (Chi-square):** If p < 0.05, depression_label distribution is not independent of gender."
))
cells.append(code(
    "spearman_cols = [\n"
    "    'daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep',\n"
    "    'stress_level', 'anxiety_level', 'addiction_level', 'academic_performance'\n"
    "]\n"
    "spearman_results = []\n"
    "for col in spearman_cols:\n"
    "    rho, p = stats.spearmanr(df['depression_label_encoded'], df[col])\n"
    "    spearman_results.append({'feature': col, 'spearman_rho': rho, 'p_value': p})\n"
    "spearman_df = pd.DataFrame(spearman_results).sort_values('spearman_rho', key=abs, ascending=False)\n"
    "display(spearman_df)"
))

# SECTION 4
cells.append(md("## Section 4 — Feature Engineering\n\nCreate derived features and prepare the modeling matrix."))
cells.append(code(
    "df['sleep_deficit'] = 8 - df['sleep_hours']\n"
    "df['screen_load'] = df['daily_social_media_hours'] + df['screen_time_before_sleep']\n\n"
    "scaler_mm = MinMaxScaler()\n"
    "mh = df[['stress_level', 'anxiety_level', 'addiction_level']]\n"
    "df['mental_health_score'] = scaler_mm.fit_transform(mh).sum(axis=1)\n\n"
    "df['usage_category'] = pd.cut(\n"
    "    df['daily_social_media_hours'],\n"
    "    bins=[-np.inf, 2, 4, np.inf],\n"
    "    labels=['Light', 'Moderate', 'Heavy']\n"
    ")\n"
    "df['age_group'] = pd.cut(\n"
    "    df['age'],\n"
    "    bins=[12, 15, 17, 19],\n"
    "    labels=['Early Teen (13-15)', 'Mid Teen (16-17)', 'Late Teen (18-19)']\n"
    ")\n"
    "display(df[['sleep_deficit', 'screen_load', 'mental_health_score', 'usage_category', 'age_group']].head())"
))
cells.append(code(
    "df_model = df.copy()\n"
    "ohe_cols = ['gender', 'platform_usage', 'usage_category', 'age_group']\n"
    "df_model = pd.get_dummies(df_model, columns=ohe_cols, drop_first=True)\n\n"
    "feature_exclude = {'depression_label', 'depression_label_encoded', 'social_interaction_level'}\n"
    "numeric_features = [\n"
    "    'age', 'daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep',\n"
    "    'academic_performance', 'physical_activity', 'stress_level', 'anxiety_level',\n"
    "    'addiction_level', 'sleep_deficit', 'screen_load', 'mental_health_score',\n"
    "    'gender_encoded', 'platform_usage_encoded', 'social_interaction_level_encoded'\n"
    "]\n"
    "ohe_feature_cols = [c for c in df_model.columns if c not in feature_exclude and c not in numeric_features]\n"
    "X_cols = [c for c in numeric_features + ohe_feature_cols if c in df_model.columns]\n\n"
    "scaler = StandardScaler()\n"
    "df_model[X_cols] = scaler.fit_transform(df_model[X_cols])\n"
    "print('Final feature matrix shape (rows, features):', df_model[X_cols].shape)"
))

# SECTION 5
cells.append(md("## Section 5 — Advanced Visualizations\n\nInteractive and multivariate views of relationships."))
cells.append(code(
    "fig = px.scatter(\n"
    "    df, x='daily_social_media_hours', y='anxiety_level', color='depression_label',\n"
    "    size='addiction_level', hover_data=['platform_usage', 'sleep_hours'],\n"
    "    title='Social Media Hours vs Anxiety (by Depression Label)',\n"
    "    labels={'daily_social_media_hours': 'Daily Social Media (h)', 'anxiety_level': 'Anxiety Level'}\n"
    ")\n"
    "fig.update_layout(legend_title_text='Depression Label')\n"
    "fig.show()"
))
cells.append(code(
    "melted = df.melt(id_vars='platform_usage', value_vars=['stress_level', 'anxiety_level'],\n"
    "                 var_name='metric', value_name='score')\n"
    "fig = px.box(melted, x='platform_usage', y='score', color='metric',\n"
    "             title='Stress & Anxiety by Platform Usage')\n"
    "fig.update_layout(xaxis_title='Platform', yaxis_title='Score', legend_title='Metric')\n"
    "fig.show()"
))
cells.append(code(
    "plt.figure(figsize=(8, 5))\n"
    "sns.violinplot(data=df, x='depression_label', y='sleep_hours', palette='muted', inner='quartile')\n"
    "plt.title('Sleep Hours by Depression Label')\n"
    "plt.xlabel('Depression Label')\n"
    "plt.ylabel('Sleep Hours')\n"
    "plt.legend([], [], frameon=False)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "plt.figure(figsize=(8, 5))\n"
    "sns.violinplot(data=df, x='depression_label', y='addiction_level', palette='muted', inner='quartile')\n"
    "plt.title('Addiction Level by Depression Label')\n"
    "plt.xlabel('Depression Label')\n"
    "plt.ylabel('Addiction Level')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "pivot_anx = df.pivot_table(values='anxiety_level', index='age_group', columns='usage_category', aggfunc='mean')\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.heatmap(pivot_anx, annot=True, fmt='.2f', cmap='YlOrRd')\n"
    "plt.title('Mean Anxiety by Age Group vs Usage Category')\n"
    "plt.xlabel('Usage Category')\n"
    "plt.ylabel('Age Group')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "pivot_stress = df.pivot_table(values='stress_level', index='gender', columns='platform_usage', aggfunc='mean')\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.heatmap(pivot_stress, annot=True, fmt='.2f', cmap='Blues')\n"
    "plt.title('Mean Stress by Gender vs Platform')\n"
    "plt.xlabel('Platform Usage')\n"
    "plt.ylabel('Gender')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "ct_gender = pd.crosstab(df['gender'], df['depression_label'], normalize='index')\n"
    "ct_gender.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='Set2')\n"
    "plt.title('Depression Label Distribution by Gender')\n"
    "plt.xlabel('Gender')\n"
    "plt.ylabel('Proportion')\n"
    "plt.legend(title='Depression Label')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "ct_usage = pd.crosstab(df['usage_category'], df['depression_label'], normalize='index')\n"
    "ct_usage.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='Set3')\n"
    "plt.title('Depression Label Distribution by Usage Category')\n"
    "plt.xlabel('Usage Category')\n"
    "plt.ylabel('Proportion')\n"
    "plt.legend(title='Depression Label')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "line_df = df.groupby('usage_category')[['stress_level', 'anxiety_level', 'addiction_level']].mean().reset_index()\n"
    "plt.figure(figsize=(9, 5))\n"
    "for col in ['stress_level', 'anxiety_level', 'addiction_level']:\n"
    "    plt.plot(line_df['usage_category'].astype(str), line_df[col], marker='o', label=col)\n"
    "plt.title('Mean Stress, Anxiety & Addiction across Usage Category')\n"
    "plt.xlabel('Usage Category')\n"
    "plt.ylabel('Mean Score')\n"
    "plt.legend(title='Metric')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "plt.figure(figsize=(8, 5))\n"
    "sns.lmplot(data=df, x='screen_time_before_sleep', y='sleep_hours', hue='depression_label',\n"
    "           height=5, aspect=1.4, scatter_kws={'alpha': 0.5})\n"
    "plt.title('Screen Time Before Sleep vs Sleep Hours')\n"
    "plt.xlabel('Screen Time Before Sleep (h)')\n"
    "plt.ylabel('Sleep Hours')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "plt.figure(figsize=(8, 5))\n"
    "sns.lmplot(data=df, x='academic_performance', y='mental_health_score', hue='depression_label',\n"
    "           height=5, aspect=1.4, scatter_kws={'alpha': 0.5})\n"
    "plt.title('Academic Performance vs Mental Health Score')\n"
    "plt.xlabel('Academic Performance')\n"
    "plt.ylabel('Mental Health Score')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# SECTION 6
cells.append(md("## Section 6 — Machine Learning\n\nTrain four classifiers and compare performance on the encoded target."))
cells.append(code(
    "X = df_model[X_cols]\n"
    "y = df_model['depression_label_encoded']\n"
    "X_train, X_test, y_train, y_test = train_test_split(\n"
    "    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y\n"
    ")\n"
    "print('Train shape:', X_train.shape, '| Test shape:', X_test.shape)"
))
cells.append(code(
    "models = {\n"
    "    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),\n"
    "    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),\n"
    "    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),\n"
    "    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)\n"
    "}\n"
    "results = []\n"
    "fitted = {}\n"
    "for name, model in models.items():\n"
    "    model.fit(X_train, y_train)\n"
    "    fitted[name] = model\n"
    "    y_pred = model.predict(X_test)\n"
    "    y_proba = model.predict_proba(X_test)\n"
    "    print('=' * 60)\n"
    "    print(name)\n"
    "    print(classification_report(y_test, y_pred, target_names=le_dep.classes_))\n"
    "    cm = confusion_matrix(y_test, y_pred)\n"
    "    plt.figure(figsize=(5, 4))\n"
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
    "                xticklabels=le_dep.classes_, yticklabels=le_dep.classes_)\n"
    "    plt.title(f'Confusion Matrix - {name}')\n"
    "    plt.xlabel('Predicted')\n"
    "    plt.ylabel('Actual')\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')\n"
    "    print(f'5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')\n"
    "    if len(le_dep.classes_) == 2:\n"
    "        roc = roc_auc_score(y_test, y_proba[:, 1])\n"
    "    else:\n"
    "        roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')\n"
    "    print(f'ROC-AUC (OvR, macro): {roc:.4f}')\n"
    "    results.append({\n"
    "        'Model': name,\n"
    "        'CV Accuracy': cv_scores.mean(),\n"
    "        'Test Accuracy': accuracy_score(y_test, y_pred),\n"
    "        'F1-Macro': f1_score(y_test, y_pred, average='macro'),\n"
    "        'ROC-AUC': roc\n"
    "    })\n\n"
    "comparison = pd.DataFrame(results).sort_values('F1-Macro', ascending=False)\n"
    "display(comparison)"
))
cells.append(code(
    "metrics = ['CV Accuracy', 'Test Accuracy', 'F1-Macro', 'ROC-AUC']\n"
    "x = np.arange(len(comparison))\n"
    "width = 0.2\n"
    "plt.figure(figsize=(12, 6))\n"
    "for i, metric in enumerate(metrics):\n"
    "    plt.bar(x + i * width, comparison[metric], width, label=metric)\n"
    "plt.xticks(x + width * 1.5, comparison['Model'], rotation=15)\n"
    "plt.title('Model Comparison')\n"
    "plt.xlabel('Model')\n"
    "plt.ylabel('Score')\n"
    "plt.legend(title='Metric')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# SECTION 7
cells.append(md("## Section 7 — Model Explainability (SHAP)\n\nExplain the best tree-based model with SHAP."))
cells.append(code(
    "best_name = comparison.iloc[0]['Model']\n"
    "if best_name not in ('Random Forest', 'XGBoost'):\n"
    "    best_name = 'Random Forest'\n"
    "best_model = fitted[best_name]\n"
    "print('SHAP model:', best_name)\n\n"
    "explainer = shap.TreeExplainer(best_model)\n"
    "shap_values = explainer.shap_values(X_test)\n"
    "if isinstance(shap_values, list):\n"
    "    shap_vals_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]\n"
    "else:\n"
    "    shap_vals_plot = shap_values"
))
cells.append(code(
    "plt.figure()\n"
    "shap.summary_plot(shap_vals_plot, X_test, plot_type='bar', show=False)\n"
    "plt.title('SHAP Summary — Mean |Impact|')\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "plt.figure()\n"
    "shap.summary_plot(shap_vals_plot, X_test, show=False)\n"
    "plt.title('SHAP Beeswarm')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code(
    "for feat in ['daily_social_media_hours', 'sleep_hours', 'anxiety_level']:\n"
    "    if feat in X_test.columns:\n"
    "        plt.figure()\n"
    "        shap.dependence_plot(feat, shap_vals_plot, X_test, show=False)\n"
    "        plt.title(f'SHAP Dependence — {feat}')\n"
    "        plt.tight_layout()\n"
    "        plt.show()"
))
cells.append(md(
    "### SHAP interpretation\n\n"
    "- **Bar plot:** Features with the largest mean |SHAP| value drive predictions most strongly.\n"
    "- **Beeswarm:** Red (high feature value) vs blue (low) shows whether higher values push toward depression.\n"
    "- **Dependence plots:** Curves for `daily_social_media_hours`, `sleep_hours`, and `anxiety_level` show "
    "non-linear thresholds where risk increases.\n"
    "- Expect **anxiety_level**, **addiction_level**, and **stress_level** (via mental_health_score) to rank "
    "high if they separate classes well in EDA."
))

# SECTION 8
cells.append(md(
    "## Section 8 — Key Insights & Conclusions\n\n"
    "### Data-driven findings (13 columns)\n"
    "1. **Target balance:** Depression prevalence and class balance should be checked from `depression_label` counts in Section 1.\n"
    "2. **Social media dose:** Mean `daily_social_media_hours` and `usage_category` (Light/Moderate/Heavy) relate to stress/anxiety trajectories in Section 5 line plots.\n"
    "3. **Sleep pathway:** `sleep_hours` and `screen_time_before_sleep` jointly shape rest; `sleep_deficit` captures sub-8-hour sleep.\n"
    "4. **Symptom cluster:** `stress_level`, `anxiety_level`, and `addiction_level` form the core mental-health burden (mental_health_score).\n"
    "5. **Platform effect:** ANOVA on `platform_usage` tests whether Instagram, TikTok, or Both differ in mean stress/anxiety.\n"
    "6. **Gender association:** Chi-square links `gender` with depression distribution.\n"
    "7. **Academics & activity:** `academic_performance` and `physical_activity` Spearman correlations show protective or risk associations with encoded depression.\n"
    "8. **Social interaction:** `social_interaction_level` (low/medium/high) adds social context alongside numeric predictors.\n\n"
    "### Recommendations\n"
    "- **Teens:** Cap heavy (>4h) `daily_social_media_hours`; reduce `screen_time_before_sleep`; prioritize 8+ `sleep_hours`.\n"
    "- **Parents:** Monitor `addiction_level` and evening screens; encourage `physical_activity` and offline social interaction.\n"
    "- **Schools:** Screen for high `stress_level`/`anxiety_level`; link counseling to academic support when `academic_performance` drops.\n"
    "- **Policymakers:** Fund platform-aware education; require age-appropriate design reducing night-time engagement.\n\n"
    "### Platform with highest risk\n"
    "See the computed table in the next cell — platform with highest mean `anxiety_level` and depression rate.\n\n"
    "### Best model\n"
    "The comparison table in Section 6 ranks models by **F1-Macro**; tree ensembles typically win on mixed numeric/categorical features after scaling and one-hot encoding.\n\n"
    "### Limitations\n"
    "- Cross-sectional data — causality cannot be inferred.\n"
    "- `depression_label` is binary in this file (not Mild/Severe tiers).\n"
    "- Self-reported hours and scales may include bias.\n"
    "- Suggested improvements: longitudinal tracking, clinical validation, more platforms, and larger sample."
))
cells.append(code(
    "platform_risk = df.groupby('platform_usage').agg(\n"
    "    depression_rate=('depression_label', lambda s: (s == 'Depression').mean()),\n"
    "    mean_anxiety=('anxiety_level', 'mean'),\n"
    "    mean_stress=('stress_level', 'mean')\n"
    ").sort_values('depression_rate', ascending=False)\n"
    "print('Platform risk summary:')\n"
    "display(platform_risk)\n"
    "print('\\nBest model (by F1-Macro):', comparison.iloc[0]['Model'])\n"
    "print(comparison.iloc[0].to_string())"
))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH} ({len(cells)} cells)")
