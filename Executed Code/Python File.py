
import os
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor

try:
    from google.colab import files
    IN_COLAB = True
except Exception:
    IN_COLAB = False


# STEP 1 - STYLE SETTINGS


MAIN_BLUE = "#0B3D91"
MID_BLUE = "#1E88E5"
LIGHT_BLUE = "#EAF3FF"
DARK_TEXT = "#111827"
SOFT_GREY = "#F3F4F6"
GREEN = "#146C43"
LIGHT_GREEN = "#DFF5E1"
YELLOW = "#F2C94C"
RED = "#DC2626"
PURPLE = "#7C3AED"
ORANGE = "#F97316"
WHITE = "#FFFFFF"

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": DARK_TEXT,
    "xtick.color": DARK_TEXT,
    "ytick.color": DARK_TEXT,
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": MAIN_BLUE,
    "axes.titlesize": 18,
    "axes.labelsize": 12,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#CBD5E1",
})

OUTPUT_DIR = Path("mm702_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def money_formatter(value, pos=None):
    """Format large money values for chart axes."""
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    return f"${value/1_000:.0f}K"


def save_figure(filename):
    """Save current matplotlib figure in the output folder."""
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.show()


def section_title(title, subtitle=""):
    """Display a premium blue section heading in Google Colab."""
    html = f"""
    <div style='background:linear-gradient(90deg,{MAIN_BLUE},{MID_BLUE});
                padding:18px 22px;border-radius:14px;margin:22px 0 14px 0;
                color:white;box-shadow:0 8px 18px rgba(11,61,145,0.20);'>
        <div style='font-size:24px;font-weight:800;letter-spacing:0.3px;'>{title}</div>
        <div style='font-size:14px;margin-top:6px;opacity:0.95;'>{subtitle}</div>
    </div>
    """
    display(HTML(html))


def info_card(title, value, note="", color=MAIN_BLUE):
    """Display one styled information card."""
    html = f"""
    <div style='display:inline-block;width:240px;min-height:110px;background:white;
                border-left:7px solid {color};border-radius:14px;padding:16px;margin:8px;
                box-shadow:0 6px 18px rgba(15,23,42,0.12);vertical-align:top;'>
        <div style='font-size:13px;color:#475569;font-weight:700;text-transform:uppercase;'>{title}</div>
        <div style='font-size:26px;color:{DARK_TEXT};font-weight:900;margin-top:8px;'>{value}</div>
        <div style='font-size:12px;color:#64748B;margin-top:8px;'>{note}</div>
    </div>
    """
    display(HTML(html))


def display_table(df_table, title="Table", max_rows=20):
    """Display a neat table in Colab."""
    shown = df_table.head(max_rows).copy()
    html_table = shown.to_html(index=False, border=0, classes="styled-table")
    html = f"""
    <style>
    .styled-table {{
        border-collapse: collapse;
        margin: 12px 0 24px 0;
        font-size: 13px;
        min-width: 720px;
        border-radius: 12px 12px 0 0;
        overflow: hidden;
        box-shadow: 0 8px 22px rgba(15,23,42,0.10);
    }}
    .styled-table thead tr {{
        background-color: {MAIN_BLUE};
        color: #ffffff;
        text-align: left;
        font-weight: bold;
    }}
    .styled-table th, .styled-table td {{
        padding: 10px 13px;
        border-bottom: 1px solid #E5E7EB;
    }}
    .styled-table tbody tr:nth-of-type(even) {{background-color: #F8FAFC;}}
    .styled-table tbody tr:hover {{background-color: {LIGHT_BLUE};}}
    </style>
    <h3 style='color:{MAIN_BLUE};font-weight:800;margin-top:18px;'>{title}</h3>
    {html_table}
    """
    display(HTML(html))


def save_table_as_png(df_table, filename, title=""):
    """Save a dataframe as a clean PNG table for the report."""
    table = df_table.copy()
    rows = min(len(table), 22)
    fig_height = max(2.4, rows * 0.38 + 1.2)
    fig_width = 12
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", color=MAIN_BLUE, pad=14)
    table_plot = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(9)
    table_plot.scale(1, 1.35)
    for (row, col), cell in table_plot.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor(MAIN_BLUE)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 == 0 else "white")
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved table image: {path}")



# STEP 2 - PROJECT TITLE CARD


section_title(
    "MM702 House Price Prediction",
    "Regression task using King County housing data, machine learning models and 5-fold cross-validation"
)

info_card("Module", "MM702", "Data Mining and Knowledge Discovery", MAIN_BLUE)
info_card("Task Type", "Regression", "Target variable: price", GREEN)
info_card("Student ID", "26804296", "University of Brighton", PURPLE)


# STEP 3 - LOAD DATASET


section_title("Step 1 - Loading the Dataset", "Upload the King County house price dataset in CSV or Excel format")

if IN_COLAB:
    print("Please upload your dataset file. Example: Houses price dataset.csv")
    uploaded = files.upload()
    dataset_path = list(uploaded.keys())[0]
else:
    dataset_path = "Houses price dataset.csv"

if dataset_path.lower().endswith(".csv"):
    df = pd.read_csv(dataset_path)
elif dataset_path.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(dataset_path)
else:
    raise ValueError("Please upload a CSV or Excel file only.")

# Clean column names early
original_shape = df.shape
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Dataset loaded successfully")
print("Dataset path:", dataset_path)
print("Original shape:", original_shape)
print("Cleaned column names:", df.columns.tolist())

info_card("Rows", f"{df.shape[0]:,}", "Before preprocessing", MAIN_BLUE)
info_card("Columns", f"{df.shape[1]:,}", "Before preprocessing", GREEN)
info_card("Target", "price", "House sale price", ORANGE)

display_table(df.head(), "First 5 Rows of Dataset", max_rows=5)
save_table_as_png(df.head(), "img03p04_dataset_preview.png", "Dataset Preview: First Five Records")

# Save column structure table
column_structure = pd.DataFrame({
    "No.": range(1, len(df.columns) + 1),
    "Column Name": df.columns,
    "Data Type": [str(dtype) for dtype in df.dtypes]
})
display_table(column_structure, "Dataset Column Structure", max_rows=30)
save_table_as_png(column_structure, "img02p03_column_structure.png", "Dataset Column Structure")


# STEP 4 - DATASET CHECKS


section_title("Step 2 - Basic Dataset Checks", "Missing values, duplicate rows and descriptive statistics")

missing_summary = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values,
    "Missing Percentage (%)": (df.isnull().sum().values / len(df) * 100).round(3)
}).sort_values(by="Missing Values", ascending=False)

duplicate_count = df.duplicated().sum()

info_card("Missing Cells", f"{int(df.isnull().sum().sum()):,}", "Across full dataset", RED if df.isnull().sum().sum() > 0 else GREEN)
info_card("Duplicate Rows", f"{duplicate_count:,}", "Checked before cleaning", ORANGE if duplicate_count > 0 else GREEN)
info_card("Numerical Columns", f"{len(df.select_dtypes(include=[np.number]).columns):,}", "Used for statistics", MAIN_BLUE)

display_table(missing_summary, "Missing Values Summary", max_rows=30)
save_table_as_png(missing_summary, "img04p06_missing_values.png", "Missing Values Summary")

numeric_describe = df.describe(include=[np.number]).T.round(2).reset_index().rename(columns={"index": "Feature"})
display_table(numeric_describe, "Descriptive Statistics", max_rows=30)
save_table_as_png(numeric_describe, "img07p08_descriptive_statistics.png", "Descriptive Statistics Summary")

# Dataset loaded summary image
loaded_summary = pd.DataFrame({
    "Item": ["Dataset path", "Rows before preprocessing", "Columns before preprocessing", "Duplicate rows", "Total missing cells", "Target variable"],
    "Value": [dataset_path, f"{df.shape[0]:,}", f"{df.shape[1]:,}", f"{duplicate_count:,}", f"{int(df.isnull().sum().sum()):,}", "price"]
})
save_table_as_png(loaded_summary, "img01p02_dataset_loaded_summary.png", "Dataset Loaded Successfully Summary")


# STEP 5 - DATA CLEANING AND FEATURE ENGINEERING


section_title("Step 3 - Data Cleaning and Feature Engineering", "Creating useful features from sale date, year built and renovation year")

required_columns = ["date", "price", "yr_built", "yr_renovated"]
missing_required = [col for col in required_columns if col not in df.columns]
if missing_required:
    raise KeyError(f"Missing required columns: {missing_required}")

rows_before_cleaning = len(df)

df = df.drop_duplicates()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df[df["price"].notna()]
df = df[df["price"] > 0]

df["sale_year"] = df["date"].dt.year
df["sale_month"] = df["date"].dt.month
df["house_age"] = df["sale_year"] - df["yr_built"]
df = df[df["house_age"] >= 0]

df["renovated"] = np.where(df["yr_renovated"] > 0, 1, 0)
df["years_since_renovation"] = np.where(
    df["yr_renovated"] > 0,
    df["sale_year"] - df["yr_renovated"],
    0
)
df["years_since_renovation"] = df["years_since_renovation"].clip(lower=0)

rows_after_cleaning = len(df)
rows_removed = rows_before_cleaning - rows_after_cleaning

feature_engineering_summary = pd.DataFrame({
    "New Feature": ["sale_year", "sale_month", "house_age", "renovated", "years_since_renovation"],
    "Meaning": [
        "Year when the house was sold",
        "Month when the house was sold",
        "Sale year minus year built",
        "1 if the house was renovated, otherwise 0",
        "Years between sale and renovation, 0 if not renovated"
    ]
})

display_table(feature_engineering_summary, "Feature Engineering Summary", max_rows=10)
save_table_as_png(feature_engineering_summary, "img05p07_feature_engineering.png", "Feature Engineering Summary")

info_card("Rows Removed", f"{rows_removed:,}", "Duplicates or invalid records", ORANGE)
info_card("Final Rows", f"{df.shape[0]:,}", "After cleaning", GREEN)
info_card("Final Columns", f"{df.shape[1]:,}", "After feature engineering", MAIN_BLUE)


# STEP 6 - EXPLORATORY DATA ANALYSIS


section_title("Step 4 - Exploratory Data Analysis", "Visual checks for price distribution and important relationships")

price_data = df["price"].dropna()

info_card("Mean Price", f"${price_data.mean():,.0f}", "Average sale price", MAIN_BLUE)
info_card("Median Price", f"${price_data.median():,.0f}", "Middle sale price", GREEN)
info_card("Max Price", f"${price_data.max():,.0f}", "Highest sale price", PURPLE)

# Price distribution
plt.figure(figsize=(12, 7))
plt.hist(
    price_data,
    bins=55,
    color=YELLOW,
    edgecolor=DARK_TEXT,
    linewidth=0.7,
    label="House price distribution"
)
plt.axvline(price_data.mean(), color=RED, linestyle="--", linewidth=2.5, label=f"Mean: ${price_data.mean():,.0f}")
plt.axvline(price_data.median(), color=GREEN, linestyle="-", linewidth=2.5, label=f"Median: ${price_data.median():,.0f}")
plt.title("Distribution of House Prices")
plt.xlabel("House Price ($)", fontweight="bold")
plt.ylabel("Frequency = Number of Houses", fontweight="bold")
plt.xlim(0, min(price_data.max(), 3_000_000))
plt.gca().xaxis.set_major_formatter(FuncFormatter(money_formatter))
plt.grid(alpha=0.28, linestyle="--")
plt.legend()
save_figure("img08p09_price_distribution.png")

# Living area vs price
plot_data = df[["sqft_living", "price"]].dropna()
x = plot_data["sqft_living"]
y = plot_data["price"]
slope, intercept = np.polyfit(x, y, 1)
trend_line = slope * x + intercept
correlation_value = x.corr(y)

plt.figure(figsize=(12, 7))
plt.scatter(x, y, alpha=0.38, color=MID_BLUE, edgecolor="white", linewidth=0.35, s=45)
plt.plot(x, trend_line, color=RED, linewidth=3, label="Linear trend line")
plt.title("Living Area vs House Price")
plt.xlabel("Square Feet Living Area", fontweight="bold")
plt.ylabel("House Price ($)", fontweight="bold")
plt.gca().yaxis.set_major_formatter(FuncFormatter(money_formatter))
plt.grid(alpha=0.28, linestyle="--")
plt.legend()
plt.text(
    0.03, 0.96,
    f"Correlation: {correlation_value:.2f}\nR² value: {correlation_value ** 2:.2f}",
    transform=plt.gca().transAxes,
    va="top",
    fontsize=11,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E1")
)
save_figure("img09p09_sqft_living_vs_price.png")

# Grade vs price
plot_data = df[["grade", "price"]].dropna()
x = plot_data["grade"]
y = plot_data["price"]
slope, intercept = np.polyfit(x, y, 1)
trend_line = slope * x + intercept
correlation_value = x.corr(y)

plt.figure(figsize=(12, 7))
plt.scatter(x, y, alpha=0.38, color=PURPLE, edgecolor="white", linewidth=0.35, s=45)
plt.plot(x, trend_line, color=RED, linewidth=3, label="Linear trend line")
plt.title("Grade vs House Price")
plt.xlabel("Property Grade", fontweight="bold")
plt.ylabel("House Price ($)", fontweight="bold")
plt.gca().yaxis.set_major_formatter(FuncFormatter(money_formatter))
plt.grid(alpha=0.28, linestyle="--")
plt.legend()
plt.text(
    0.03, 0.96,
    f"Correlation: {correlation_value:.2f}\nR² value: {correlation_value ** 2:.2f}",
    transform=plt.gca().transAxes,
    va="top",
    fontsize=11,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E1")
)
save_figure("img10p10_grade_vs_price.png")

# Correlation table and graph
correlation = df.corr(numeric_only=True)["price"].sort_values(ascending=False)
correlation_table = correlation.reset_index()
correlation_table.columns = ["Feature", "Correlation with Price"]
correlation_table["Correlation with Price"] = correlation_table["Correlation with Price"].round(4)

display_table(correlation_table, "Correlation with House Price", max_rows=30)
correlation_table.to_csv(OUTPUT_DIR / "correlation_with_price.csv", index=False)
save_table_as_png(correlation_table, "img11p11_correlation_table.png", "Correlation Table")

plot_corr = correlation.drop("price", errors="ignore").sort_values()
plt.figure(figsize=(12, 9))
colors = [RED if v < 0 else MAIN_BLUE for v in plot_corr.values]
plt.barh(plot_corr.index, plot_corr.values, color=colors, edgecolor="white")
plt.axvline(0, color=DARK_TEXT, linewidth=1.2)
plt.title("Feature Correlation with House Price")
plt.xlabel("Correlation Coefficient", fontweight="bold")
plt.ylabel("Feature", fontweight="bold")
plt.grid(axis="x", alpha=0.28, linestyle="--")
for i, value in enumerate(plot_corr.values):
    label_position = value + 0.02 if value >= 0 else value - 0.02
    ha = "left" if value >= 0 else "right"
    plt.text(label_position, i, f"{value:.2f}", va="center", ha=ha, fontsize=9, fontweight="bold")
save_figure("img12p12_correlation_bar_chart.png")


# STEP 7 - FEATURE SELECTION


section_title("Step 5 - Feature and Target Selection", "Price is the target variable and property attributes are the inputs")

features = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long",
    "sqft_living15", "sqft_lot15", "sale_year", "sale_month",
    "house_age", "renovated", "years_since_renovation"
]

target = "price"
available_features = [feature for feature in features if feature in df.columns]
missing_features = [feature for feature in features if feature not in df.columns]

if missing_features:
    print("These selected features were not found and were removed:", missing_features)

X = df[available_features].copy()
y = df[target].copy()

feature_table = pd.DataFrame({
    "Feature Number": range(1, len(available_features) + 1),
    "Feature": available_features
})

display_table(feature_table, "Selected Model Features", max_rows=30)
feature_table.to_csv(OUTPUT_DIR / "selected_model_features.csv", index=False)
save_table_as_png(feature_table, "img14p14_selected_model_features.png", "Selected Model Features")

final_setup = pd.DataFrame({
    "Item": ["Final dataset rows", "Final dataset columns", "Input rows", "Selected features", "Target variable", "Problem type"],
    "Value": [f"{df.shape[0]:,}", f"{df.shape[1]:,}", f"{X.shape[0]:,}", f"{X.shape[1]:,}", target, "Regression"]
})
display_table(final_setup, "Final Machine Learning Setup", max_rows=10)
save_table_as_png(final_setup, "img06p07_final_dataset_setup.png", "Final Dataset Shape and Machine Learning Setup")


# STEP 8 - TRAIN TEST SPLIT


section_title("Step 6 - Train-Test Split", "80% of data is used for training and 20% is used for testing")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

split_summary = pd.DataFrame({
    "Dataset Part": ["Training data", "Testing data", "Total data"],
    "Rows": [f"{X_train.shape[0]:,}", f"{X_test.shape[0]:,}", f"{X.shape[0]:,}"],
    "Percentage": ["80%", "20%", "100%"],
    "Features": [X_train.shape[1], X_test.shape[1], X.shape[1]]
})

display_table(split_summary, "Train-Test Split Summary", max_rows=10)
save_table_as_png(split_summary, "img15p15_train_test_split.png", "Train-Test Split Summary")

train_missing = pd.DataFrame({
    "Part": ["Training features", "Testing features"],
    "Missing Values": [int(X_train.isnull().sum().sum()), int(X_test.isnull().sum().sum())]
})
display_table(train_missing, "Missing Values Before Model Imputation", max_rows=10)
save_table_as_png(train_missing, "img16p16_missing_values_training_testing.png", "Missing Values in Training and Testing Data")


# STEP 9 - PREPROCESSING PIPELINES


section_title("Step 7 - Model Setup", "Linear Regression, Decision Tree, Random Forest, Neural Network and Stacking Ensemble")

categorical_features = ["zipcode"] if "zipcode" in available_features else []
numeric_features = [col for col in available_features if col not in categorical_features]

try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_preprocessor_scaled = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_preprocessor_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", onehot)
])

preprocessor_scaled = ColumnTransformer(transformers=[
    ("num", numeric_preprocessor_scaled, numeric_features),
    ("cat", categorical_preprocessor, categorical_features)
], remainder="drop")

preprocessor_tree = ColumnTransformer(transformers=[
    ("num", numeric_preprocessor_tree, numeric_features),
    ("cat", categorical_preprocessor, categorical_features)
], remainder="drop")

model_setup = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest", "Neural Network", "Stacking Ensemble"],
    "Purpose": [
        "Simple baseline model",
        "Captures non-linear rules",
        "Strong tree-based ensemble",
        "Learns complex patterns using hidden layers",
        "Combines Random Forest and Neural Network with Ridge as meta-model"
    ]
})
display_table(model_setup, "Machine Learning Models Setup", max_rows=10)
save_table_as_png(model_setup, "img13p13_model_setup.png", "Machine Learning Models Setup")

models = {
    "Linear Regression": Pipeline(steps=[
        ("preprocessor", preprocessor_scaled),
        ("model", LinearRegression())
    ]),
    "Decision Tree Regressor": Pipeline(steps=[
        ("preprocessor", preprocessor_tree),
        ("model", DecisionTreeRegressor(max_depth=10, random_state=42))
    ]),
    "Random Forest Regressor": Pipeline(steps=[
        ("preprocessor", preprocessor_tree),
        ("model", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
    ]),
    "Neural Network Regressor": Pipeline(steps=[
        ("preprocessor", preprocessor_scaled),
        ("model", MLPRegressor(
            hidden_layer_sizes=(80, 40),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=350,
            random_state=42,
            early_stopping=True
        ))
    ]),
    "Stacking Ensemble Regressor": Pipeline(steps=[
        ("preprocessor", preprocessor_scaled),
        ("model", StackingRegressor(
            estimators=[
                ("forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                ("neural_network", MLPRegressor(
                    hidden_layer_sizes=(60, 30),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    max_iter=300,
                    random_state=42,
                    early_stopping=True
                ))
            ],
            final_estimator=Ridge(alpha=1.0),
            n_jobs=-1
        ))
    ])
}


# STEP 10 - MODEL TRAINING AND TEST EVALUATION


section_title("Step 8 - Model Training and Evaluation", "Models are evaluated using MAE, MSE, RMSE, R² and Adjusted R²")


def adjusted_r2_score(r2, n, p):
    if n <= p + 1:
        return np.nan
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

results = []
predictions = {}
training_times = {}

for model_name, model in models.items():
    print(f"\nTraining model: {model_name}")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred
    training_times[model_name] = training_time

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = adjusted_r2_score(r2, X_test.shape[0], X_test.shape[1])

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2,
        "Adjusted R2": adj_r2,
        "Training Time Seconds": training_time
    })

results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
results_display = results_df.copy()
for col in ["MAE", "MSE", "RMSE"]:
    results_display[col] = results_display[col].map(lambda v: f"${v:,.2f}")
for col in ["R2 Score", "Adjusted R2"]:
    results_display[col] = results_display[col].map(lambda v: f"{v:.4f}")
results_display["Training Time Seconds"] = results_display["Training Time Seconds"].map(lambda v: f"{v:.2f}")

display_table(results_display, "Model Performance Results", max_rows=10)
results_df.to_csv(OUTPUT_DIR / "model_performance_results.csv", index=False)
save_table_as_png(results_display, "img17p17_model_performance_metrics.png", "Model Performance Metrics")

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
best_predictions = predictions[best_model_name]

info_card("Best Model", best_model_name, "Highest R² score", MAIN_BLUE)
info_card("Best MAE", f"${results_df.iloc[0]['MAE']:,.0f}", "Average prediction error", GREEN)
info_card("Best R²", f"{results_df.iloc[0]['R2 Score']:.4f}", "Variation explained", PURPLE)

best_summary = pd.DataFrame({
    "Metric": ["Best Model", "MAE", "MSE", "RMSE", "R2 Score", "Adjusted R2"],
    "Value": [
        best_model_name,
        f"${results_df.iloc[0]['MAE']:,.2f}",
        f"${results_df.iloc[0]['MSE']:,.2f}",
        f"${results_df.iloc[0]['RMSE']:,.2f}",
        f"{results_df.iloc[0]['R2 Score']:.4f}",
        f"{results_df.iloc[0]['Adjusted R2']:.4f}"
    ]
})
display_table(best_summary, "Best-Performing Model Summary", max_rows=10)
save_table_as_png(best_summary, "img21p19_best_model_summary.png", "Best-Performing Model Summary")


# STEP 11 - 5-FOLD CROSS-VALIDATION


section_title("Step 9 - 5-Fold Cross-Validation", "Mean cross-validation result checks whether performance is stable")

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
cv_fold_scores = {}

for model_name, model in models.items():
    print(f"Running 5-fold cross-validation: {model_name}")
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    cv_fold_scores[model_name] = scores
    cv_results.append({
        "Model": model_name,
        "Fold 1 R2": scores[0],
        "Fold 2 R2": scores[1],
        "Fold 3 R2": scores[2],
        "Fold 4 R2": scores[3],
        "Fold 5 R2": scores[4],
        "Mean 5-fold CV R2": scores.mean(),
        "Std 5-fold CV R2": scores.std()
    })

cv_results_df = pd.DataFrame(cv_results).sort_values(by="Mean 5-fold CV R2", ascending=False).reset_index(drop=True)
cv_display = cv_results_df.copy()
for col in cv_display.columns:
    if col != "Model":
        cv_display[col] = cv_display[col].map(lambda v: f"{v:.4f}")

display_table(cv_display, "5-Fold Cross-Validation Results", max_rows=10)
cv_results_df.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)
save_table_as_png(cv_display, "img18p17_cross_validation_results.png", "5-Fold Cross-Validation Results")

# CV fold line graph
plt.figure(figsize=(12, 7))

fold_numbers = np.arange(1, 6)

for model_name, scores in cv_fold_scores.items():
    plt.plot(
        fold_numbers,
        scores,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=model_name
    )

plt.title("5-Fold R² Evaluation for House Price Prediction")
plt.xlabel("Fold Number", fontweight="bold")
plt.ylabel("R² Score", fontweight="bold")
plt.xticks(fold_numbers)


plt.ylim(0.6, 1.0)

plt.grid(alpha=0.30, linestyle="--")
plt.legend()

save_figure("img29p_cv_fold_scores.png")

# Mean CV bar graph
plt.figure(figsize=(12, 7))
cv_plot = cv_results_df.sort_values(by="Mean 5-fold CV R2", ascending=True)
bars = plt.barh(cv_plot["Model"], cv_plot["Mean 5-fold CV R2"], color=MAIN_BLUE, edgecolor=DARK_TEXT)
plt.title("Mean 5-Fold Cross-Validation R² by Model")
plt.xlabel("Mean 5-Fold CV R²", fontweight="bold")
plt.ylabel("Machine Learning Model", fontweight="bold")
plt.xlim(0, 1)
plt.grid(axis="x", alpha=0.30, linestyle="--")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.4f}", va="center", fontweight="bold")
save_figure("img30p_mean_cv_r2_comparison.png")


# STEP 12 - MODEL COMPARISON GRAPHS


section_title("Step 10 - Model Comparison Graphs", "Visual comparison of errors and R² scores")

models_list = results_df["Model"].tolist()
x_positions = np.arange(len(models_list))
bar_width = 0.25

plt.figure(figsize=(13, 7))
mae_bars = plt.bar(x_positions - bar_width, results_df["MAE"], width=bar_width, label="MAE", color=GREEN, edgecolor=DARK_TEXT)
mse_scaled = results_df["MSE"] / 1_000_000
rmse_bars = plt.bar(x_positions, results_df["RMSE"], width=bar_width, label="RMSE", color=YELLOW, edgecolor=DARK_TEXT)
plt.title("Model Prediction Error Comparison")
plt.xlabel("Machine Learning Models", fontweight="bold")
plt.ylabel("Prediction Error Amount ($)", fontweight="bold")
plt.xticks(x_positions, models_list, rotation=22, ha="right", fontsize=10, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.30)
plt.legend(title="Error Metric")
for bars in [mae_bars, rmse_bars]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 5000, f"${height:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
save_figure("img19p18_model_error_comparison.png")

plt.figure(figsize=(12, 7))
bars = plt.bar(results_df["Model"], results_df["R2 Score"], color=[MAIN_BLUE, MID_BLUE, PURPLE, GREEN, ORANGE][:len(results_df)], edgecolor=DARK_TEXT, width=0.52)
plt.title("Model Comparison using R-Squared Score")
plt.xlabel("Machine Learning Model", fontweight="bold")
plt.ylabel("R² Score", fontweight="bold")
plt.ylim(0, 1)
plt.xticks(rotation=22, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.35)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.025, f"{height:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
save_figure("img20p18_model_r2_comparison.png")


# STEP 13 - ACTUAL VS PREDICTED AND RESIDUALS


section_title("Step 11 - Prediction Analysis", "Actual values are compared with predicted values for the best model")

prediction_table = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": best_predictions,
    "Error": y_test.values - best_predictions,
    "Absolute Error": np.abs(y_test.values - best_predictions)
}).head(15)

prediction_display = prediction_table.copy()
for col in prediction_display.columns:
    prediction_display[col] = prediction_display[col].map(lambda v: f"${v:,.2f}")

display_table(prediction_display, "Actual vs Predicted House Price Table", max_rows=15)
save_table_as_png(prediction_display, "img22p20_actual_vs_predicted_table.png", "Actual versus Predicted House Price Table")
prediction_table.to_csv(OUTPUT_DIR / "actual_vs_predicted_sample.csv", index=False)

plt.figure(figsize=(8.5, 8.5))
plt.scatter(y_test, best_predictions, alpha=0.45, color=MID_BLUE, edgecolor="white", linewidth=0.3)
minimum_value = min(y_test.min(), best_predictions.min())
maximum_value = max(y_test.max(), best_predictions.max())
plt.plot([minimum_value, maximum_value], [minimum_value, maximum_value], color=RED, linewidth=2.8, label="Perfect prediction line")
plt.title(f"Actual vs Predicted Prices: {best_model_name}")
plt.xlabel("Actual House Price ($)", fontweight="bold")
plt.ylabel("Predicted House Price ($)", fontweight="bold")
plt.gca().xaxis.set_major_formatter(FuncFormatter(money_formatter))
plt.gca().yaxis.set_major_formatter(FuncFormatter(money_formatter))
plt.grid(alpha=0.28, linestyle="--")
plt.legend()
save_figure("img23p21_actual_vs_predicted_best_model.png")

residuals = y_test - best_predictions
plt.figure(figsize=(12, 7))
plt.scatter(best_predictions, residuals, alpha=0.45, color=PURPLE, edgecolor="white", linewidth=0.3)
plt.axhline(0, color=RED, linewidth=2.8, linestyle="--")
plt.title(f"Residual Plot: {best_model_name}")
plt.xlabel("Predicted House Price ($)", fontweight="bold")
plt.ylabel("Residuals ($)", fontweight="bold")
plt.gca().xaxis.set_major_formatter(FuncFormatter(money_formatter))
plt.gca().yaxis.set_major_formatter(FuncFormatter(money_formatter))
plt.grid(alpha=0.28, linestyle="--")
save_figure("img24p22_residual_plot_best_model.png")

plt.figure(figsize=(12, 7))
plt.hist(residuals, bins=50, color=LIGHT_BLUE, edgecolor=MAIN_BLUE, linewidth=0.9)
plt.axvline(0, color=RED, linestyle="--", linewidth=2.8, label="Zero error")
plt.title("Distribution of Prediction Errors")
plt.xlabel("Prediction Error / Residual ($)", fontweight="bold")
plt.ylabel("Frequency", fontweight="bold")
plt.gca().xaxis.set_major_formatter(FuncFormatter(money_formatter))
plt.grid(alpha=0.28, linestyle="--")
plt.legend()
save_figure("img25p22_prediction_error_distribution.png")


# STEP 14 - FEATURE IMPORTANCE


section_title("Step 12 - Feature Importance", "Most influential predictors from the Random Forest model")

rf_pipeline = models["Random Forest Regressor"]
rf_preprocessor = rf_pipeline.named_steps["preprocessor"]
rf_model = rf_pipeline.named_steps["model"]

# Build feature names after preprocessing
processed_feature_names = []
processed_feature_names.extend(numeric_features)
if categorical_features:
    try:
        cat_names = rf_preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features).tolist()
    except Exception:
        cat_names = categorical_features
    processed_feature_names.extend(cat_names)

importance_values = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": processed_feature_names[:len(importance_values)],
    "Importance": importance_values
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

feature_importance_display = feature_importance.head(20).copy()
feature_importance_display["Importance"] = feature_importance_display["Importance"].map(lambda v: f"{v:.4f}")

display_table(feature_importance_display, "Top Random Forest Feature Importance Values", max_rows=20)
feature_importance.to_csv(OUTPUT_DIR / "feature_importance_results.csv", index=False)
save_table_as_png(feature_importance_display, "img26p24_feature_importance_table.png", "Random Forest Feature Importance Table")

top_features = feature_importance.head(15).sort_values(by="Importance")
plt.figure(figsize=(11, 7))
plt.barh(top_features["Feature"], top_features["Importance"], color=MAIN_BLUE, edgecolor=DARK_TEXT)
plt.title("Top 15 Random Forest Feature Importance Values")
plt.xlabel("Importance", fontweight="bold")
plt.ylabel("Feature", fontweight="bold")
plt.grid(axis="x", alpha=0.35, linestyle="--")
for i, value in enumerate(top_features["Importance"]):
    plt.text(value + 0.002, i, f"{value:.4f}", va="center", fontsize=9, fontweight="bold")
save_figure("img27p25_top_15_feature_importance.png")






