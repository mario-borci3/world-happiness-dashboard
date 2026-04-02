import argparse
import sys
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.io import to_html


# --------------------------------------
# Data loading / basic helpers
# --------------------------------------

def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file using the first row as the header."""
    try:
        df = pd.read_csv(file_path, header=0)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise


def preview_head(df: pd.DataFrame, n: int = 5) -> None:
    """Print the first n rows of the dataframe to verify loading."""
    print(f"\nFirst {n} rows of the dataset:")
    print(df.head(n).to_string(index=False))


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by stripping extra whitespace."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


# --------------------------------------
# Cleaning helpers
# --------------------------------------

def strip_whitespace_values(df: pd.DataFrame, columns=None) -> None:
    """
    Remove leading/trailing whitespace from values in specified columns.
    If columns is None, apply to all object/string columns.
    """
    if columns is None:
        columns = [
            c for c in df.columns
            if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
        ]

    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)


def replace_empty_strings_with_nan(df: pd.DataFrame, columns=None) -> None:
    """
    Replace empty strings / whitespace-only strings with NaN in specified columns.
    If columns is None, apply to all object/string columns.
    """
    if columns is None:
        columns = [
            c for c in df.columns
            if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
        ]

    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)


def cast_columns_appropriate_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to appropriate types based on simple inference.
    """
    df = df.copy()

    for col in df.columns:
        ser = df[col]

        if (
            pd.api.types.is_numeric_dtype(ser)
            or pd.api.types.is_datetime64_any_dtype(ser)
            or pd.api.types.is_bool_dtype(ser)
        ):
            continue

        ser_str = ser.astype("string").str.strip()

        s_num = pd.to_numeric(ser_str, errors="coerce")
        if s_num.notna().mean() >= 0.8:
            df[col] = s_num
            continue

        s_dt = pd.to_datetime(ser_str, errors="coerce")
        if s_dt.notna().mean() >= 0.8:
            df[col] = s_dt
            continue

        mapping = {
            "true": True, "false": False,
            "yes": True, "no": False,
            "1": True, "0": False,
            "t": True, "f": False
        }
        s_bool = ser_str.str.lower().map(mapping)
        if s_bool.notna().mean() >= 0.8:
            df[col] = s_bool.astype("boolean")
            continue

        df[col] = ser_str

    return df


def identify_missing_columns(df: pd.DataFrame) -> list:
    """Return list of columns that have any missing values."""
    return [col for col in df.columns if df[col].isna().any()]


def fill_missing_with_mean(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    For numeric columns in 'columns', fill missing values with the column mean.
    """
    df = df.copy()

    if columns is None:
        columns = identify_missing_columns(df)

    for col in columns:
        if col not in df.columns:
            continue
        ser = df[col]
        if pd.api.types.is_numeric_dtype(ser):
            mean_val = ser.mean()
            df[col] = ser.fillna(mean_val)
        else:
            print(f"[Info] Column '{col}' is non-numeric (dtype={ser.dtype}); skipping mean-fill.")
    return df


# --------------------------------------
# Column identification helpers
# --------------------------------------

def first_match(df: pd.DataFrame, conditions):
    """
    Return first column whose lowercase name satisfies any condition.
    """
    for col in df.columns:
        s = col.lower()
        for cond in conditions:
            if callable(cond) and cond(s):
                return col
            if isinstance(cond, (tuple, list)) and all(x in s for x in cond):
                return col
            if isinstance(cond, str) and cond in s:
                return col
    return None


def identify_columns(df: pd.DataFrame):
    """
    Identify key columns: country, happiness/score, GDP per capita, and Healthy Life Expectancy.
    """
    country_col = first_match(df, ["country"])
    if country_col is None:
        raise ValueError("Could not identify Country column.")

    happiness_col = first_match(df, [
        ("happiness", "score"),
        "happiness",
        "score",
    ])
    if happiness_col is None:
        raise ValueError("Could not identify Happiness Score column.")

    gdp_col = first_match(df, [
        ("gdp", "capita"),
        "gdp",
        "economy",
    ])
    if gdp_col is None:
        raise ValueError("Could not identify GDP per Capita column.")

    health_col = first_match(df, [
        ("health", "life"),
        ("life", "expect"),
        ("life", "expectancy"),
        "health",
    ])
    if health_col is None:
        raise ValueError("Could not identify Healthy Life Expectancy column.")

    return country_col, happiness_col, gdp_col, health_col


def identify_region_column(df: pd.DataFrame):
    """Identify Region column."""
    for c in df.columns:
        if "region" in c.lower():
            return c
    raise ValueError("Could not identify Region column.")


def identify_sub_dataset_columns(df: pd.DataFrame) -> dict:
    """
    Identify columns needed for the correlation sub-dataset.
    """
    economy_col = first_match(df, [
        ("gdp", "capita"),
        "gdp",
        "economy",
    ])
    if economy_col is None:
        raise ValueError("Could not identify Economy (GDP per Capita) column.")

    family_col = first_match(df, [
        "family",
    ])
    if family_col is None:
        raise ValueError("Could not identify Family column.")

    health_col = first_match(df, [
        ("health", "life"),
        ("life", "expect"),
        ("life", "expectancy"),
        "health",
    ])
    if health_col is None:
        raise ValueError("Could not identify Health (Life Expectancy) column.")

    freedom_col = first_match(df, [
        "freedom",
    ])
    if freedom_col is None:
        raise ValueError("Could not identify Freedom column.")

    trust_col = first_match(df, [
        "trust",
        ("government", "corruption"),
        "corruption",
    ])
    if trust_col is None:
        raise ValueError("Could not identify Trust (Government Corruption) column.")

    generosity_col = first_match(df, [
        "generosity",
    ])
    if generosity_col is None:
        raise ValueError("Could not identify Generosity column.")

    happiness_col = first_match(df, [
        ("happiness", "score"),
        "happiness",
        "score",
    ])
    if happiness_col is None:
        raise ValueError("Could not identify Happiness Score column.")

    return {
        "economy": economy_col,
        "family": family_col,
        "health": health_col,
        "freedom": freedom_col,
        "trust": trust_col,
        "generosity": generosity_col,
        "happiness": happiness_col
    }


# --------------------------------------
# Figure 1
# --------------------------------------

def build_fig1(top10_df: pd.DataFrame, country_col: str, gdp_col: str, health_col: str,
               output_path: str) -> go.Figure:
    """
    Build fig1 with two Y-axes showing GDP per capita and Healthy Life Expectancy
    for the top 10 countries by happiness score.
    """
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    fig1.add_trace(
        go.Bar(
            x=top10_df[country_col],
            y=top10_df[gdp_col],
            name="GDP per Capita"
        ),
        secondary_y=False
    )

    fig1.add_trace(
        go.Bar(
            x=top10_df[country_col],
            y=top10_df[health_col],
            name="Healthy Life Expectancy"
        ),
        secondary_y=True
    )

    fig1.update_layout(
        title="GDP per Capita and Healthy Life Expectancy (Top 10 Countries by Happiness Score)",
        barmode="group",
        width=1200,
        height=500,
        margin=dict(l=70, r=70, t=80, b=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    fig1.update_xaxes(title_text="Country", tickangle=-30)
    fig1.update_yaxes(title_text="GDP per Capita", secondary_y=False)
    fig1.update_yaxes(title_text="Healthy Life Expectancy", secondary_y=True)

    fig1.write_html(output_path)
    print(f"Figure fig1 created and saved to: {output_path}")
    return fig1


# --------------------------------------
# Figure 2
# --------------------------------------

def build_fig2(sub_df: pd.DataFrame, output_path: str = None, width: int = 800, height: int = 600) -> go.Figure:
    """
    Build a correlation heatmap (fig2) for the numeric sub_df.
    """
    numeric_sub = sub_df.apply(pd.to_numeric, errors="coerce")
    numeric_sub = numeric_sub.dropna(how="all")

    if numeric_sub.shape[1] < 2:
        raise ValueError("Sub-dataset for correlation must contain at least 2 numeric columns.")

    corr = numeric_sub.corr()

    fig2 = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation")
        )
    )

    annotations = []
    for i, row_label in enumerate(corr.index):
        for j, col_label in enumerate(corr.columns):
            annotations.append(
                dict(
                    x=col_label,
                    y=row_label,
                    text=f"{corr.values[i, j]:.2f}",
                    xref="x",
                    yref="y",
                    showarrow=False,
                    font=dict(color="black", size=12)
                )
            )

    fig2.update_layout(
        title="Correlation Heatmap",
        width=800,
        height=600,
        margin=dict(l=100, r=50, t=70, b=100),
        annotations=annotations
    )

    if output_path:
        fig2.write_html(output_path)
        print(f"Figure fig2 created and saved to: {output_path}")

    return fig2


# --------------------------------------
# Figure 3
# --------------------------------------

def build_fig3(df: pd.DataFrame,
               happiness_col: str,
               gdp_col: str,
               region_col: str,
               country_col: str,
               output_path: str = "fig3.html") -> go.Figure:
    """
    Build scatter plot (fig3) between Happiness Score and GDP per Capita.
    Points are colored by Region.
    """
    fig3 = go.Figure()

    regions = df[region_col].dropna().unique()

    for region in regions:
        subset = df[df[region_col] == region]

        fig3.add_trace(
            go.Scatter(
                x=subset[gdp_col],
                y=subset[happiness_col],
                mode="markers",
                name=str(region),
                marker=dict(size=8),
                text=subset[country_col],
                hovertemplate=(
                    "Country: %{text}<br>"
                    "GDP per Capita: %{x:.2f}<br>"
                    "Happiness Score: %{y:.2f}<br>"
                    f"Region: {region}<extra></extra>"
                )
            )
        )

    fig3.update_layout(
        title="Happiness Score vs GDP per Capita by Region",
        xaxis_title="GDP per Capita",
        yaxis_title="Happiness Score",
        legend_title="Region",
        width=900,
        height=600,
        margin=dict(l=70, r=40, t=70, b=70)
    )

    fig3.write_html(output_path)
    print(f"Figure fig3 created and saved to: {output_path}")

    return fig3


# --------------------------------------
# Figure 4
# --------------------------------------

def build_fig4(df: pd.DataFrame,
               happiness_col: str,
               region_col: str,
               output_path: str = "fig4.html") -> go.Figure:
    """
    Build a pie chart (fig4) showing average Happiness Score by Region.
    """
    region_happiness = (
        df[[region_col, happiness_col]]
        .dropna()
        .groupby(region_col, as_index=False)[happiness_col]
        .mean()
        .sort_values(happiness_col, ascending=False)
    )

    fig4 = go.Figure(
        data=[
            go.Pie(
                labels=region_happiness[region_col],
                values=region_happiness[happiness_col],
                textinfo="label+percent",
                textposition="outside",
                hole=0.25,
                hovertemplate="%{label}<br>Average Happiness Score: %{value:.2f}<extra></extra>"
            )
        ]
    )

    fig4.update_layout(
        title="Average Happiness Score by Region",
        width=900,
        height=700,
        margin=dict(l=40, r=40, t=80, b=40),
        showlegend=True
    )

    fig4.write_html(output_path)
    print(f"Figure fig4 created and saved to: {output_path}")

    return fig4


# --------------------------------------
# Figure 5
# --------------------------------------

def build_fig5(df: pd.DataFrame,
               country_col: str,
               gdp_col: str,
               health_col: str,
               output_path: str = "fig5.html") -> go.Figure:
    """
    Build a choropleth world map (fig5) displaying GDP per Capita by country.
    Healthy Life Expectancy is shown in the tooltip.
    """
    map_df = df[[country_col, gdp_col, health_col]].dropna().copy()

    fig5 = go.Figure(
        data=go.Choropleth(
            locations=map_df[country_col],
            locationmode="country names",
            z=map_df[gdp_col],
            colorscale="Blues",
            colorbar_title="GDP per Capita",
            text=map_df[country_col],
            customdata=map_df[[health_col]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "GDP per Capita: %{z:.2f}<br>"
                "Healthy Life Expectancy: %{customdata[0]:.2f}"
                "<extra></extra>"
            )
        )
    )

    fig5.update_layout(
        title="GDP per Capita by Country",
        width=1200,
        height=650,
        margin=dict(l=20, r=20, t=70, b=20),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="equirectangular",
            bgcolor="white"
        )
    )

    fig5.write_html(output_path)
    print(f"Figure fig5 created and saved to: {output_path}")

    return fig5


# --------------------------------------
# Combined dashboard
# --------------------------------------

def build_dashboard(fig1, fig2, fig3, fig4, fig5, output_path="dashboard.html") -> None:
    """
    Combine fig1-fig5 into a single HTML dashboard.
    """
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Happiness Dataset Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 24px;
                background-color: #f5f5f5;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
                max-width: 1600px;
                margin: 0 auto;
            }}
            .full-width {{
                grid-column: 1 / span 2;
            }}
            .card {{
                background: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                overflow: hidden;
            }}
            .plot-wrap {{
                width: 100%;
                overflow-x: auto;
            }}
            h2 {{
                margin-top: 0;
                margin-bottom: 12px;
                font-size: 24px;
            }}
        </style>
    </head>
    <body>
        <h1>World Happiness Report Dashboard</h1>

        <div class="grid">
            <div class="card full-width">
                <h2>Figure 1: Top 10 Countries by Happiness Score</h2>
                <div class="plot-wrap">
                    {to_html(fig1, full_html=False, include_plotlyjs='cdn', config={"responsive": False})}
                </div>
            </div>

            <div class="card">
                <h2>Figure 2: Correlation Heatmap</h2>
                <div class="plot-wrap">
                    {to_html(fig2, full_html=False, include_plotlyjs=False, config={"responsive": False})}
                </div>
            </div>

            <div class="card">
                <h2>Figure 3: Happiness Score vs GDP per Capita</h2>
                <div class="plot-wrap">
                    {to_html(fig3, full_html=False, include_plotlyjs=False, config={"responsive": False})}
                </div>
            </div>

            <div class="card">
                <h2>Figure 4: Average Happiness Score by Region</h2>
                <div class="plot-wrap">
                    {to_html(fig4, full_html=False, include_plotlyjs=False, config={"responsive": False})}
                </div>
            </div>

            <div class="card full-width">
                <h2>Figure 5: GDP per Capita by Country</h2>
                <div class="plot-wrap">
                    {to_html(fig5, full_html=False, include_plotlyjs=False, config={"responsive": False})}
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard created and saved to: {output_path}")

# --------------------------------------
# Main pipeline
# --------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load CSV, clean data, generate fig1 through fig5 and a combined dashboard."
    )
    parser.add_argument("path", help="Path to the CSV file (first row as header)")
    parser.add_argument("--head", "-n", type=int, default=5,
                        help="Number of rows to preview (default: 5)")
    parser.add_argument("--strip-columns", nargs="*",
                        help="Columns to strip whitespace. If omitted, all object/string columns are used.")
    parser.add_argument("--empty-to-nan-columns", nargs="*",
                        help="Columns to replace empty strings with NaN. If omitted, all object/string columns are used.")
    parser.add_argument("--cast-types", action="store_true",
                        help="Cast columns to appropriate types.")
    parser.add_argument("--fill-missing-with-mean", action="store_true",
                        help="Fill numeric missing values with the column means.")
    parser.add_argument("--fig-output", default="fig1.html",
                        help="Output HTML file name for fig1 (default: fig1.html)")
    parser.add_argument("--fig2-output", default="fig2.html",
                        help="Output HTML file name for fig2 (default: fig2.html)")
    parser.add_argument("--fig3-output", default="fig3.html",
                        help="Output HTML file name for fig3 (default: fig3.html)")
    parser.add_argument("--fig4-output", default="fig4.html",
                        help="Output HTML file name for fig4 (default: fig4.html)")
    parser.add_argument("--fig5-output", default="fig5.html",
                        help="Output HTML file name for fig5 (default: fig5.html)")
    parser.add_argument("--dashboard-output", default="dashboard.html",
                        help="Output HTML file name for combined dashboard (default: dashboard.html)")

    args = parser.parse_args()

    # 1) Load dataset
    df = load_csv(args.path)
    df = standardize_column_names(df)

    # 2) Preview
    preview_head(df, max(1, args.head))

    # 3) Cleaning steps
    strip_whitespace_values(df, args.strip_columns)
    replace_empty_strings_with_nan(df, args.empty_to_nan_columns)

    if args.cast_types:
        df = cast_columns_appropriate_types(df)

    # Force known numeric columns for this dataset
    known_numeric_cols = [
        "Happiness Rank",
        "Happiness Score",
        "Lower Confidence Interval",
        "Upper Confidence Interval",
        "Economy (GDP per Capita)",
        "Family",
        "Health (Life Expectancy)",
        "Freedom",
        "Trust (Government Corruption)",
        "Generosity",
        "Dystopia Residual"
    ]

    for col in known_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_cols = identify_missing_columns(df)
    print("\nColumns with missing values:")
    print(missing_cols if missing_cols else "None")

    if args.fill_missing_with_mean:
        df = fill_missing_with_mean(df, missing_cols)

    print("\nData types after cleaning:")
    print(df.dtypes)

    print("\nPreview after cleaning (first 10 rows):")
    print(df.head(10).to_string(index=False))

    try:
        # Identify main columns
        country_col, happiness_col, gdp_col, health_col = identify_columns(df)
        region_col = identify_region_column(df)

        # Top 10 countries by happiness score
        top10_df = (
            df.dropna(subset=[country_col, happiness_col, gdp_col, health_col])
              .sort_values(by=happiness_col, ascending=False)
              [[country_col, happiness_col, gdp_col, health_col]]
              .head(10)
              .copy()
        )

        print("\nTop 10 countries by happiness score (with GDP and Health):")
        print(top10_df.to_string(index=False))

        # fig1
        fig1 = build_fig1(top10_df, country_col, gdp_col, health_col, args.fig_output)

        # fig2
        cols = identify_sub_dataset_columns(df)
        sub_cols = [
            cols["economy"],
            cols["family"],
            cols["health"],
            cols["freedom"],
            cols["trust"],
            cols["generosity"],
            cols["happiness"],
        ]
        sub_df = df[sub_cols].copy()
        fig2 = build_fig2(sub_df, output_path=args.fig2_output, width=800, height=600)

        # fig3
        fig3 = build_fig3(
            df=df.dropna(subset=[happiness_col, gdp_col, region_col, country_col]),
            happiness_col=happiness_col,
            gdp_col=gdp_col,
            region_col=region_col,
            country_col=country_col,
            output_path=args.fig3_output
        )

        # fig4
        fig4 = build_fig4(
            df=df,
            happiness_col=happiness_col,
            region_col=region_col,
            output_path=args.fig4_output
        )

        # fig5
        fig5 = build_fig5(
            df=df,
            country_col=country_col,
            gdp_col=gdp_col,
            health_col=health_col,
            output_path=args.fig5_output
        )

        # dashboard
        build_dashboard(fig1, fig2, fig3, fig4, fig5, args.dashboard_output)

        return fig1, fig2, fig3, fig4, fig5

    except Exception as e:
        print(f"Error during processing or figure generation: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()