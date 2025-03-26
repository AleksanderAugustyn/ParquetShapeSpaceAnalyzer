import argparse
import math  # Needed for ceil/floor
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# --- Configuration ---
DEFAULT_OUTPUT_DIR = '.'  # Base directory for outputs
DEFAULT_PLOT_BASE_DIR = 'contour_plots'  # Base name for plot directories
GRID_RESOLUTION = 100j  # Complex number for meshgrid density (e.g., 100x100 grid)
INTERPOLATION_METHOD = 'cubic'  # 'linear', 'cubic', or 'nearest'
FILLED_CONTOUR_LEVELS = 15  # Number of color levels for contourf

# --- Column Definitions ---
X_AXIS_COL = 'B20'  # Fixed X-axis column
POSSIBLE_Y_AXIS_COLS = ['B10', 'B30', 'B40', 'B50', 'B60']  # Allowed choices for Y
ENERGY_COL = 'TotalEnergy'
FLOAT_COLS_TO_PLOT = [
    'Mass',
    'TotalEnergy',
    'LiquidDropEnergy',
    'ShellCorrectionEnergy',
    'TotalSurfaceBindingEnergy',
    'TotalCoulombBindingEnergy',
    'LambdaPairingNeutrons',
    'LambdaPairingProtons',
    'DensityPairingNeutrons',
    'DensityPairingProtons',
    'LambdaShellNeutrons',
    'LambdaShellProtons'
]
AXIS_SCALE_FACTOR = 0.05

# --- Contour Line Step Definitions ---
CONTOUR_LINE_STEPS = {
    'DensityPairingNeutrons': 0.05, 'DensityPairingProtons': 0.05,
    'LambdaPairingNeutrons': 0.5, 'LambdaPairingProtons': 0.5,
    'LambdaShellNeutrons': 0.5, 'LambdaShellProtons': 0.5,
    'TotalSurfaceBindingEnergy': 10, 'TotalCoulombBindingEnergy': 10,
    'Mass': 1, 'TotalEnergy': 1, 'LiquidDropEnergy': 1, 'ShellCorrectionEnergy': 1,
}
DEFAULT_CONTOUR_LINE_STEP = 1
CONTOUR_LINE_COLOR = 'black'
CONTOUR_LINE_WIDTH = 0.6
CONTOUR_LABEL_FONTSIZE = 8
CONTOUR_LABEL_FORMAT = '%g'


# --- Main Analysis Function ---
def analyze_data(parquet_file, y_axis_col, output_dir, plot_base_dir):
    """
    Reads parquet, finds minimum energy rows for a given (y_axis_col, B20) combo,
    sorts them, saves them, and generates contour plots.
    """
    if y_axis_col not in POSSIBLE_Y_AXIS_COLS:
        print(f"Error: Invalid Y-axis column specified: '{y_axis_col}'. "
              f"Choose from: {POSSIBLE_Y_AXIS_COLS}")
        return

    group_cols = [y_axis_col, X_AXIS_COL]
    output_suffix = f"{y_axis_col}_vs_{X_AXIS_COL}"
    output_txt_file = os.path.join(output_dir, f"min_energy_{output_suffix}.txt")
    plot_dir = os.path.join(output_dir, f"{plot_base_dir}_{output_suffix}")

    print(f"--- Starting Analysis for ({y_axis_col}, {X_AXIS_COL}) ---")
    print(f"Input Parquet: {parquet_file}")
    print(f"Grouping Columns: {group_cols}")
    print(f"Output Text: {output_txt_file}")
    print(f"Plot Directory: {plot_dir}")

    # --- 1. Read Parquet Data ---
    try:
        print(f"\n[1/4] Reading Parquet file: {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        print(f"Successfully read {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: Parquet file not found at '{parquet_file}'")
        return
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return

    # --- 2. Find Minimum Energy Rows ---
    print(f"\n[2/4] Finding rows with minimum '{ENERGY_COL}' for each unique ({', '.join(group_cols)}) combination...")
    required_cols = group_cols + [ENERGY_COL]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame missing one or more required columns: {required_cols}")
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Missing: {missing}")
        return

    df_filtered = df.dropna(subset=required_cols)
    if len(df_filtered) < len(df):
        print(f"Dropped {len(df) - len(df_filtered)} rows with NaN values in {required_cols}")
    if df_filtered.empty:
        print("Error: No valid data remaining after dropping NaN in required columns.")
        return

    try:
        idx = df_filtered.groupby(group_cols)[ENERGY_COL].idxmin()
        min_energy_df = df_filtered.loc[idx].copy()  # Use .copy() to avoid SettingWithCopyWarning later if needed
        print(f"Found {len(min_energy_df)} unique ({', '.join(group_cols)}) combinations with minimum energy data.")
    except Exception as e:
        print(f"Error during group-by minimum operation: {e}")
        return

    # --- 3. Sort and Save Minimum Energy Data ---
    print(f"\n[3/4] Sorting and saving minimum energy data to '{output_txt_file}'...")
    try:
        # Sort the DataFrame before saving
        print(f"Sorting by {X_AXIS_COL} then {y_axis_col}...")
        min_energy_df.sort_values(by=[X_AXIS_COL, y_axis_col], inplace=True)

        # Save the sorted data
        os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)
        min_energy_df.to_csv(output_txt_file, sep='\t', index=False, float_format='%.6f')
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error sorting or saving data to text file: {e}")
        # Continue to plotting even if saving fails/has issues

    # --- 4. Generate Contour Plots ---
    # (Plotting code remains unchanged from the previous version)
    print(f"\n[4/4] Generating contour plots in '{plot_dir}'...")
    os.makedirs(plot_dir, exist_ok=True)

    plot_data = min_energy_df.dropna(subset=group_cols).copy()  # Use the already potentially modified min_energy_df
    if plot_data.empty:
        print("No data available for plotting after final NaN check.")
        return

    y = plot_data[y_axis_col] * AXIS_SCALE_FACTOR
    x = plot_data[X_AXIS_COL] * AXIS_SCALE_FACTOR

    try:
        y_min, y_max = y.min(), y.max()
        x_min, x_max = x.min(), x.max()
        if y_min == y_max or x_min == x_max:
            print("Warning: Data range for X or Y axis is zero. Cannot create interpolation grid.")
            return

        grid_y_sparse, grid_x_sparse = np.mgrid[
                                       y_min:y_max:GRID_RESOLUTION,
                                       x_min:x_max:GRID_RESOLUTION
                                       ]
    except ValueError as ve:
        print(f"Error: Could not create interpolation grid: {ve}. Check X/Y coordinate ranges.")
        print(f"Y range ({y_axis_col}): {y_min} to {y_max}")
        print(f"X range ({X_AXIS_COL}): {x_min} to {x_max}")
        return

    plots_generated = 0
    plots_failed = 0

    for col in FLOAT_COLS_TO_PLOT:
        if col not in plot_data.columns:
            print(f"Skipping plot for '{col}': Column not found in minimum energy data.")
            plots_failed += 1
            continue

        print(f"  Generating plot for: {col}...")
        z = plot_data[col].copy()
        valid_indices = ~z.isna() & ~x.isna() & ~y.isna()

        if not valid_indices.any():
            print(f"    Skipping '{col}': No valid (non-NaN) data points for plotting.")
            plots_failed += 1
            continue

        x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        z_valid = z[valid_indices]

        try:
            grid_z = griddata(
                (y_valid, x_valid),
                z_valid,
                (grid_y_sparse, grid_x_sparse),
                method=INTERPOLATION_METHOD
            )
        except Exception as e:
            print(f"    Error during interpolation for '{col}': {e}")
            plots_failed += 1
            continue

        if np.all(np.isnan(grid_z)):
            print(f"    Skipping '{col}': Interpolation resulted in all NaNs.")
            plots_failed += 1
            continue

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 8))
        grid_z_masked = np.ma.masked_invalid(grid_z)

        # 1. Filled Contours
        contour_fill = ax.contourf(
            grid_x_sparse, grid_y_sparse, grid_z_masked,
            levels=FILLED_CONTOUR_LEVELS, cmap='viridis'
        )
        cbar = fig.colorbar(contour_fill, ax=ax)
        cbar.set_label(col)

        # 2. Labeled Contour Lines
        z_min = np.nanmin(grid_z)
        z_max = np.nanmax(grid_z)
        step = CONTOUR_LINE_STEPS.get(col, DEFAULT_CONTOUR_LINE_STEP)

        if not np.isnan(z_min) and not np.isnan(z_max) and z_max > z_min:
            start_level = math.floor(z_min / step) * step
            while start_level < z_min: start_level += step
            end_level = math.ceil(z_max / step) * step
            while end_level > z_max: end_level -= step

            num_steps = int(round((end_level - start_level) / step)) if step > 0 else 0
            if num_steps >= 0 and num_steps < 1000:
                contour_line_levels = np.linspace(start_level, end_level, num_steps + 1)
            elif num_steps >= 1000:
                contour_line_levels = []
                print(f"    Warning: Too many contour lines requested ({num_steps + 1}) for {col}. Skipping.")
            else:
                contour_line_levels = []

            contour_line_levels = np.unique(contour_line_levels)

            if len(contour_line_levels) > 1:
                try:
                    contour_lines = ax.contour(
                        grid_x_sparse, grid_y_sparse, grid_z_masked,
                        levels=contour_line_levels, colors=CONTOUR_LINE_COLOR,
                        linewidths=CONTOUR_LINE_WIDTH
                    )
                    ax.clabel(contour_lines, inline=True,
                              fontsize=CONTOUR_LABEL_FONTSIZE, fmt=CONTOUR_LABEL_FORMAT
                              )
                except Exception as e:
                    print(f"    Warning: Could not draw/label contour lines for {col}: {e}")
            elif len(contour_line_levels) <= 1 and num_steps >= 0:
                print(
                    f"    Info: Not enough distinct contour levels ({len(contour_line_levels)}) to draw for {col} in range [{z_min:.3f}, {z_max:.3f}] with step {step}.")

        elif not (np.isnan(z_min) or np.isnan(z_max)):
            print(f"    Info: Cannot determine distinct contour lines for {col} (min={z_min}, max={z_max}). Skipping.")

        # Labels and Title
        ax.set_xlabel(f'{X_AXIS_COL}')
        ax.set_ylabel(f'{y_axis_col}')
        ax.set_title(f'Contour Plot of {col} ({y_axis_col} vs {X_AXIS_COL})')
        ax.set_aspect('equal', adjustable='box')

        # Save the plot
        plot_filename = os.path.join(plot_dir, f'contour_{col}.png')
        try:
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plots_generated += 1
        except Exception as e:
            print(f"    Error saving plot {plot_filename}: {e}")
            plots_failed += 1

        plt.close(fig)
    # --- End of Plotting Loop ---

    print(f"\n--- Plotting Summary for ({y_axis_col}, {X_AXIS_COL}) ---")
    print(f"Successfully generated: {plots_generated} plots")
    print(f"Failed or skipped:    {plots_failed} plots")
    print(f"Plots saved in directory: '{plot_dir}'")
    print(f"--- Analysis Complete for ({y_axis_col}, {X_AXIS_COL}) ---")


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze Parquet file: find min energy points for a (Y_COL, B20) combination and create contour plots.'
    )
    parser.add_argument(
        'parquet_file',
        help='Path to the input Parquet file.'
    )
    parser.add_argument(
        '--y-col',
        required=True,
        choices=POSSIBLE_Y_AXIS_COLS,
        help='Which B column (B10, B30, B40, B50, B60) to use for the Y-axis and grouping with B20.'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Base directory to save the output text file (default: current directory "{DEFAULT_OUTPUT_DIR}")'
    )
    parser.add_argument(
        '-p', '--plot-basedir',
        default=DEFAULT_PLOT_BASE_DIR,
        help=f'Base name for the plot directory (suffixed with Y_COL_vs_B20) (default: "{DEFAULT_PLOT_BASE_DIR}")'
    )

    args = parser.parse_args()

    analyze_data(args.parquet_file, args.y_col, args.output_dir, args.plot_basedir)
