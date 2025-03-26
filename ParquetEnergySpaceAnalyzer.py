import argparse
import math  # Needed for ceil/floor
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# --- Configuration ---
DEFAULT_PARQUET_FILE = 'your_data.parquet'  # CHANGE THIS to your actual file name
DEFAULT_OUTPUT_TXT_FILE = 'min_energy_data.txt'
DEFAULT_PLOT_DIR = 'contour_plots'
GRID_RESOLUTION = 100j  # Complex number for meshgrid density (e.g., 100x100 grid)
INTERPOLATION_METHOD = 'cubic'  # 'linear', 'cubic', or 'nearest'
FILLED_CONTOUR_LEVELS = 15  # Number of color levels for contourf

# --- Columns ---
GROUP_COLS = ['B10', 'B20']
ENERGY_COL = 'TotalEnergy'
FLOAT_COLS_TO_PLOT = [
    'Mass',
    'TotalEnergy',
    'LiquidDropEnergy',
    'ShellCorrectionEnergy',
    'TotalSurfaceBindingEnergy',  # Special contour line step
    'TotalCoulombBindingEnergy',  # Special contour line step
    'LambdaPairingNeutrons',
    'LambdaPairingProtons',
    'DensityPairingNeutrons',
    'DensityPairingProtons',
    'LambdaShellNeutrons',
    'LambdaShellProtons'
]
AXIS_SCALE_FACTOR = 0.05
# Define columns with special contour line steps
SPECIAL_STEP_COLS = {
    'TotalSurfaceBindingEnergy': 10,
    'TotalCoulombBindingEnergy': 10
}
DEFAULT_CONTOUR_LINE_STEP = 1
CONTOUR_LINE_COLOR = 'black'
CONTOUR_LINE_WIDTH = 0.6
CONTOUR_LABEL_FONTSIZE = 8


# --- Main Function ---
def analyze_data(parquet_file, output_txt_file, plot_dir):
    """
    Reads parquet, finds minimum energy rows per (B10, B20) combo,
    saves them, and generates contour plots with labeled contour lines.
    """
    print(f"--- Starting Analysis ---")
    print(f"Input Parquet: {parquet_file}")
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
    print(f"\n[2/4] Finding rows with minimum '{ENERGY_COL}' for each unique ({', '.join(GROUP_COLS)}) combination...")
    required_cols = GROUP_COLS + [ENERGY_COL]
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
        idx = df_filtered.groupby(GROUP_COLS)[ENERGY_COL].idxmin()
        min_energy_df = df_filtered.loc[idx]
        print(f"Found {len(min_energy_df)} unique ({', '.join(GROUP_COLS)}) combinations with minimum energy data.")
    except Exception as e:
        print(f"Error during group-by minimum operation: {e}")
        return

    # --- 3. Save Minimum Energy Data ---
    print(f"\n[3/4] Saving minimum energy data to '{output_txt_file}'...")
    try:
        min_energy_df.to_csv(output_txt_file, sep='\t', index=False, float_format='%.6f')
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data to text file: {e}")

    # --- 4. Generate Contour Plots ---
    print(f"\n[4/4] Generating contour plots in '{plot_dir}'...")
    os.makedirs(plot_dir, exist_ok=True)

    plot_data = min_energy_df.dropna(subset=GROUP_COLS).copy()
    if plot_data.empty:
        print("No data available for plotting after final NaN check.")
        return

    y = plot_data['B10'] * AXIS_SCALE_FACTOR
    x = plot_data['B20'] * AXIS_SCALE_FACTOR

    try:
        grid_y_sparse, grid_x_sparse = np.mgrid[
                                       y.min():y.max():GRID_RESOLUTION,
                                       x.min():x.max():GRID_RESOLUTION
                                       ]
    except ValueError:
        print("Error: Could not create interpolation grid. Do X/Y coordinates have valid ranges?")
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

        # 1. Filled Contours (Color Gradient)
        contour_fill = ax.contourf(
            grid_x_sparse, grid_y_sparse, grid_z_masked,
            levels=FILLED_CONTOUR_LEVELS,
            cmap='viridis'
        )
        cbar = fig.colorbar(contour_fill, ax=ax)
        cbar.set_label(col)

        # 2. Labeled Contour Lines
        z_min = np.nanmin(grid_z)
        z_max = np.nanmax(grid_z)

        # Determine the step for contour lines
        step = SPECIAL_STEP_COLS.get(col, DEFAULT_CONTOUR_LINE_STEP)

        # Calculate levels for contour lines, ensuring range is valid
        if not np.isnan(z_min) and not np.isnan(z_max) and z_max > z_min:
            # Ensure start is rounded appropriately based on step
            start_level = math.floor(z_min / step) * step
            if start_level < z_min:  # Adjust if floor brought it below min
                start_level += step

            # Ensure end level includes the max value potentially
            end_level = math.ceil(z_max / step) * step
            if end_level > z_max:  # Adjust if ceil took it above max
                end_level -= step

            # Generate levels, adding a small epsilon to include the potential end_level if arange excludes it
            contour_line_levels = np.arange(start_level, end_level + step * 0.5, step)

            if len(contour_line_levels) > 0 and len(contour_line_levels) < 1000:  # Add sanity check for too many lines
                try:
                    contour_lines = ax.contour(
                        grid_x_sparse, grid_y_sparse, grid_z_masked,
                        levels=contour_line_levels,
                        colors=CONTOUR_LINE_COLOR,
                        linewidths=CONTOUR_LINE_WIDTH
                    )
                    # Add labels to the contour lines
                    ax.clabel(
                        contour_lines,
                        inline=True,
                        fontsize=CONTOUR_LABEL_FONTSIZE,
                        fmt='%1.0f'  # Format labels as integers
                    )
                except Exception as e:
                    print(f"    Warning: Could not draw/label contour lines for {col}: {e}")
            elif len(contour_line_levels) >= 1000:
                print(
                    f"    Warning: Too many contour lines requested ({len(contour_line_levels)}) for {col}. Skipping line drawing.")
            else:
                print(
                    f"    Info: No contour lines to draw for {col} in the calculated range [{z_min:.2f}, {z_max:.2f}] with step {step}.")

        else:
            print(
                f"    Info: Could not determine valid range for contour lines for {col} (min={z_min}, max={z_max}). Skipping line drawing.")

        # Labels and Title
        ax.set_xlabel(f'B20 * {AXIS_SCALE_FACTOR}')
        ax.set_ylabel(f'B10 * {AXIS_SCALE_FACTOR}')
        ax.set_title(f'Contour Plot of {col}')
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

    print(f"\n--- Plotting Summary ---")
    print(f"Successfully generated: {plots_generated} plots")
    print(f"Failed or skipped:    {plots_failed} plots")
    print(f"Plots saved in directory: '{plot_dir}'")
    print(f"--- Analysis Complete ---")


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze Parquet file: find min energy points and create contour plots.')
    parser.add_argument(
        'parquet_file',
        nargs='?',
        default=DEFAULT_PARQUET_FILE,
        help=f'Path to the input Parquet file (default: {DEFAULT_PARQUET_FILE})'
    )
    parser.add_argument(
        '-o', '--output',
        default=DEFAULT_OUTPUT_TXT_FILE,
        help=f'Path to the output text file for minimum energy data (default: {DEFAULT_OUTPUT_TXT_FILE})'
    )
    parser.add_argument(
        '-p', '--plotdir',
        default=DEFAULT_PLOT_DIR,
        help=f'Directory to save the contour plots (default: {DEFAULT_PLOT_DIR})'
    )

    args = parser.parse_args()

    analyze_data(args.parquet_file, args.output, args.plotdir)
