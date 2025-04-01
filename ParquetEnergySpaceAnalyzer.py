import argparse
import math  # Needed for ceil/floor
import os
import re  # For sanitizing filenames
import sys  # To exit on error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# --- Configuration ---
# DEFAULT_OUTPUT_DIR is replaced by a derived path
DEFAULT_PLOT_BASE_DIR = 'contour_plots'  # Base name for plot directories within the main output folder
INPUT_BASE_DIR = './MapFragments'
OUTPUT_BASE_DIR = './MiniMaps'
GRID_RESOLUTION = 100j  # Complex number for meshgrid density (e.g., 100x100 grid)
INTERPOLATION_METHOD = 'cubic'  # 'linear', 'cubic', or 'nearest'
FILLED_CONTOUR_LEVELS = 15  # Number of color levels for contourf
MAX_EXPLICIT_CONTOUR_LINES = 100  # Max number of lines to calculate using step size
DEFAULT_AUTO_CONTOUR_LINES = 80  # Number of lines if explicit calculation yields too many

# --- Column Definitions ---
X_AXIS_COL = 'B20'  # Fixed X-axis column FOR THE FIRST ANALYSIS TYPE
POSSIBLE_Y_AXIS_COLS = ['B10', 'B30', 'B40', 'B50', 'B60']  # Allowed choices for Y (iterated in first analysis)
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
AXIS_SCALE_FACTOR = 0.05  # Apply scale factor for plotting axes

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

# --- Constants for B10 Constant Analysis ---
B10_CONST_ANALYSIS_DIR = 'B10ConstantMaps'
B10_CONST_FIXED_COL = 'B10'
B10_CONST_X_COL = 'B20'
B10_CONST_Y_COLS = ['B30', 'B40', 'B50', 'B60']  # B10 vs B20 is already done in the first analysis


# --- Helper Function: Sanitize Filename (FIXED) ---
def sanitize_filename(name):
    """Removes or replaces characters problematic for filenames.
       Specifically handles negative signs to avoid collisions.
    """
    name_str = str(name)

    # Replace hyphen with 'neg' *before* other sanitation
    # This ensures '-29' becomes 'neg29' and '29' remains '29'
    name_str = name_str.replace('-', 'neg')

    # Replace '.' with 'p' (useful for floats)
    name_str = name_str.replace('.', 'p')

    # Remove any remaining characters that are not alphanumeric or underscore
    name_str = re.sub(r'\W+', '_', name_str)

    # Remove leading/trailing underscores
    name_str = re.sub(r'^_+', '', name_str)
    name_str = re.sub(r'_+$', '', name_str)

    return name_str if name_str else "value"  # Return "value" if sanitation results in empty string


# --- Helper Function: Generate Single Contour Plot (Filename construction modified) ---
def generate_contour_plot(
        plot_data_df, x_col, y_col, z_col, plot_dir,
        title_prefix="", title_suffix="", filename_suffix=""
):
    """
    Generates and saves a single contour plot for the given Z column.
    Filename construction uses the provided filename_suffix. If empty, defaults to contour_{z_col}.png.

    Args:
        plot_data_df (pd.DataFrame): DataFrame containing the minimum energy data for plotting.
        x_col (str): Column name for the X-axis.
        y_col (str): Column name for the Y-axis.
        z_col (str): Column name for the Z-axis (the value being plotted).
        plot_dir (str): Directory to save the plot.
        title_prefix (str): Optional prefix for the plot title.
        title_suffix (str): Optional suffix for the plot title.
        filename_suffix (str): Optional suffix for the plot filename (before .png), used to add context like constant values.

    Returns:
        bool: True if plot generation succeeded, False otherwise.
    """
    print(f"    Generating plot for: {z_col} ({y_col} vs {x_col})...")

    # Ensure necessary columns exist
    if not all(c in plot_data_df.columns for c in [x_col, y_col, z_col]):
        print(f"    Skipping plot for '{z_col}': Missing one or more columns ({x_col}, {y_col}, {z_col}).")
        return False

    # Prepare data, apply scaling, handle NaNs
    plot_data = plot_data_df.dropna(subset=[x_col, y_col]).copy()
    if plot_data.empty:
        print(f"    Skipping '{z_col}': No data available after dropping NaNs in {x_col}/{y_col}.")
        return False

    y = plot_data[y_col] * AXIS_SCALE_FACTOR
    x = plot_data[x_col] * AXIS_SCALE_FACTOR
    z = plot_data[z_col]  # Z data (value being plotted)

    valid_indices = ~z.isna() & ~x.isna() & ~y.isna()
    if not valid_indices.any():
        print(f"    Skipping '{z_col}': No valid (non-NaN) data points for plotting (X, Y, Z).")
        return False

    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    z_valid = z[valid_indices]

    # Check if we have enough unique points for interpolation
    if len(np.unique(x_valid)) < 2 or len(np.unique(y_valid)) < 2:
        print(
            f"    Skipping '{z_col}': Not enough unique X or Y points for interpolation ({len(np.unique(x_valid))} unique X, {len(np.unique(y_valid))} unique Y).")
        return False
    if len(x_valid) < 3 and INTERPOLATION_METHOD != 'nearest':
        print(
            f"    Skipping '{z_col}': Not enough data points ({len(x_valid)}) for '{INTERPOLATION_METHOD}' interpolation.")
        return False

    # Create grid
    try:
        y_min, y_max = y_valid.min(), y_valid.max()
        x_min, x_max = x_valid.min(), x_valid.max()
        if y_min == y_max or x_min == x_max:
            print(
                f"    Warning for '{z_col}': Data range for X ({x_col}) or Y ({y_col}) axis is zero after scaling. Cannot create interpolation grid.")
            return False

        grid_y_sparse, grid_x_sparse = np.mgrid[
                                       y_min:y_max:GRID_RESOLUTION,
                                       x_min:x_max:GRID_RESOLUTION
                                       ]
    except ValueError as ve:
        print(f"    Error: Could not create interpolation grid for '{z_col}': {ve}. Check X/Y coordinate ranges.")
        print(f"    Scaled Y range ({y_col}): {y_min} to {y_max}")
        print(f"    Scaled X range ({x_col}): {x_min} to {x_max}")
        return False

    # Interpolate
    try:
        grid_z = griddata(
            (y_valid, x_valid),  # Note: griddata expects points as (y, x)
            z_valid,
            (grid_y_sparse, grid_x_sparse),
            method=INTERPOLATION_METHOD,
            fill_value=np.nan  # Use NaN for points outside the convex hull
        )
    except Exception as e:
        print(f"    Error during interpolation for '{z_col}': {e}")
        return False

    if np.all(np.isnan(grid_z)):
        print(f"    Skipping '{z_col}': Interpolation resulted in all NaNs.")
        return False

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))
    grid_z_masked = np.ma.masked_invalid(grid_z)  # Mask NaNs in the grid for contouring

    # 1. Filled Contours
    try:
        # Check if there's any valid data left after masking for contourf
        if not grid_z_masked.mask.all():
            contour_fill = ax.contourf(
                grid_x_sparse, grid_y_sparse, grid_z_masked,
                levels=FILLED_CONTOUR_LEVELS, cmap='viridis', extend='both'
            )
            cbar = fig.colorbar(contour_fill, ax=ax)
            cbar.set_label(z_col)
        else:
            print(
                f"    Info: No valid data in the interpolated grid for {z_col} after masking. Skipping filled contour.")
    except ValueError as ve:
        print(f"    Error generating filled contour for {z_col}: {ve}.")
        plt.close(fig)
        return False
    except Exception as e:
        print(f"    Unexpected error during filled contour plotting for {z_col}: {e}")
        plt.close(fig)
        return False

    # 2. Labeled Contour Lines -- REVISED LOGIC --
    z_min_grid = np.nanmin(grid_z_masked)
    z_max_grid = np.nanmax(grid_z_masked)
    step = CONTOUR_LINE_STEPS.get(z_col, DEFAULT_CONTOUR_LINE_STEP)

    contour_levels_arg = None
    should_label_contours = False

    if not pd.isna(z_min_grid) and not pd.isna(z_max_grid) and z_max_grid > z_min_grid and step > 0:
        start_level = math.ceil(z_min_grid / step) * step
        end_level = math.floor(z_max_grid / step) * step

        if start_level <= end_level:
            num_steps = int(round((end_level - start_level) / step))

            if num_steps >= 0 and num_steps < MAX_EXPLICIT_CONTOUR_LINES:
                explicit_levels = np.linspace(start_level, end_level, num_steps + 1)
                explicit_levels = np.unique(np.round(explicit_levels, 6))

                if len(explicit_levels) > 1:
                    contour_levels_arg = explicit_levels
                    should_label_contours = True
                    print(f"    Using {len(contour_levels_arg)} explicit contour levels for {z_col}.")
                else:
                    print(f"    Info: Only one distinct explicit contour level calculated for {z_col}. Skipping lines.")
            elif num_steps >= MAX_EXPLICIT_CONTOUR_LINES:
                print(
                    f"    Info: Explicit step calculation yielded too many levels ({num_steps + 1}). Using ~{DEFAULT_AUTO_CONTOUR_LINES} automatic levels for {z_col}.")
                contour_levels_arg = DEFAULT_AUTO_CONTOUR_LINES
                should_label_contours = False
        else:
            print(
                f"    Info: Grid range [{z_min_grid:.3f}, {z_max_grid:.3f}] too narrow for step {step}. Skipping contour lines for {z_col}.")
    else:
        if pd.isna(z_min_grid) or pd.isna(z_max_grid):
            print(
                f"    Info: Cannot determine grid data range for contour lines for {z_col} (grid min/max is NaN). Skipping lines.")
        else:
            print(
                f"    Info: Cannot determine distinct contour lines for {z_col} (grid_min={z_min_grid:.3f}, grid_max={z_max_grid:.3f}, step={step}). Skipping lines.")

    if contour_levels_arg is not None:
        try:
            contour_lines = ax.contour(
                grid_x_sparse, grid_y_sparse, grid_z_masked,
                levels=contour_levels_arg,
                colors=CONTOUR_LINE_COLOR,
                linewidths=CONTOUR_LINE_WIDTH
            )
            if should_label_contours:
                ax.clabel(contour_lines, inline=True,
                          fontsize=CONTOUR_LABEL_FONTSIZE, fmt=CONTOUR_LABEL_FORMAT)
        except ValueError as ve:
            print(f"    Warning: Could not draw contour lines for {z_col} (levels arg: {contour_levels_arg}): {ve}")
        except Exception as e:
            print(f"    Warning: Unexpected error drawing contour lines for {z_col}: {e}")

    # Labels and Title
    ax.set_xlabel(f'{x_col}')
    ax.set_ylabel(f'{y_col}')
    title = f'{title_prefix}Contour Plot of {z_col} ({y_col} vs {x_col}){title_suffix}'
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

    # Save the plot - Use filename_suffix if provided
    if filename_suffix:
        # Ensure suffix starts appropriately, e.g., with '_' if desired by caller
        plot_filename_base = f'contour_{z_col}{filename_suffix}.png'
    else:
        plot_filename_base = f'contour_{z_col}.png'

    plot_filename = os.path.join(plot_dir, plot_filename_base)
    try:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"    Error saving plot {plot_filename}: {e}")
        plt.close(fig)
        return False

    plt.close(fig)
    return True


# --- Main Analysis Function (First Type - Unchanged) ---
def analyze_data_for_y_col(df, y_axis_col, base_output_dir):
    """
    Reads data, finds minimum energy rows for a given (y_axis_col, B20) combo,
    sorts them, saves them, and generates contour plots using a helper function.

    Args:
        df (pd.DataFrame): The input DataFrame.
        y_axis_col (str): The column name to use for the Y-axis (e.g., 'B10', 'B30').
        base_output_dir (str): The main directory for outputs (e.g., './MiniMaps/Z_N').
    """
    group_cols = [y_axis_col, X_AXIS_COL]  # X_AXIS_COL is fixed 'B20' here
    output_suffix = f"{y_axis_col}_vs_{X_AXIS_COL}"
    output_txt_file = os.path.join(base_output_dir, f"min_energy_{output_suffix}.txt")
    plot_dir = os.path.join(base_output_dir, f"{DEFAULT_PLOT_BASE_DIR}_{output_suffix}")

    print(f"\n--- Starting Analysis for ({y_axis_col}, {X_AXIS_COL}) ---")
    print(f"Grouping Columns: {group_cols}")
    print(f"Output Text: {output_txt_file}")
    print(f"Plot Directory: {plot_dir}")

    # --- 1. Data Already Read ---
    print("\n[1/4] Using pre-loaded DataFrame...")
    if df is None or df.empty:
        print("Error: Input DataFrame is missing or empty.")
        return False

    # --- 2. Find Minimum Energy Rows ---
    print(f"\n[2/4] Finding rows with minimum '{ENERGY_COL}' for each unique ({', '.join(group_cols)}) combination...")
    required_cols = group_cols + [ENERGY_COL]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame missing one or more required columns: {required_cols}")
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Missing: {missing}")
        return False

    df_filtered = df.dropna(subset=group_cols + [ENERGY_COL])
    if len(df_filtered) < len(df):
        print(
            f"Dropped {len(df) - len(df_filtered)} rows with NaN values in required columns {group_cols + [ENERGY_COL]}")
    if df_filtered.empty:
        print("Error: No valid data remaining after dropping NaN in required columns.")
        return False

    try:
        idx = df_filtered.groupby(group_cols, observed=True)[ENERGY_COL].idxmin()
        min_energy_df = df_filtered.loc[idx].copy()
        print(f"Found {len(min_energy_df)} unique ({', '.join(group_cols)}) combinations with minimum energy data.")
        if min_energy_df.empty:
            print("Warning: Minimum energy DataFrame is empty after grouping. No data to process or plot.")
            return False
    except KeyError as ke:
        print(f"Error during group-by minimum operation: Missing column {ke}.")
        return False
    except Exception as e:
        print(f"Error during group-by minimum operation: {e}")
        return False

    # --- 3. Sort and Save Minimum Energy Data ---
    print(f"\n[3/4] Sorting and saving minimum energy data to '{output_txt_file}'...")
    try:
        os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)
        print(f"Sorting by {X_AXIS_COL} then {y_axis_col}...")
        min_energy_df.sort_values(by=[X_AXIS_COL, y_axis_col], inplace=True)
        min_energy_df.to_csv(output_txt_file, sep='\t', index=False, float_format='%.6f')
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error sorting or saving data to text file: {e}")


    # --- 4. Generate Contour Plots using Helper ---
    print(f"\n[4/4] Generating contour plots in '{plot_dir}'...")
    os.makedirs(plot_dir, exist_ok=True)

    plots_generated = 0
    plots_failed = 0

    title_suffix_main = f" ({y_axis_col} vs {X_AXIS_COL})"
    # *** Use filename_suffix for this analysis type to distinguish plots from different Y-axes ***
    filename_suffix_main = f"_{y_axis_col}_vs_{X_AXIS_COL}"  # e.g., _B10_vs_B20

    for col in FLOAT_COLS_TO_PLOT:
        if col not in min_energy_df.columns:
            print(f"    Skipping plot for '{col}': Column not found in minimum energy data.")
            plots_failed += 1
            continue

        # Call the helper function
        success = generate_contour_plot(
            plot_data_df=min_energy_df,
            x_col=X_AXIS_COL,
            y_col=y_axis_col,
            z_col=col,
            plot_dir=plot_dir,
            title_suffix=title_suffix_main,
            filename_suffix=filename_suffix_main  # Pass suffix here
        )
        if success:
            plots_generated += 1
        else:
            plots_failed += 1

    print(f"\n--- Plotting Summary for ({y_axis_col}, {X_AXIS_COL}) ---")
    print(f"Successfully generated: {plots_generated} plots")
    print(f"Failed or skipped:    {plots_failed} plots")
    print(f"Plots saved in directory: '{plot_dir}'")
    print(f"--- Analysis Complete for ({y_axis_col}, {X_AXIS_COL}) ---")
    return True


# --- NEW Analysis Function: B10 Constant (MODIFIED for Filenames) ---
def analyze_constant_b10(full_df, base_output_dir_zn):
    """
    Generates contour plots for B20 vs (B30..B60) keeping B10 constant.
    Organizes plots into subdirectories: B10ConstantMaps/B10_value/YCol_vs_XCol/.
    Filenames now include the B10 value: contour_{Zcol}_{B10col}_{B10val}.png

    Args:
        full_df (pd.DataFrame): The complete input DataFrame for Z, N.
        base_output_dir_zn (str): The base output directory for the Z, N combination (e.g., './MiniMaps/Z_N').
    """
    print(f"\n{'=' * 10} Starting B10 Constant Analysis {'=' * 10}")
    b10_const_base_path = os.path.join(base_output_dir_zn, B10_CONST_ANALYSIS_DIR)

    # Check required columns in the full dataframe
    required_cols_for_analysis = [B10_CONST_FIXED_COL, B10_CONST_X_COL,
                                  ENERGY_COL] + B10_CONST_Y_COLS + FLOAT_COLS_TO_PLOT
    missing_cols = [col for col in required_cols_for_analysis if col not in full_df.columns]
    if missing_cols:
        print(f"Error: Cannot perform B10 Constant analysis. DataFrame is missing required columns: {missing_cols}")
        return False

    # Find unique B10 values, dropping NaNs
    unique_b10_values = full_df[B10_CONST_FIXED_COL].dropna().unique()
    unique_b10_values.sort()

    if len(unique_b10_values) == 0:
        print(
            f"Warning: No valid (non-NaN) values found for the constant column '{B10_CONST_FIXED_COL}'. Skipping B10 Constant analysis.")
        return True

    print(f"Found {len(unique_b10_values)} unique values for {B10_CONST_FIXED_COL}.")
    os.makedirs(b10_const_base_path, exist_ok=True)
    print(f"Base directory for this analysis: {b10_const_base_path}")

    overall_success = True
    processed_b10_count = 0

    # --- Loop over each unique B10 value ---
    for b10_val in unique_b10_values:
        # Sanitize b10_val for directory AND filename component
        b10_val_str_safe = sanitize_filename(b10_val)
        b10_subdir_name = f"{B10_CONST_FIXED_COL}_{b10_val_str_safe}"  # e.g., B10_1p5
        # Base directory for THIS specific B10 value
        b10_specific_base_dir = os.path.join(b10_const_base_path, b10_subdir_name)
        os.makedirs(b10_specific_base_dir, exist_ok=True)

        print(f"\n--- Analyzing for constant {B10_CONST_FIXED_COL} = {b10_val} ---")
        print(f"Base output dir for this value: {b10_specific_base_dir}")

        # Filter data for the current B10 value
        df_b10_filtered = full_df[full_df[B10_CONST_FIXED_COL] == b10_val].copy()
        if df_b10_filtered.empty:
            print("  No data found for this B10 value. Skipping.")
            continue

        processed_b10_count += 1
        b10_level_plots_generated = 0
        b10_level_plots_failed = 0

        # --- Loop through the Y-axis columns for this B10 value (B30, B40, etc.) ---
        for y_col in B10_CONST_Y_COLS:
            print(f"  -- Processing Y-axis: {y_col} (X-axis: {B10_CONST_X_COL}) --")

            # *** Define and create the specific subdirectory for this Y vs X plot ***
            plot_subdir_name = f"{y_col}_vs_{B10_CONST_X_COL}"
            specific_plot_dir = os.path.join(b10_specific_base_dir, plot_subdir_name)
            os.makedirs(specific_plot_dir, exist_ok=True)
            print(f"    Outputting plots to: {specific_plot_dir}")

            # Define grouping columns for minimum energy search within this B10 subset
            group_cols = [y_col, B10_CONST_X_COL]
            required_cols = group_cols + [ENERGY_COL]

            if not all(c in df_b10_filtered.columns for c in required_cols):
                print(
                    f"    Skipping {y_col}: Missing required columns {required_cols} in data subset for B10={b10_val}.")
                continue

            # Find minimum energy within the B10 subset for each (y_col, B20) pair
            df_subset_filtered = df_b10_filtered.dropna(subset=group_cols + [ENERGY_COL])
            if df_subset_filtered.empty:
                print(f"    Skipping {y_col}: No valid data after dropping NaNs in {group_cols + [ENERGY_COL]}.")
                continue

            try:
                idx = df_subset_filtered.groupby(group_cols, observed=True)[ENERGY_COL].idxmin()
                min_energy_subset_df = df_subset_filtered.loc[idx].copy()
                print(
                    f"    Found {len(min_energy_subset_df)} min energy points for ({y_col}, {B10_CONST_X_COL}) at {B10_CONST_FIXED_COL}={b10_val}.")
                if min_energy_subset_df.empty:
                    print("    Warning: Minimum energy DataFrame is empty for this combination. Skipping plots.")
                    continue
            except KeyError as ke:
                print(
                    f"    Error finding minimum energy for {y_col} vs {B10_CONST_X_COL} at {B10_CONST_FIXED_COL}={b10_val}: Missing column {ke}")
                overall_success = False
                continue
            except Exception as e:
                print(
                    f"    Error finding minimum energy for {y_col} vs {B10_CONST_X_COL} at {B10_CONST_FIXED_COL}={b10_val}: {e}")
                overall_success = False
                continue

            # --- Generate plots for this (B10_val, y_col) combination ---
            title_suffix_b10 = f" ({y_col} vs {B10_CONST_X_COL}) for {B10_CONST_FIXED_COL}={b10_val}"

            # *** MODIFICATION: Create the filename suffix containing the B10 value ***
            # Format: _{B10_CONST_FIXED_COL}_{b10_val_str_safe}
            # Example: _B10_1p5 or _B10_neg2p0
            filename_suffix_b10 = f"_{B10_CONST_FIXED_COL}_{b10_val_str_safe}"

            for z_col_to_plot in FLOAT_COLS_TO_PLOT:
                if z_col_to_plot not in min_energy_subset_df.columns:
                    print(
                        f"    Skipping plot for '{z_col_to_plot}': Column not found in B10-specific minimum energy data.")
                    b10_level_plots_failed += 1
                    continue

                # *** Call plot helper with the NEW specific_plot_dir AND the NEW filename_suffix ***
                success = generate_contour_plot(
                    plot_data_df=min_energy_subset_df,
                    x_col=B10_CONST_X_COL,
                    y_col=y_col,
                    z_col=z_col_to_plot,
                    plot_dir=specific_plot_dir,  # Use the YCol_vs_XCol directory
                    title_suffix=title_suffix_b10,
                    filename_suffix=filename_suffix_b10  # Pass the B10 value suffix
                )
                if success:
                    b10_level_plots_generated += 1
                else:
                    b10_level_plots_failed += 1
                    overall_success = False

        # --- End of y_col loop ---
        print(f"--- Plotting Summary for {B10_CONST_FIXED_COL} = {b10_val} ---")
        print(f"  Successfully generated: {b10_level_plots_generated} plots")
        print(f"  Failed or skipped:    {b10_level_plots_failed} plots")

    # --- End of b10_val loop ---
    print(f"\nProcessed {processed_b10_count} unique {B10_CONST_FIXED_COL} values.")
    print(f"{'=' * 10} B10 Constant Analysis Complete {'=' * 10}")
    return overall_success


# --- Command Line Argument Parsing and Main Execution (Unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze Parquet fragment: find min energy points for (Bxx, B20) combinations, '
                    'create contour plots, and perform B10-constant analysis.'
    )
    parser.add_argument(
        'NumberOfProtons',
        type=int,
        help='Number of Protons (Z).'
    )
    parser.add_argument(
        'NumberOfNeutrons',
        type=int,
        help='Number of Neutrons (N).'
    )

    args = parser.parse_args()

    z = args.NumberOfProtons
    n = args.NumberOfNeutrons

    # Derive input and output paths
    input_filename = f"{z}_{n}_AllMapFragments.parquet"
    parquet_file_path = os.path.join(INPUT_BASE_DIR, input_filename)
    output_directory_zn = os.path.join(OUTPUT_BASE_DIR, f"{z}_{n}")

    print(f"=== Starting Full Analysis for Z={z}, N={n} ===")
    print(f"Derived Input File: {parquet_file_path}")
    print(f"Derived Base Output Directory: {output_directory_zn}")

    # Create the base output directory for this Z, N combination
    try:
        os.makedirs(output_directory_zn, exist_ok=True)
        print(f"Ensured base output directory exists: {output_directory_zn}")
    except OSError as e:
        print(f"Error: Could not create base output directory '{output_directory_zn}': {e}")
        sys.exit(1)

    # --- Read Parquet Data ONCE ---
    full_dataframe = None
    try:
        print(f"\nReading Parquet file: {parquet_file_path}...")
        if not os.path.exists(parquet_file_path):
            print(f"Error: Input Parquet file not found at '{parquet_file_path}'")
            print("Please ensure the file exists in the correct MapFragments directory.")
            sys.exit(1)
        full_dataframe = pd.read_parquet(parquet_file_path)
        print(f"Successfully read {len(full_dataframe)} rows.")
    except Exception as e:
        print(f"Error reading Parquet file '{parquet_file_path}': {e}")
        sys.exit(1)

    # --- Run First Analysis Type (Y vs B20) ---
    print(f"\n=== Starting Y-Axis vs {X_AXIS_COL} Analysis ===")
    y_vs_b20_success_count = 0
    y_vs_b20_failure_count = 0
    for y_col in POSSIBLE_Y_AXIS_COLS:
        success = analyze_data_for_y_col(full_dataframe, y_col, output_directory_zn)
        if success:
            y_vs_b20_success_count += 1
        else:
            y_vs_b20_failure_count += 1
        print("-" * 50)

    # --- Run Second Analysis Type (B10 Constant) ---
    b10_const_analysis_success = analyze_constant_b10(full_dataframe, output_directory_zn)

    # --- Final Summary ---
    print(f"\n=== Full Analysis Summary for Z={z}, N={n} ===")
    print("\n--- Y-Axis vs B20 Analysis Summary ---")
    print(f"Processed {len(POSSIBLE_Y_AXIS_COLS)} Y-axis columns ({', '.join(POSSIBLE_Y_AXIS_COLS)}).")
    print(f"Successful analyses: {y_vs_b20_success_count}")
    print(f"Failed analyses:     {y_vs_b20_failure_count}")

    print("\n--- B10 Constant Analysis Summary ---")
    if b10_const_analysis_success is False:
        print("B10 Constant analysis encountered errors during execution.")
    elif b10_const_analysis_success is True:
        print("B10 Constant analysis completed (check logs for details on individual plots/values).")

    print(f"\nAll outputs located in base directory: '{output_directory_zn}'")
    print("==============================================")