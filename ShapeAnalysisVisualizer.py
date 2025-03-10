import argparse
import json
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def get_display_names(col_names):
    """
    Create a mapping of internal column names to display names.

    Parameters:
    col_names (list): List of all column names

    Returns:
    dict: Mapping of internal column names to display names
    """
    display_names = {}

    # Define display names for specific columns
    # "B" + colnumber + "0"
    for col in col_names:
        display_names[col] = "B" + col[5] + "0"

    # For any column not explicitly mapped, use the original name
    for col in col_names:
        if col not in display_names:
            display_names[col] = col

    print(display_names)

    return display_names


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize nuclear shape analysis data from JSON file.')
    parser.add_argument('json_file', help='Path to the JSON file containing analysis data')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output PNG (default: 300)')
    parser.add_argument('--output', help='Output PNG filename (default: based on input filename)')
    return parser.parse_args()


def load_json_data(filename):
    """Load and parse the JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)


def create_value_count_plots(gs, data, col_names, row_start=0, display_names=None):
    """Create plots of column values vs true counts for each column."""
    # Use the provided display names or fall back to original names
    if display_names is None:
        display_names = {col: col for col in col_names}

    n_cols = len(col_names)
    plot_indices = {}

    # Organize plots in a grid
    n_rows = (n_cols + 2) // 3  # 3 columns of plots

    for i, col_name in enumerate(col_names):
        row = row_start + i // 3
        col = i % 3
        ax = plt.subplot(gs[row, col])

        ax.set_yscale('log')

        # Get the column data
        col_data = data["column_analysis"][col_name]

        # Extract values and counts
        values = [item["value"] for item in col_data]
        counts = [item["count"] for item in col_data]

        # Plot the data
        bars = ax.bar(range(len(values)), counts, color='skyblue', edgecolor='black', linewidth=0.5)

        # Add value labels to x-axis for small number of values
        if len(values) <= 20:
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values, rotation=45, ha='right')
        else:
            # For larger sets, show fewer labels
            step = max(1, len(values) // 10)
            ax.set_xticks(range(0, len(values), step))
            ax.set_xticklabels([values[i] for i in range(0, len(values), step)], rotation=45, ha='right')

        # Add title and labels with display names
        ax.set_title(f'{display_names[col_name]}')
        ax.set_ylabel('Count')

        # Store the axis for later reference
        plot_indices[col_name] = (row, col)

    return plot_indices, row_start + n_rows


def create_heatmap_plots(gs, data, col_names, col2_name, row_start=0, display_names=None):
    """Create heatmaps of other columns vs col2 (col2 always on x-axis)."""
    # Use the provided display names or fall back to original names
    if display_names is None:
        display_names = {col: col for col in col_names}

    # Filter out col2 from the columns to compare
    cols_to_plot = [col for col in col_names if col != col2_name]
    n_cols = len(cols_to_plot)

    # Organize plots in a grid
    n_rows = (n_cols + 2) // 3  # 3 columns of plots

    for i, col_name in enumerate(cols_to_plot):
        row = row_start + i // 3
        col = i % 3
        ax = plt.subplot(gs[row, col])

        # Get the combination data
        try:
            combo_data = data["column2_combinations"]["combinations"][col_name]

            # Create a matrix for the heatmap
            # First identify all unique values for both columns
            col2_values = set()
            col_values = set()
            for combo in combo_data:
                col2_values.add(combo["values"][0])
                col_values.add(combo["values"][1])

            col2_values = sorted(list(col2_values))
            col_values = sorted(list(col_values))

            # Create a mapping of values to indices
            col2_value_to_idx = {val: idx for idx, val in enumerate(col2_values)}
            col_value_to_idx = {val: idx for idx, val in enumerate(col_values)}

            # Initialize the matrix with zeros
            matrix = np.zeros((len(col_values), len(col2_values)))

            # Fill the matrix with counts
            for combo in combo_data:
                col2_val = combo["values"][0]
                col_val = combo["values"][1]
                count = combo["count"]

                matrix[col_value_to_idx[col_val], col2_value_to_idx[col2_val]] = count

            # Plot the heatmap
            im = ax.imshow(matrix, cmap='viridis', aspect='auto', norm=LogNorm(vmin=1, vmax=matrix.max()))

            # Add title and labels using display names
            ax.set_title(f'{display_names[col_name]} vs {display_names[col2_name]}')

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Count')

            # Label axes
            if len(col_values) <= 10:
                ax.set_yticks(range(len(col_values)))
                ax.set_yticklabels(col_values)
            else:
                step = max(1, len(col_values) // 10)
                ax.set_yticks(range(0, len(col_values), step))
                ax.set_yticklabels([col_values[i] for i in range(0, len(col_values), step)])

            if len(col2_values) <= 10:
                ax.set_xticks(range(len(col2_values)))
                ax.set_xticklabels(col2_values, rotation=45, ha='right')
            else:
                step = max(1, len(col2_values) // 10)
                ax.set_xticks(range(0, len(col2_values), step))
                ax.set_xticklabels([col2_values[i] for i in range(0, len(col2_values), step)], rotation=45, ha='right')

            # Use display names for axis labels
            ax.set_xlabel(display_names[col2_name])
            ax.set_ylabel(display_names[col_name])

        except KeyError:
            ax.text(0.5, 0.5, f"No data for {display_names[col_name]} vs {display_names[col2_name]}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)

    return row_start + n_rows


def create_minmax_plots(gs, data, col_names, col2_name, row_start=0, display_names=None):
    """Create plots of min and max values of other columns vs col2."""
    # Use the provided display names or fall back to original names
    if display_names is None:
        display_names = {col: col for col in col_names}

    # Filter out col2 from the columns to compare
    cols_to_plot = [col for col in col_names if col != col2_name]
    n_cols = len(cols_to_plot)

    # Organize plots in a grid
    n_rows = (n_cols + 2) // 3  # 3 columns of plots

    for i, col_name in enumerate(cols_to_plot):
        row = row_start + i // 3
        col = i % 3
        ax = plt.subplot(gs[row, col])

        try:
            # Get the min/max data for this column
            col2_values = []
            min_values = []
            max_values = []
            min_counts = []
            max_counts = []

            for col2_val, cols_info in data["column2_value_analysis"]["values"].items():
                if col_name in cols_info:
                    col2_values.append(float(col2_val))
                    min_values.append(cols_info[col_name]["min"])
                    max_values.append(cols_info[col_name]["max"])
                    min_counts.append(cols_info[col_name]["min_count"])
                    max_counts.append(cols_info[col_name]["max_count"])

            # Sort all lists based on col2_values
            sorted_indices = np.argsort(col2_values)
            col2_values = [col2_values[i] for i in sorted_indices]
            min_values = [min_values[i] for i in sorted_indices]
            max_values = [max_values[i] for i in sorted_indices]
            min_counts = [min_counts[i] for i in sorted_indices]
            max_counts = [max_counts[i] for i in sorted_indices]

            # Plot min and max values
            ax.plot(col2_values, min_values, 'b-', label='Min')
            ax.plot(col2_values, max_values, 'r-', label='Max')

            # Use second y-axis for counts
            ax2 = ax.twinx()
            ax2.plot(col2_values, min_counts, 'bo', alpha=0.3, label='Min Count')
            ax2.plot(col2_values, max_counts, 'ro', alpha=0.3, label='Max Count')

            # Add title and labels using display names
            ax.set_title(f'{display_names[col_name]} Min/Max vs {display_names[col2_name]}')
            ax.set_xlabel(display_names[col2_name])
            ax.set_ylabel('Min/Max Values')
            ax2.set_ylabel('Counts')

            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize='small')

        except KeyError:
            ax.text(0.5, 0.5, f"No min/max data for {display_names[col_name]} vs {display_names[col2_name]}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)

    return row_start + n_rows


def create_summary_table(gs, data, row, col):
    """Create a summary table with file info and statistics."""
    ax = plt.subplot(gs[row, col])
    ax.axis('off')

    # Extract summary data
    file_info = data["summary"]["file_info"]
    stats = data["summary"]["statistics"]

    # Create table data
    table_data = [
        ["File", file_info["filename"]],
        ["File Size", f"{file_info['file_size']} {file_info['file_size_unit']}"],
        ["Total Rows", f"{stats['total_rows']:,}"],
        ["True Values", f"{stats['true_values']:,}"],
        ["Memory Usage", f"{stats['memory_usage_mb']} MB"]
    ]

    # Create and format the table
    table = ax.table(
        cellText=table_data,
        colWidths=[0.3, 0.7],
        loc='center',
        cellLoc='left'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Add title
    ax.set_title("Analysis Summary", pad=20)


def visualize_data(data, output_filename, dpi=300):
    """Create visualizations based on the JSON data."""
    # Get column names (excluding 'flag')
    col_names = list(data["column_analysis"].keys())

    # Get the base column (col2)
    col2_name = data["column2_combinations"]["base_column"]

    # Create display name mapping
    display_names = get_display_names(col_names)

    # Calculate total number of plots and rows needed
    n_cols = len(col_names)
    n_rows_value_plots = (n_cols + 2) // 3
    n_rows_heatmap = ((n_cols - 1) + 2) // 3  # Exclude col2
    n_rows_minmax = ((n_cols - 1) + 2) // 3  # Exclude col2

    # Calculate total rows needed for all visualizations
    total_rows = n_rows_value_plots + n_rows_heatmap + n_rows_minmax + 1  # +1 for summary

    # Create the figure and GridSpec with more spacing
    fig = plt.figure(figsize=(15, 5 * total_rows), dpi=dpi)
    gs = gridspec.GridSpec(total_rows, 3, figure=fig, hspace=0.6, wspace=0.4)

    # Add title with adjusted position
    plt.suptitle(
        f"Nuclear Shape Analysis Visualization\n"
        f"Protons: {data['summary']['file_info']['filename'].split('/')[-1].split('_')[0]}, "
        f"Neutrons: {data['summary']['file_info']['filename'].split('/')[-1].split('_')[1]}",
        fontsize=16, y=0.98
    )

    # Create value count plots
    row_end = 0
    plt.figtext(0.5, 0.96, "Column Value Distributions", ha='center', fontsize=14)
    row_end = create_value_count_plots(gs, data, col_names, row_end, display_names)[1]

    # Create heatmap plots - Use display name for the section title
    plt.figtext(0.5, 0.96 - (row_end / total_rows) * 0.95,
                f"Heatmaps: Other Columns vs {display_names[col2_name]}", ha='center', fontsize=14)
    row_end = create_heatmap_plots(gs, data, col_names, col2_name, row_end, display_names)

    # Create min/max plots - Use display name for the section title
    plt.figtext(0.5, 0.96 - (row_end / total_rows) * 0.95,
                f"Min/Max Analysis: Column Values vs {display_names[col2_name]}", ha='center', fontsize=14)
    row_end = create_minmax_plots(gs, data, col_names, col2_name, row_end, display_names)

    # Create summary table
    create_summary_table(gs, data, row_end, slice(0, 3))

    # Adjust layout manually instead of using tight_layout
    # This avoids the "not compatible with tight_layout" warning
    fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.5, wspace=0.2)

    # Save the figure
    plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')
    print(f"Visualization saved to {output_filename}")


def main():
    # Parse command line arguments
    args = parse_args()

    # Load the JSON data
    data = load_json_data(args.json_file)

    # Determine output filename if not specified
    if args.output:
        output_filename = args.output
    else:
        # Remove .json extension if present and add _viz.png
        if args.json_file.lower().endswith('.json'):
            output_filename = args.json_file[:-5] + '_viz.png'
        else:
            output_filename = args.json_file + '_viz.png'

    # Create visualizations
    visualize_data(data, output_filename, args.dpi)


if __name__ == "__main__":
    main()
