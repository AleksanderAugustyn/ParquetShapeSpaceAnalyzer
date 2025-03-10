import json
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq


def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python ParquetShapeSpaceFullAnalyzer.py number_of_protons number_of_neutrons")
        sys.exit(1)

    # Get number_of_protons and number_of_neutrons arguments
    number_of_protons = sys.argv[1]
    number_of_neutrons = sys.argv[2]

    # Construct filename with single underscore
    filename = f"./ShapeFragments/{number_of_protons}_{number_of_neutrons}_AllShapeFragments.parquet"

    # Construct output filename
    output_filename = f"{number_of_protons}_{number_of_neutrons}_analysis.json"

    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist.")
        sys.exit(1)

    print(f"Processing file: {filename}")

    # Get total number of rows and column names in the Parquet file
    parquet_file = pq.ParquetFile(filename)
    total_rows = parquet_file.metadata.num_rows
    column_names = parquet_file.schema_arrow.names

    # Process file in row groups
    t_count = 0
    processed_rows = 0
    memory_per_row = None

    # Create dictionary to store count of True values for each unique value in each column
    column_value_counts = {}
    # Create dictionary to store all unique values for each column
    column_unique_values = {}

    # Create dictionary to store combinations of columns 1 and 2
    column_pair_counts = defaultdict(int)

    # Create dictionary to store combinations of column 2 with all other columns
    column2_combinations = defaultdict(lambda: defaultdict(int))

    # Create dictionary to store min/max values for each column per column 2 value
    column2_value_analysis = defaultdict(lambda: defaultdict(lambda: {
        "min": None,
        "max": None,
        "min_count": 0,
        "max_count": 0
    }))

    # Get the first two non-flag columns
    non_flag_columns = [col for col in column_names if col != 'flag']
    col1 = non_flag_columns[0] if len(non_flag_columns) > 0 else None
    col2 = non_flag_columns[1] if len(non_flag_columns) > 1 else None

    for col in column_names:
        if col != 'flag':  # Skip the flag column itself
            column_value_counts[col] = defaultdict(int)
            column_unique_values[col] = set()

    # Process each row group
    num_row_groups = parquet_file.num_row_groups
    chunk_count = 0

    for i in range(num_row_groups):
        chunk_count += 1
        # Read the row group
        row_group = parquet_file.read_row_group(i)
        # Convert to pandas DataFrame
        chunk = row_group.to_pandas()

        # Count True values
        t_count += chunk['flag'].sum()

        # Get the subset of rows where flag is True
        true_rows = chunk[chunk['flag']]

        # For each column (except 'flag')
        for col in column_names:
            if col != 'flag':
                # Collect all unique values in this chunk
                column_unique_values[col].update(chunk[col].unique())

                # Count occurrences of each unique value in true rows
                counts_chunk = true_rows.groupby([col]).size()
                for val, count in counts_chunk.items():
                    column_value_counts[col][val] += count

        # Count combinations of columns 1 and 2 in true rows using vectorized operations
        if col1 is not None and col2 is not None and not true_rows.empty:
            # Use pandas groupby to count combinations efficiently
            combo_counts = true_rows.groupby([col1, col2]).size().reset_index(name='count')
            # Update the counts dictionary
            for _, row in combo_counts.iterrows():
                column_pair_counts[(row[col1], row[col2])] += row['count']

        # Process combinations of column 2 with all other columns
        if col2 is not None and not true_rows.empty:
            for col in column_names:
                if col != 'flag' and col != col2:
                    # Use pandas groupby to count combinations efficiently
                    combo_counts = true_rows.groupby([col2, col]).size().reset_index(name='count')
                    # Update the counts dictionary
                    for _, row in combo_counts.iterrows():
                        col2_val = row[col2]
                        col_val = row[col]
                        count = row['count']

                        # Update combination counts
                        column2_combinations[col][(col2_val, col_val)] += count

                        # Update min/max analysis for each column 2 value
                        if column2_value_analysis[col2_val][col]["min"] is None or col_val < column2_value_analysis[col2_val][col]["min"]:
                            column2_value_analysis[col2_val][col]["min"] = col_val
                            column2_value_analysis[col2_val][col]["min_count"] = count
                        elif col_val == column2_value_analysis[col2_val][col]["min"]:
                            column2_value_analysis[col2_val][col]["min_count"] += count

                        if column2_value_analysis[col2_val][col]["max"] is None or col_val > column2_value_analysis[col2_val][col]["max"]:
                            column2_value_analysis[col2_val][col]["max"] = col_val
                            column2_value_analysis[col2_val][col]["max_count"] = count
                        elif col_val == column2_value_analysis[col2_val][col]["max"]:
                            column2_value_analysis[col2_val][col]["max_count"] += count

        # Calculate memory usage if not already calculated
        if memory_per_row is None:
            memory_usage = chunk.memory_usage(deep=True).sum()
            memory_per_row = memory_usage / len(chunk)

        # Update row count
        processed_rows += len(chunk)

        # Print progress only every 100 chunks
        if chunk_count % 100 == 0 or i == num_row_groups - 1:
            sys.stdout.write(f"\rProcessed chunk {chunk_count}/{num_row_groups} | Rows: {processed_rows:,} of {total_rows:,} | True values: {t_count:,}")
            sys.stdout.flush()

    # Calculate total memory
    total_memory = memory_per_row * processed_rows / (1024 * 1024)  # MB

    # Calculate file size
    file_size_bytes = os.path.getsize(filename)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
    file_size = file_size_gb if file_size_gb >= 1 else file_size_mb
    file_size_unit = "GB" if file_size_gb >= 1 else "MB"

    # Create a dictionary to store all analysis results
    analysis_results = {
        "summary": {
            "file_info": {
                "filename": filename,
                "file_size": round(file_size, 2),
                "file_size_unit": file_size_unit
            },
            "statistics": {
                "total_rows": processed_rows,
                "true_values": t_count,
                "memory_usage_mb": round(total_memory, 2)
            }
        },
        "column_analysis": {},
        "combination_analysis": {
            "columns": [col1, col2] if col1 is not None and col2 is not None else []
        }
    }

    count, min_count, max_count = 0, 0, 0

    # Add column analysis results
    for col in column_names:
        if col != 'flag':
            column_data = []

            # Sort all unique values for this column
            for val in sorted(column_unique_values[col]):
                # Get the true count (0 if the value doesn't appear in true rows)
                count = column_value_counts[col][val]

                # If the column is a quantized float column, convert back to original value
                if 'quantized' in col:
                    orig_val = val * 0.05
                    val_display = round(orig_val, 2)
                else:
                    val_display = val

                percentage = (count / t_count) * 100 if t_count > 0 else 0

                # Convert numpy types to Python native types
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    val = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    val = float(val)

                if isinstance(val_display, (np.integer, np.int64, np.int32)):
                    val_display = int(val_display)
                elif isinstance(val_display, (np.floating, np.float64, np.float32)):
                    val_display = float(val_display)

                if isinstance(min_count, (np.integer, np.int64, np.int32)):
                    min_count = int(min_count)

                if isinstance(max_count, (np.integer, np.int64, np.int32)):
                    max_count = int(max_count)

                # Add data for this value
                column_data.append({
                    "value": val_display,
                    "raw_value": val,  # Keep original value for reference
                    "count": count,
                    "percentage": round(percentage, 2)
                })

            # Add this column's data to the results
            analysis_results["column_analysis"][col] = column_data

    # Add combination analysis results
    if col1 is not None and col2 is not None:
        combo_data = []

        # Sort combinations for consistent output
        for (val1, val2), count in sorted(column_pair_counts.items()):
            # Handle quantized float columns if needed
            if 'quantized' in col1:
                val1_display = round(val1 * 0.05, 2)
            else:
                val1_display = val1

            if 'quantized' in col2:
                val2_display = round(val2 * 0.05, 2)
            else:
                val2_display = val2

            percentage = (count / t_count) * 100 if t_count > 0 else 0

            # Convert numpy types to Python native types
            if isinstance(val1, (np.integer, np.int64, np.int32)):
                val1 = int(val1)
            elif isinstance(val1, (np.floating, np.float64, np.float32)):
                val1 = float(val1)

            if isinstance(val2, (np.integer, np.int64, np.int32)):
                val2 = int(val2)
            elif isinstance(val2, (np.floating, np.float64, np.float32)):
                val2 = float(val2)

            if isinstance(val1_display, (np.integer, np.int64, np.int32)):
                val1_display = int(val1_display)
            elif isinstance(val1_display, (np.floating, np.float64, np.float32)):
                val1_display = float(val1_display)

            if isinstance(val2_display, (np.integer, np.int64, np.int32)):
                val2_display = int(val2_display)
            elif isinstance(val2_display, (np.floating, np.float64, np.float32)):
                val2_display = float(val2_display)

            if isinstance(count, (np.integer, np.int64, np.int32)):
                count = int(count)

            # Add data for this combination
            combo_data.append({
                "values": [val1_display, val2_display],
                "raw_values": [val1, val2],  # Keep original values
                "count": count,
                "percentage": round(percentage, 2)
            })

        # Add combination data to the results
        analysis_results["combination_analysis"]["data"] = combo_data

    # Add column2_combinations analysis to the results
    if col2 is not None:
        analysis_results["column2_combinations"] = {
            "base_column": col2,
            "combinations": {}
        }

        for col, combinations in column2_combinations.items():
            combo_data = []

            # Sort combinations for consistent output
            for (val1, val2), count in sorted(combinations.items()):
                # Handle quantized float columns if needed
                if 'quantized' in col2:
                    val1_display = round(val1 * 0.05, 2)
                else:
                    val1_display = val1

                if 'quantized' in col:
                    val2_display = round(val2 * 0.05, 2)
                else:
                    val2_display = val2

                percentage = (count / t_count) * 100 if t_count > 0 else 0

                # Convert numpy types to Python native types
                if isinstance(val1, (np.integer, np.int64, np.int32)):
                    val1 = int(val1)
                elif isinstance(val1, (np.floating, np.float64, np.float32)):
                    val1 = float(val1)

                if isinstance(val2, (np.integer, np.int64, np.int32)):
                    val2 = int(val2)
                elif isinstance(val2, (np.floating, np.float64, np.float32)):
                    val2 = float(val2)

                if isinstance(val1_display, (np.integer, np.int64, np.int32)):
                    val1_display = int(val1_display)
                elif isinstance(val1_display, (np.floating, np.float64, np.float32)):
                    val1_display = float(val1_display)

                if isinstance(val2_display, (np.integer, np.int64, np.int32)):
                    val2_display = int(val2_display)
                elif isinstance(val2_display, (np.floating, np.float64, np.float32)):
                    val2_display = float(val2_display)

                if isinstance(count, (np.integer, np.int64, np.int32)):
                    count = int(count)

                # Add data for this combination
                combo_data.append({
                    "values": [val1_display, val2_display],
                    "raw_values": [val1, val2],
                    "count": count,
                    "percentage": round(percentage, 2)
                })

            # Add combination data to the results
            analysis_results["column2_combinations"]["combinations"][col] = combo_data

    # Add column2_value_analysis to the results
    if col2 is not None:
        analysis_results["column2_value_analysis"] = {
            "column": col2,
            "values": {}
        }

        for val2 in sorted(column2_value_analysis.keys()):
            # Handle quantized float columns if needed
            if 'quantized' in col2:
                val2_display = round(val2 * 0.05, 2)
            else:
                val2_display = val2

            # Convert numpy types to Python native types
            if isinstance(val2, (np.integer, np.int64, np.int32)):
                val2 = int(val2)
            elif isinstance(val2, (np.floating, np.float64, np.float32)):
                val2 = float(val2)

            if isinstance(val2_display, (np.integer, np.int64, np.int32)):
                val2_display = int(val2_display)
            elif isinstance(val2_display, (np.floating, np.float64, np.float32)):
                val2_display = float(val2_display)

            columns_info = {}

            for col, minmax_data in column2_value_analysis[val2].items():
                min_val = minmax_data["min"]
                max_val = minmax_data["max"]
                min_count = minmax_data["min_count"]
                max_count = minmax_data["max_count"]

                # Handle quantized float columns if needed
                if min_val is not None and 'quantized' in col:
                    min_val_display = round(min_val * 0.05, 2)
                else:
                    min_val_display = min_val

                if max_val is not None and 'quantized' in col:
                    max_val_display = round(max_val * 0.05, 2)
                else:
                    max_val_display = max_val

                # Convert numpy types to Python native types
                if isinstance(min_val, (np.integer, np.int64, np.int32)):
                    min_val = int(min_val)
                elif isinstance(min_val, (np.floating, np.float64, np.float32)) and min_val is not None:
                    min_val = float(min_val)

                if isinstance(max_val, (np.integer, np.int64, np.int32)):
                    max_val = int(max_val)
                elif isinstance(max_val, (np.floating, np.float64, np.float32)) and max_val is not None:
                    max_val = float(max_val)

                if isinstance(min_val_display, (np.integer, np.int64, np.int32)):
                    min_val_display = int(min_val_display)
                elif isinstance(min_val_display, (np.floating, np.float64, np.float32)) and min_val_display is not None:
                    min_val_display = float(min_val_display)

                if isinstance(max_val_display, (np.integer, np.int64, np.int32)):
                    max_val_display = int(max_val_display)
                elif isinstance(max_val_display, (np.floating, np.float64, np.float32)) and max_val_display is not None:
                    max_val_display = float(max_val_display)

                if isinstance(count, (np.integer, np.int64, np.int32)):
                    count = int(count)

                columns_info[col] = {
                    "min": min_val_display,
                    "max": max_val_display,
                    "raw_min": min_val,
                    "raw_max": max_val,
                    "min_count": min_count,
                    "max_count": max_count
                }

            str_val2_display = str(val2_display)
            analysis_results["column2_value_analysis"]["values"][str_val2_display] = columns_info

    # Print completion message
    print(f"\n\nAnalysis complete. Saving results to {output_filename}")

    # Define a custom function to handle NumPy types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Convert all NumPy types to standard Python types
    analysis_results = convert_numpy_types(analysis_results)

    # Save results to JSON file
    with open(output_filename, 'w') as f:
        # noinspection PyTypeChecker
        json.dump(analysis_results, f, indent=2)

    print(f"Results saved successfully.")


if __name__ == "__main__":
    main()