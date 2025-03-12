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
    shape_check_count = 0
    shape_and_neck_count = 0  # Counter for rows with both conditions true
    shape_without_neck_count = 0  # Counter for rows with ShapeCheck=true and NeckConditionCheck=false
    processed_rows = 0
    memory_per_row = None

    # Create dictionary to store count of True values for each unique value in each column
    column_value_counts = {}
    # Create dictionary to store count of rows with both conditions true for each unique value
    column_value_both_true_counts = {}
    # Create dictionary to store count of rows with ShapeCheck=true and NeckConditionCheck=false
    column_value_shape_without_neck_counts = {}
    # Create dictionary to store all unique values for each column
    column_unique_values = {}

    # Create dictionary to store combinations of column 2 with all other columns
    column2_combinations = defaultdict(lambda: defaultdict(int))
    # Create dictionary to store combinations where both conditions are true
    column2_combinations_both_true = defaultdict(lambda: defaultdict(int))
    # Create dictionary to store combinations where ShapeCheck=true and NeckConditionCheck=false
    column2_combinations_shape_without_neck = defaultdict(lambda: defaultdict(int))

    # Create dictionary to store min/max values for each column per column 2 value (ShapeCheck true)
    column2_value_analysis = defaultdict(lambda: defaultdict(lambda: {
        "min": None,
        "max": None,
        "min_count": 0,
        "max_count": 0
    }))

    # Create dictionary to store min/max values when both conditions are true
    column2_value_both_true_analysis = defaultdict(lambda: defaultdict(lambda: {
        "min": None,
        "max": None,
        "min_count": 0,
        "max_count": 0
    }))

    # Create dictionary to store min/max values when ShapeCheck=true and NeckConditionCheck=false
    column2_value_shape_without_neck_analysis = defaultdict(lambda: defaultdict(lambda: {
        "min": None,
        "max": None,
        "min_count": 0,
        "max_count": 0
    }))

    # Get the first two non-boolean columns
    non_bool_columns = [col for col in column_names if col not in ['ShapeCheck', 'NeckConditionCheck']]
    col1 = non_bool_columns[0] if len(non_bool_columns) > 0 else None
    col2 = non_bool_columns[1] if len(non_bool_columns) > 1 else None

    for col in column_names:
        if col not in ['ShapeCheck', 'NeckConditionCheck']:  # Skip the boolean columns
            column_value_counts[col] = defaultdict(int)
            column_value_both_true_counts[col] = defaultdict(int)
            column_value_shape_without_neck_counts[col] = defaultdict(int)
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

        # Count True values for ShapeCheck
        shape_check_count += chunk['ShapeCheck'].sum()

        # Count rows where both ShapeCheck and NeckConditionCheck are True
        shape_and_neck_count += (chunk['ShapeCheck'] & chunk['NeckConditionCheck']).sum()

        # Count rows where ShapeCheck=true and NeckConditionCheck=false
        shape_without_neck_count += (chunk['ShapeCheck'] & ~chunk['NeckConditionCheck']).sum()

        # Get the subset of rows where ShapeCheck is True
        true_rows = chunk[chunk['ShapeCheck']]

        # Get the subset of rows where both ShapeCheck and NeckConditionCheck are True
        both_true_rows = chunk[chunk['ShapeCheck'] & chunk['NeckConditionCheck']]

        # Get the subset of rows where ShapeCheck=true and NeckConditionCheck=false
        shape_without_neck_rows = chunk[chunk['ShapeCheck'] & ~chunk['NeckConditionCheck']]

        # For each column (except boolean columns)
        for col in column_names:
            if col not in ['ShapeCheck', 'NeckConditionCheck']:
                # Collect all unique values in this chunk
                column_unique_values[col].update(chunk[col].unique())

                # Count occurrences of each unique value in true rows
                counts_chunk = true_rows.groupby([col]).size()
                for val, count in counts_chunk.items():
                    column_value_counts[col][val] += count

                # Count occurrences of each unique value in rows where both conditions are true
                if not both_true_rows.empty:
                    both_true_counts_chunk = both_true_rows.groupby([col]).size()
                    for val, count in both_true_counts_chunk.items():
                        column_value_both_true_counts[col][val] += count

                # Count occurrences of each unique value in rows where ShapeCheck=true and NeckConditionCheck=false
                if not shape_without_neck_rows.empty:
                    shape_without_neck_counts_chunk = shape_without_neck_rows.groupby([col]).size()
                    for val, count in shape_without_neck_counts_chunk.items():
                        column_value_shape_without_neck_counts[col][val] += count

        # Process column_name and get the first two non-boolean columns
        non_bool_columns = [col for col in column_names if col not in ['ShapeCheck', 'NeckConditionCheck']]
        col1 = non_bool_columns[0] if len(non_bool_columns) > 0 else None
        col2 = non_bool_columns[1] if len(non_bool_columns) > 1 else None

        # Process combinations of column 2 with all other columns (including col1)
        if col2 is not None and not true_rows.empty:
            for col in column_names:
                if col not in ['ShapeCheck', 'NeckConditionCheck'] and col != col2:  # Exclude boolean columns and col2 itself
                    # Use pandas groupby to count combinations efficiently
                    combo_counts = true_rows.groupby([col2, col]).size().reset_index(name='count')
                    # Update the counts dictionary
                    for _, row in combo_counts.iterrows():
                        col2_val = row[col2]
                        col_val = row[col]
                        count = row['count']

                        # Update combination counts
                        column2_combinations[col][(col2_val, col_val)] += count

                        # Update min/max analysis for each column 2 value using ALL rows where ShapeCheck is true
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

                    # Count combinations where both conditions are true
                    if not both_true_rows.empty:
                        both_true_combo_counts = both_true_rows.groupby([col2, col]).size().reset_index(name='count')
                        for _, row in both_true_combo_counts.iterrows():
                            col2_val = row[col2]
                            col_val = row[col]
                            count = row['count']

                            # Update combination counts for both true conditions
                            column2_combinations_both_true[col][(col2_val, col_val)] += count

                            # Track min/max for the "both conditions true" case
                            if column2_value_both_true_analysis[col2_val][col]["min"] is None or col_val < column2_value_both_true_analysis[col2_val][col]["min"]:
                                column2_value_both_true_analysis[col2_val][col]["min"] = col_val
                                column2_value_both_true_analysis[col2_val][col]["min_count"] = count
                            elif col_val == column2_value_both_true_analysis[col2_val][col]["min"]:
                                column2_value_both_true_analysis[col2_val][col]["min_count"] += count

                            if column2_value_both_true_analysis[col2_val][col]["max"] is None or col_val > column2_value_both_true_analysis[col2_val][col]["max"]:
                                column2_value_both_true_analysis[col2_val][col]["max"] = col_val
                                column2_value_both_true_analysis[col2_val][col]["max_count"] = count
                            elif col_val == column2_value_both_true_analysis[col2_val][col]["max"]:
                                column2_value_both_true_analysis[col2_val][col]["max_count"] += count

                    # Count combinations where ShapeCheck=true and NeckConditionCheck=false
                    if not shape_without_neck_rows.empty:
                        shape_without_neck_combo_counts = shape_without_neck_rows.groupby([col2, col]).size().reset_index(name='count')
                        for _, row in shape_without_neck_combo_counts.iterrows():
                            col2_val = row[col2]
                            col_val = row[col]
                            count = row['count']

                            # Update combination counts for shape without neck condition
                            column2_combinations_shape_without_neck[col][(col2_val, col_val)] += count

                            # Track min/max for the "shape without neck" case
                            if column2_value_shape_without_neck_analysis[col2_val][col]["min"] is None or col_val < column2_value_shape_without_neck_analysis[col2_val][col]["min"]:
                                column2_value_shape_without_neck_analysis[col2_val][col]["min"] = col_val
                                column2_value_shape_without_neck_analysis[col2_val][col]["min_count"] = count
                            elif col_val == column2_value_shape_without_neck_analysis[col2_val][col]["min"]:
                                column2_value_shape_without_neck_analysis[col2_val][col]["min_count"] += count

                            if column2_value_shape_without_neck_analysis[col2_val][col]["max"] is None or col_val > column2_value_shape_without_neck_analysis[col2_val][col]["max"]:
                                column2_value_shape_without_neck_analysis[col2_val][col]["max"] = col_val
                                column2_value_shape_without_neck_analysis[col2_val][col]["max_count"] = count
                            elif col_val == column2_value_shape_without_neck_analysis[col2_val][col]["max"]:
                                column2_value_shape_without_neck_analysis[col2_val][col]["max_count"] += count

        # Calculate memory usage if not already calculated
        if memory_per_row is None:
            memory_usage = chunk.memory_usage(deep=True).sum()
            memory_per_row = memory_usage / len(chunk)

        # Update row count
        processed_rows += len(chunk)

        # Print progress only every 100 chunks
        if chunk_count % 100 == 0 or i == num_row_groups - 1:
            sys.stdout.write(
                f"\rProcessed chunk {chunk_count}/{num_row_groups} | Rows: {processed_rows:,} of {total_rows:,} | ShapeCheck true: {shape_check_count:,} | Both conditions true: {shape_and_neck_count:,} | Shape without neck: {shape_without_neck_count:,}")
            sys.stdout.flush()

    # Calculate total memory
    total_memory = memory_per_row * processed_rows / (1024 * 1024)  # MB

    # Calculate file size
    file_size_bytes = os.path.getsize(filename)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
    file_size = file_size_gb if file_size_gb >= 1 else file_size_mb
    file_size_unit = "GB" if file_size_gb >= 1 else "MB"

    # Calculate percentage of shape check rows that also have neck condition check
    neck_percentage = (shape_and_neck_count / shape_check_count * 100) if shape_check_count > 0 else 0
    # Calculate percentage of shape check rows that don't have neck condition
    without_neck_percentage = (shape_without_neck_count / shape_check_count * 100) if shape_check_count > 0 else 0

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
                "shape_check_true": shape_check_count,
                "shape_and_neck_true": shape_and_neck_count,
                "shape_without_neck": shape_without_neck_count,
                "neck_condition_percentage": round(neck_percentage, 2),
                "without_neck_percentage": round(without_neck_percentage, 2),
                "memory_usage_mb": round(total_memory, 2)
            }
        },
        "column_analysis": {},
        "column2_combinations": {
            "base_column": col2
        }
    }

    # Add column analysis results
    for col in column_names:
        if col not in ['ShapeCheck', 'NeckConditionCheck']:
            column_data = []

            # Sort all unique values for this column
            for val in sorted(column_unique_values[col]):
                # Get the counts
                count = column_value_counts[col][val]
                both_true_count = column_value_both_true_counts[col][val]
                shape_without_neck_count = column_value_shape_without_neck_counts[col][val]

                # If the column is a quantized float column, convert back to original value
                if 'quantized' in col:
                    orig_val = val * 0.05
                    val_display = round(orig_val, 2)
                else:
                    val_display = val

                percentage = (count / shape_check_count) * 100 if shape_check_count > 0 else 0
                both_true_percentage = (both_true_count / count) * 100 if count > 0 else 0
                shape_without_neck_percentage = (shape_without_neck_count / count) * 100 if count > 0 else 0

                # Convert numpy types to Python native types
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    val = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    val = float(val)

                if isinstance(val_display, (np.integer, np.int64, np.int32)):
                    val_display = int(val_display)
                elif isinstance(val_display, (np.floating, np.float64, np.float32)):
                    val_display = float(val_display)

                # Add data for this value
                column_data.append({
                    "value": val_display,
                    "raw_value": val,  # Keep original value for reference
                    "count": count,
                    "percentage": round(percentage, 2),
                    "both_true_count": both_true_count,
                    "both_true_percentage": round(both_true_percentage, 2),
                    "shape_without_neck_count": shape_without_neck_count,
                    "shape_without_neck_percentage": round(shape_without_neck_percentage, 2)
                })

            # Add this column's data to the results
            analysis_results["column_analysis"][col] = column_data

    # Add column2_combinations analysis to the results
    if col2 is not None:
        analysis_results["column2_combinations"]["combinations"] = {}

        for col, combinations in column2_combinations.items():
            combo_data = []

            # Sort combinations for consistent output
            for (val1, val2), count in sorted(combinations.items()):
                # Get count for both conditions true (0 if not present)
                both_true_count = column2_combinations_both_true[col].get((val1, val2), 0)
                # Get count for shape without neck (0 if not present)
                shape_without_neck_count = column2_combinations_shape_without_neck[col].get((val1, val2), 0)

                # Handle quantized float columns if needed
                if 'quantized' in col2:
                    val1_display = round(val1 * 0.05, 2)
                else:
                    val1_display = val1

                if 'quantized' in col:
                    val2_display = round(val2 * 0.05, 2)
                else:
                    val2_display = val2

                percentage = (count / shape_check_count) * 100 if shape_check_count > 0 else 0
                both_true_percentage = (both_true_count / count) * 100 if count > 0 else 0
                shape_without_neck_percentage = (shape_without_neck_count / count) * 100 if count > 0 else 0

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

                if isinstance(both_true_count, (np.integer, np.int64, np.int32)):
                    both_true_count = int(both_true_count)

                if isinstance(shape_without_neck_count, (np.integer, np.int64, np.int32)):
                    shape_without_neck_count = int(shape_without_neck_count)

                # Add data for this combination
                combo_data.append({
                    "values": [val1_display, val2_display],
                    "raw_values": [val1, val2],
                    "count": count,
                    "percentage": round(percentage, 2),
                    "both_true_count": both_true_count,
                    "both_true_percentage": round(both_true_percentage, 2),
                    "shape_without_neck_count": shape_without_neck_count,
                    "shape_without_neck_percentage": round(shape_without_neck_percentage, 2)
                })

            # Add combination data to the results
            analysis_results["column2_combinations"]["combinations"][col] = combo_data

    # Add column2_value_analysis to the results
    if col2 is not None:
        analysis_results["column2_value_analysis"] = {
            "column": col2,
            "values": {}
        }

        # Add a new section for min/max when both conditions are true
        analysis_results["column2_value_both_true_analysis"] = {
            "column": col2,
            "values": {}
        }

        # Add a new section for min/max when ShapeCheck=true and NeckConditionCheck=false
        analysis_results["column2_value_shape_without_neck_analysis"] = {
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

            # Process ShapeCheck=true data
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

                if isinstance(min_count, (np.integer, np.int64, np.int32)):
                    min_count = int(min_count)

                if isinstance(max_count, (np.integer, np.int64, np.int32)):
                    max_count = int(max_count)

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

            # Process both conditions true data
            both_true_columns_info = {}
            if val2 in column2_value_both_true_analysis:
                for col, minmax_data in column2_value_both_true_analysis[val2].items():
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

                    if isinstance(min_count, (np.integer, np.int64, np.int32)):
                        min_count = int(min_count)

                    if isinstance(max_count, (np.integer, np.int64, np.int32)):
                        max_count = int(max_count)

                    both_true_columns_info[col] = {
                        "min": min_val_display,
                        "max": max_val_display,
                        "raw_min": min_val,
                        "raw_max": max_val,
                        "min_count": min_count,
                        "max_count": max_count
                    }

                analysis_results["column2_value_both_true_analysis"]["values"][str_val2_display] = both_true_columns_info

            # Process ShapeCheck=true and NeckConditionCheck=false data
            shape_without_neck_columns_info = {}
            if val2 in column2_value_shape_without_neck_analysis:
                for col, minmax_data in column2_value_shape_without_neck_analysis[val2].items():
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

                    if isinstance(min_count, (np.integer, np.int64, np.int32)):
                        min_count = int(min_count)

                    if isinstance(max_count, (np.integer, np.int64, np.int32)):
                        max_count = int(max_count)

                    shape_without_neck_columns_info[col] = {
                        "min": min_val_display,
                        "max": max_val_display,
                        "raw_min": min_val,
                        "raw_max": max_val,
                        "min_count": min_count,
                        "max_count": max_count
                    }

                analysis_results["column2_value_shape_without_neck_analysis"]["values"][str_val2_display] = shape_without_neck_columns_info

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