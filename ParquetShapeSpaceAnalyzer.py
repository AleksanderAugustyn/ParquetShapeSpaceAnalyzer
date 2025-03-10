import os
import sys
from collections import defaultdict

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

    # Print final results on new lines
    print("\n\nResults:")
    print(f"Total rows processed: {processed_rows:,}")
    print(f"Number of True values: {t_count:,}")
    print(f"Memory usage of DataFrame: {total_memory:.2f} MB")

    # Output all unique values and their True counts for each column
    for col in column_names:
        if col != 'flag':
            print(f"\nUnique values in {col} and their True counts:")
            print(f"{col:^15}  {'True Count':^10}  {'Percentage':^10}")
            print("-" * 38)

            # Sort all unique values for this column
            for val in sorted(column_unique_values[col]):
                # Get the true count (0 if the value doesn't appear in true rows)
                count = column_value_counts[col][val]

                # If the column is a quantized float column, convert back to original value
                if 'quantized' in col:
                    orig_val = val * 0.05
                    val_display = f"{orig_val:.2f}"
                else:
                    val_display = str(val)

                percentage = (count / t_count) * 100 if t_count > 0 else 0
                print(f"{val_display:^15}  {count:^10,}  {percentage:^10.2f}%")

    # Output combinations of columns 1 and 2
    if col1 is not None and col2 is not None:
        print(f"\nCombinations of unique values in {col1} and {col2} and their True counts:")
        print(f"{col1:^15}  {col2:^15}  {'True Count':^10}  {'Percentage':^10}")
        print("-" * 55)

        # Sort combinations for consistent output
        for (val1, val2), count in sorted(column_pair_counts.items()):
            # Handle quantized float columns if needed
            if 'quantized' in col1:
                val1_display = f"{val1 * 0.05:.2f}"
            else:
                val1_display = str(val1)

            if 'quantized' in col2:
                val2_display = f"{val2 * 0.05:.2f}"
            else:
                val2_display = str(val2)

            percentage = (count / t_count) * 100 if t_count > 0 else 0
            print(f"{val1_display:^15}  {val2_display:^15}  {count:^10,}  {percentage:^10.2f}%")

    # Calculate and display file size
    file_size_bytes = os.path.getsize(filename)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)

    if file_size_gb >= 1:
        print(f"\nParquet file size: {file_size_gb:.2f} GB")
    else:
        print(f"\nParquet file size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
