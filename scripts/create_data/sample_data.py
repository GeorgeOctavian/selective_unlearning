import pandas as pd

def sample_and_save_csv(input_file, output_file, num_samples):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Check if the number of samples requested is greater than the number of rows in the original CSV
    if num_samples > len(df):
        raise ValueError("Number of samples requested is greater than the number of rows in the CSV.")

    # Sample rows without replacement
    sampled_df = df.sample(n=num_samples, replace=False)

    # Save the sampled DataFrame to a new CSV file
    sampled_df.to_csv(output_file, index=False)


for id in range(5):

    # Example usage
    input_csv_file = f'experiments/data/extraction/lm_extraction_128_{id}.csv'
    output_csv_file = f'experiments/data/extraction/lm_extraction_64_{id}.csv'
    number_of_samples = 64  # Replace this with the desired number of samples

    sample_and_save_csv(input_csv_file, output_csv_file, number_of_samples)