import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Function to analyze the DataFrame for highest and lowest CER values

def analyze_cer(df):
    # Sort by CER
    sorted_df = df.sort_values(by='CER', ascending=False)
    
    # Get top 10 and bottom 10 entries
    top_10 = sorted_df.head(10)
    bottom_10 = sorted_df.tail(10)
    
    # Filter out None values
    top_10_ground_truth = top_10['ground_truth'].dropna()
    bottom_10_ground_truth = bottom_10['ground_truth'].dropna()

    # Analyze Chinese symbols in top and bottom 10
    top_symbols = Counter(''.join(top_10_ground_truth))
    bottom_symbols = Counter(''.join(bottom_10_ground_truth))
    
    # Get most common symbols
    top_common_symbols = top_symbols.most_common(10)
    bottom_common_symbols = bottom_symbols.most_common(10)
    
    # Analyze prediction lengths
    top_lengths = top_10['prediction'].apply(len)
    bottom_lengths = bottom_10['prediction'].apply(len)
    
    # Print analysis
    print("Top 10 CER Analysis:")
    print("Most common symbols:", top_common_symbols)
    print("Prediction lengths:", top_lengths.describe())
    
    print("\nBottom 10 CER Analysis:")
    print("Most common symbols:", bottom_common_symbols)
    print("Prediction lengths:", bottom_lengths.describe())

    # Additional analysis suggestions
    # - Compare CER with prediction length
    # - Visualize CER distribution
    # - Analyze correlation between CER and specific symbols

# Function to plot histogram of prediction lengths

def plot_prediction_lengths_histogram(df, num_predictions=10):
    # Sort by CER
    sorted_df = df.sort_values(by='CER', ascending=False)
    
    # Get top and bottom entries
    top_entries = sorted_df.head(num_predictions)
    bottom_entries = sorted_df.tail(num_predictions)
    
    # Analyze prediction lengths
    top_lengths = top_entries['prediction'].apply(len)
    bottom_lengths = bottom_entries['prediction'].apply(len)
    
    # Determine the range for the bins
    min_length = min(top_lengths.min(), bottom_lengths.min())
    max_length = max(top_lengths.max(), bottom_lengths.max())
    bins = range(min_length, max_length + 1)

    # Plot histogram with the same bins
    plt.figure(figsize=(10, 6))
    plt.hist(top_lengths, bins=bins, alpha=0.5, label='Top CER Predictions', color='red')
    plt.hist(bottom_lengths, bins=bins, alpha=0.5, label='Bottom CER Predictions', color='blue')
    plt.xlabel('Prediction Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Lengths')
    plt.legend(loc='upper right')
    plt.show()

# Function to extract cases with spaces in the bottom 10 predictions

def extract_cases_with_spaces(df, num_predictions=10):
    # Sort by CER
    sorted_df = df.sort_values(by='CER', ascending=False)
    
    # Get bottom entries
    bottom_entries = sorted_df.tail(num_predictions)
    
    # Filter entries with spaces in ground_truth or prediction
    cases_with_spaces = bottom_entries[bottom_entries['ground_truth'].str.contains(' ') |
                                       bottom_entries['prediction'].str.contains(' ')]
    
    # Extract two cases
    extracted_cases = cases_with_spaces.head(2)
    
    # Print extracted cases
    print("Extracted Cases with Spaces:")
    print(extracted_cases)

# Example usage
# df = pd.read_csv('path_to_res.csv')
# analyze_cer(df)
# plot_prediction_lengths_histogram(df, num_predictions=10)
# extract_cases_with_spaces(df, num_predictions=10) 