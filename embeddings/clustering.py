import argparse
import pandas as pd
import yaml
import torch
from sklearn.cluster import KMeans
from collections import Counter
from tqdm import tqdm


# Function to read YAML config file
def read_config(config_file):
    """
    Read YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary with configuration parameters.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def calculate_most_common_tokens(col, n=None):
    text = ' '.join(col)
    tokens = text.split()
    length = len(tokens)
    
    token_counts = Counter(tokens)
    most_common_tokens = token_counts.most_common(n)
    token_col, count_col = zip(*most_common_tokens)
    token_col = list(token_col)
    percentage_col = [round(x / length * 100, 2) for x in list(count_col)]

    return token_col, percentage_col


def getRowPercentages(tokens, col):
    row_percentages = []
    for token in tqdm(tokens, desc="Calculating row percentages"):
        rows_with_token = sum(1 for row in col if token in row)
        percentage = round(rows_with_token / len(col) * 100, 2)
        row_percentages.append(percentage)
    return row_percentages


def map_token_athena(tokens, df_concept):
    concept_map = df_concept.set_index('concept_id')['concept_name'].to_dict()
    return [concept_map.get(int(token), None) if token.isdigit() else None for token in tokens]


def getPR(tokens, patient_percentages, overall_df):
    """
    Calculate the precentage ratio for each token in the cluster compared to the overall dataset.

    Args:
        tokens (list): List of tokens in the cluster.
        patient_percentages (list): List of patient percentages for each token in the cluster.
        overall_df (DataFrame): DataFrame with overall token statistics.

    Returns:
        list: List of percentage ratios for each token in the cluster.
    """
    PR = []
    for token, percentage in tqdm(zip(tokens, patient_percentages), desc="Calculating PR", total=len(tokens)):
        overall_percentage = overall_df[('Overall', '% (patients)')][overall_df[('Overall', 'Code')] == token].values
        if overall_percentage.size == 0:
            PR.append(None)
        else:
            overall_percentage = overall_percentage[0]
            if overall_percentage == 0:
                PR.append(None)
            else:
                PR.append(round(percentage / overall_percentage, 2))
    return PR


def getOR(tokens, patient_percentages, overall_df):
    """
    Calculate the odds ratio for each token in the cluster compared to the overall dataset.
    
    Args:
        tokens (list): List of tokens in the cluster.
        patient_percentages (list): List of patient percentages for each token in the cluster.
        overall_df (DataFrame): DataFrame with overall token statistics.
        
    Returns:
        list: List of odds ratios for each token in the cluster.
    """
    OR = []
    for token, percentage in tqdm(zip(tokens, patient_percentages), desc="Calculating OR", total=len(tokens)):
        overall_percentage = overall_df[('Overall', '% (patients)')][overall_df[('Overall', 'Code')] == token].values
        if overall_percentage.size == 0:
            OR.append(None)
        else:
            overall_percentage = overall_percentage[0]
            OR.append(round(((percentage + 0.1) / (100 - percentage + 0.1)) / ((overall_percentage + 0.1) / (100 - overall_percentage + 0.1)), 2))
    return OR


def most_common_tokens(df, cluster_col, text_col, df_concept):
    clusters = sorted(df[cluster_col].unique())

    cluster_dict = []
    cluster_tokens = []
    for cluster in tqdm(clusters, desc="Processing clusters"):
        cluster_df = df[df[cluster_col] == cluster]
        tokens, token_percentages = calculate_most_common_tokens(cluster_df[text_col], n=500)
        cluster_tokens.append(tokens)
        cluster_dict.append({})
        cluster_dict[cluster][(f'Cluster №{cluster}', 'Code')] = tokens
        cluster_dict[cluster][(f'Cluster №{cluster}', 'Name')] = map_token_athena(tokens, df_concept)
        cluster_dict[cluster][(f'Cluster №{cluster}', '% (patients)')] = getRowPercentages(tokens, cluster_df[text_col])
        cluster_dict[cluster][(f'Cluster №{cluster}', '% (tokens)')] = token_percentages

    cluster_tokens_merged = list(set([item for sublist in cluster_tokens for item in sublist]))
    overall_dict = {}
    overall_dict[('Overall', 'Code')] = cluster_tokens_merged
    overall_dict[('Overall', 'Name')] = map_token_athena(cluster_tokens_merged, df_concept)
    overall_dict[('Overall', '% (patients)')] = getRowPercentages(cluster_tokens_merged, df[text_col])

    overall_df = pd.DataFrame(overall_dict).sort_values(by=[('Overall', '% (patients)')], ascending=False)
    overall_df = overall_df.reset_index(drop=True)
    result_df = overall_df.copy() 

    for cluster in tqdm(clusters, desc="Processing clusters"):
        cluster_dict[cluster][(f'Cluster №{cluster}', 'PR')] = getPR(cluster_dict[cluster][(f'Cluster №{cluster}', 'Code')], cluster_dict[cluster][(f'Cluster №{cluster}', '% (patients)')], overall_df)
        cluster_dict[cluster][(f'Cluster №{cluster}', 'OR')] = getOR(cluster_dict[cluster][(f'Cluster №{cluster}', 'Code')], cluster_dict[cluster][(f'Cluster №{cluster}', '% (patients)')], overall_df)
        cluster_df = pd.DataFrame(cluster_dict[cluster]).sort_values(by=[(f'Cluster №{cluster}', 'OR')], ascending=False)
        cluster_df = cluster_df.reset_index(drop=True)
        result_df = pd.concat([result_df, cluster_df], axis=1)
    return result_df


def calculate_features(df):
    """
    Calculate features for each cluster.

    Args:
        df (DataFrame): DataFrame with the data.

    Returns:
        DataFrame: DataFrame with the features for each cluster.
    """
    numerical_features = ['death_age', 'text_len', 'visit_count', 'n_tokens']
    numerical_stats = df.groupby('cluster')[numerical_features].agg(['mean', 'std', 'median', 'min', 'max', 'count']).round(1).reset_index()
    numerical_stats.set_index('cluster', inplace=True)
    return numerical_stats


def calculate_gender_distribution(df):
    """
    Calculate gender distribution for each cluster.

    Args:
        df (DataFrame): DataFrame with the data.

    Returns:
        DataFrame: DataFrame with the gender distribution for each cluster.
    """
    gender_distribution = df.groupby(['cluster', 'gender']).size().unstack(fill_value=0).reset_index().set_index('cluster')
    gender_distribution['total'] = gender_distribution['F'] + gender_distribution['M']
    gender_distribution['F_perc'] = (gender_distribution['F'] / gender_distribution['total'] * 100).round(2)
    gender_distribution['M_perc'] = (gender_distribution['M'] / gender_distribution['total'] * 100).round(2)
    return gender_distribution


def calculate_dead_ratio(df):
    """
    Calculate the ratio of dead patients for each cluster.

    Args:
        df (DataFrame): DataFrame with the data.

    Returns:
        DataFrame: DataFrame with the dead ratio for each cluster.
    """
    df['dead'] = df['death_age'].notnull()
    dead_ratio = df.groupby('cluster')['dead'].mean().reset_index().set_index('cluster')
    dead_ratio['dead'] = (dead_ratio['dead'] * 100).round(2)
    dead_ratio.rename(columns={'dead': 'dead_perc'}, inplace=True)
    return dead_ratio


def get_report(df):
    """
    Generate a report with features for each cluster.

    Args:
        df (DataFrame): DataFrame with the data.

    Returns:
        DataFrame: DataFrame with the report.
    """
    features = calculate_features(df)
    gender_distribution = calculate_gender_distribution(df)
    dead_ratio = calculate_dead_ratio(df)
    report_df = pd.concat([dead_ratio, gender_distribution, features], axis=1)  
    return report_df


def main(config_file, device):
    # Read the config file
    config = read_config(config_file)

    # Load the data
    data = pd.read_csv(config["clustering"]["embeddings_path"])

    data = data.to(device)

    print("Clustering data...")
    kmeans = KMeans(n_clusters=config["clustering"]["k"])
    kmeans_labels = kmeans.fit_predict(data)

    data["cluster"] = kmeans_labels
    data.to_csv(config["clustering"]["output_path"])

    print("Calculating most common tokens...")
    df_concept = pd.read_csv(config["clustering"]["athena_concept_path"], sep=config["clustering"]["athena_concept_sep"])
    common_tokens = most_common_tokens(data, 'cluster', 'text', df_concept)
    common_tokens.to_csv(config["clustering"]["common_tokens_path"])

    print("Generating report...")

    report = get_report(data)
    report.to_csv(config["clustering"]["report_path"])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Using CPU!")
        quit()
    print("Using device:", device)

    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Load and process data from CSV files."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    args = parser.parse_args()

    # Run the main function with the config file provided
    main(args.config, device)
