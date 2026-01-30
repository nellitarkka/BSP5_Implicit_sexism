import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def prepare_task1_data(
    raw_data_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Prepare train/test datasets for Task 1: Implicit Sexism Detection (binary).

    Label mapping:
        Sexist     -> 1
        Not Sexist -> 0
    """

    #Load raw data
    df = pd.read_csv(raw_data_path)

    #Create binary label for Task 1
    df["task1_label"] = (df["label_sexist"].str.lower() == "sexist").astype(int)

    #Keep only relevant columns
    df_task1 = df[["text", "task1_label"]]

    #Train / test split
    train_df, test_df = train_test_split(
        df_task1,
        test_size=test_size,
        stratify=df_task1["task1_label"],
        random_state=random_state
    )

    #Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    #Save processed files
    train_path = output_path / "task1_train.csv"
    test_path = output_path / "task1_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Task 1 data prepared successfully.")
    print(f"Train set saved to: {train_path}")
    print(f"Test set saved to:  {test_path}")
    print("\nClass distribution (train):")
    print(train_df["task1_label"].value_counts(normalize=True))


if __name__ == "__main__":
    prepare_task1_data(
        raw_data_path="data/raw/edos_labelled_aggregated.csv",
        output_dir="data/processed"
    )
