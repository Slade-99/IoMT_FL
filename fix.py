import pandas as pd

def binarize_label(input_csv, output_csv):
    # Read CSV
    df = pd.read_csv(input_csv)

    # Ensure the label column exists
    if "label" not in df.columns:
        raise ValueError("The CSV file does not contain a 'label' column.")

    # Convert any non-zero value to 1
    df["label"] = df["label"].apply(lambda x: 1 if x != 0 else 0)

    # Save to new CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved new file: {output_csv}")


if __name__ == "__main__":
    input_file = "/home/azwad/Works/IoMT_FL/Dataset/for_thresholding_experiment/test_balanced.csv"
    output_file = "/home/azwad/Works/IoMT_FL/Dataset/for_thresholding_experiment/binary_test_balanced.csv"
    binarize_label(input_file, output_file)
