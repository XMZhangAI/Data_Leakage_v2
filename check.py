from datasets import load_from_disk, Dataset, DatasetDict

def get_schema(dataset):
    """Helper function to extract schema information from a dataset."""
    return [(name, str(type(dataset[name][0]))) for name in dataset.column_names]

def check_datasets_alignment(dataset_path1, dataset_path2):
    # Load the datasets from the specified paths
    dataset1 = load_from_disk(dataset_path1)
    dataset2 = load_from_disk(dataset_path2)

    if isinstance(dataset1, DatasetDict) and isinstance(dataset2, DatasetDict):
        print("Comparing DatasetDicts...")
        # Check each split in the DatasetDict
        splits1 = dataset1.keys()
        splits2 = dataset2.keys()

        if splits1 != splits2:
            print("The datasets do not have the same splits.")
            print(f"Dataset 1 splits: {splits1}")
            print(f"Dataset 2 splits: {splits2}")
            return

        aligned = True

        for split in splits1:
            schema1 = get_schema(dataset1[split])
            schema2 = get_schema(dataset2[split])

            if schema1 != schema2:
                aligned = False
                print(f"The datasets are not aligned in split '{split}'.")
                print("Dataset 1 fields:")
                for field in schema1:
                    print(f"  {field[0]}: {field[1]}")
                print("Dataset 2 fields:")
                for field in schema2:
                    print(f"  {field[0]}: {field[1]}")

        if aligned:
            print("The datasets are aligned.")
    
    elif isinstance(dataset1, Dataset) and isinstance(dataset2, Dataset):
        print("Comparing Datasets...")
        # Directly compare the two datasets
        schema1 = get_schema(dataset1)
        schema2 = get_schema(dataset2)

        if schema1 != schema2:
            print("The datasets are not aligned.")
            print("Dataset 1 fields:")
            for field in schema1:
                print(f"  {field[0]}: {field[1]}")
            print("Dataset 2 fields:")
            for field in schema2:
                print(f"  {field[0]}: {field[1]}")
        else:
            print("The datasets are aligned.")
    
    else:
        print("The datasets are not of the same type. One is DatasetDict and the other is Dataset.")

if __name__ == "__main__":
    dataset_path1 = "/home/zhangxuanming/DataLeakage_v2/datasets/detect_dataset_all_v3/original_data_contamination_detection_dataset_truncate_epoch2"
    dataset_path2 = "/home/zhangxuanming/DataLeakage_v2/datasets/detect_dataset_all_varient_v3/variant_data_contamination_detection_dataset_truncate_epoch2"

    check_datasets_alignment(dataset_path1, dataset_path2)
