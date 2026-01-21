from sklearn.datasets import fetch_lfw_people
import os

def load_small_subset():
    print("Step 1: Fetching high-resolution LFW subset...")
    # min_faces_per_person=20 ensures we get names with enough samples for a 1:1 test
    dataset = fetch_lfw_people(min_faces_per_person=20, resize=1.4, color=True)
    print(f"Dataset Ready! Loaded {len(dataset.images)} images for {len(dataset.target_names)} people.")
    return dataset

if __name__ == "__main__":
    load_small_subset()
