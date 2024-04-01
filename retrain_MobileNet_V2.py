import cxr_dataset as CXR
import eval_model as E
import model_MobileNet_V2 as M
import os

# Define your potential paths
PATH_TO_IMAGES_OPTIONS = [
    "/kaggle/input/nih-chest-xrays-224-gray/images",
    "/kaggle/input/images"
]

# Function to determine the valid path
def find_valid_path(paths_list):
    for path in paths_list:
        if os.path.exists(path):
            print(f"Using path: {path}")
            return path
    raise FileNotFoundError("None of the paths exist.")

# Try to find a valid path
PATH_TO_IMAGES = find_valid_path(PATH_TO_IMAGES_OPTIONS)
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY)

