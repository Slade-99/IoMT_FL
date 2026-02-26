import torch
import os
import joblib
import json
from Models.model import DAE
from Preprocessing.ph0_th_loaders.data import get_dataloaders
from Phase_0_Baselines.Thresholding import engine


DATA_PATH = r"/home/azwad/Works/IoMT_FL/Dataset/for_thresholding_experiment"
ENCODER_PATH = r"/home/azwad/Works/IoMT_FL/Dataset/after_scaling_encoding"
TRAINING_OUTPUT_DIR = "/home/azwad/Works/IoMT_FL/Results/Thresholding"
TEST_OUTPUT_DIR = "/home/azwad/Works/IoMT_FL/Results/Thresholding"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


MODEL_PATH = os.path.join(TRAINING_OUTPUT_DIR, 'best_model.pth')
THRESHOLD_PATH = os.path.join(TRAINING_OUTPUT_DIR, 'threshold.txt')
ENCODER_FILE = os.path.join(ENCODER_PATH, 'label_encoder.joblib')

BATCH_SIZE = 1024


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = engine.setup_logging(log_file=os.path.join(TEST_OUTPUT_DIR, 'test.log'))
logger.info(f"Using device: {DEVICE}")

# --- 3. Load Data, Encoder, and Threshold ---
logger.info("Loading dataloaders...")
# We only need the test loader and input_dim
loaders, input_dim = get_dataloaders(base_path=DATA_PATH, batch_size=BATCH_SIZE)
if loaders is None:
    logger.error("Failed to load dataloaders. Exiting.")
    exit()
test_loader = loaders['test_balanced']
logger.info(f"Using 'test_balanced' loader with {len(test_loader.dataset)} samples.")

logger.info("Loading label encoder and threshold...")
try:
    label_encoder = joblib.load(ENCODER_FILE)
    with open(THRESHOLD_PATH, 'r') as f:
        threshold = float(f.read())
    
    # Find benign_label
    try:
        benign_label = int(label_encoder.transform(['Benign'])[0])
    except ValueError:
        benign_label = int(label_encoder.transform(['benign'])[0])

    logger.info(f"Loaded threshold: {threshold}")
    logger.info(f"Loaded benign label: {benign_label}")
    logger.info(f"All classes: {label_encoder.classes_}")

except FileNotFoundError as e:
    logger.error(f"Error: Missing required file: {e.filename}")
    exit()

# --- 4. Load Model ---
logger.info(f"Loading model from {MODEL_PATH}...")
# We don't need noise_factor or dropout for testing
model = DAE(input_dim=input_dim, latent_dim=16, dropout_p=0).to(DEVICE)
model = engine.load_checkpoint(model, MODEL_PATH, DEVICE)
model.eval()

# --- 5. Run Evaluation ---
logger.info("--- Starting Evaluation on 'test_balanced' ---")
results = engine.test_model(
    model, 
    test_loader, 
    DEVICE, 
    threshold, 
    benign_label, 
    label_encoder.classes_
)

# --- 6. Log and Save Results ---
logger.info("\n--- TEST RESULTS ---")

# Log main metrics
main_metrics = {k: v for k, v in results.items() if k != 'Per_Attack_Recall'}
for metric, value in main_metrics.items():
    logger.info(f"{metric:<20}: {value:.6f}")

# Log per-class recall
logger.info("\n--- PER-ATTACK-CLASS RECALL ---")
per_class_recall = results['Per_Attack_Recall']
for class_name, recall in per_class_recall.items():
    if isinstance(recall, float):
        logger.info(f"{class_name:<20}: {recall:.6f}")
    else:
        logger.info(f"{class_name:<20}: {recall}")
        
# Save results to a JSON file for easy access
results_path = os.path.join(TEST_OUTPUT_DIR, 'test_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)
    
logger.info(f"\nTest results saved to {results_path}")
print(f"Testing complete. Results and logs saved to {TEST_OUTPUT_DIR}")