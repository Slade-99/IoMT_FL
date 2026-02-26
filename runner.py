import os
import logging
import traceback
from datetime import datetime

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# ------------------------------------------------------------------
# 🔧 HARD-CODE YOUR NOTEBOOK PATHS HERE
# ------------------------------------------------------------------

NOTEBOOKS = [
    "/home/azwad/Works/IoMT_FL/Revised_Implementation/Scalable/phase1_BlockFL_attack_10.ipynb",
    "/home/azwad/Works/IoMT_FL/Revised_Implementation/Scalable/phase1_BlockFL_attack_100.ipynb",
    "/home/azwad/Works/IoMT_FL/Revised_Implementation/Scalable/phase1_Bulyan_attack_10.ipynb",
    "/home/azwad/Works/IoMT_FL/Revised_Implementation/Scalable/phase1_Bulyan_attack_100.ipynb",
    "/home/azwad/Works/IoMT_FL/Revised_Implementation/Scalable/phase1_SecureFL_attack_10.ipynb",
    "/home/azwad/Works/IoMT_FL/Revised_Implementation/Scalable/phase1_SecureFL_attack_100.ipynb",
    "/home/azwad/Works/IoMT_FL/Revised_Implementation/Scalable/phase1_vanilla_attack_10.ipynb",
    "/home/azwad/Works/IoMT_FL/Revised_Implementation/Scalable/phase1_vanilla_attack_100.ipynb",
    "/home/azwad/Works/IoMT_FL/Revised_Implementation/Scalable/phase2_proposed_final_attack_10.ipynb",
    "/home/azwad/Works/IoMT_FL/Revised_Implementation/Scalable/phase2_proposed_final_attack_100.ipynb"
]

KERNEL_NAME = "python3"     # must match IoMT_FL env kernel
TIMEOUT = None             # seconds per cell
SAVE_EXECUTED = False       # set True if you want *_executed.ipynb saved

LOG_FILE = f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Execution Logic
# ------------------------------------------------------------------

def execute_notebook(path):
    logger.info(f"\n🚀 Running: {path}")

    if not os.path.exists(path):
        logger.error(f"❌ File not found: {path}")
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(
            nb,
            kernel_name=KERNEL_NAME,
            timeout=TIMEOUT,
            allow_errors=False,   # stop notebook on first error
        )

        client.execute()

        logger.info(f"✅ SUCCESS: {path}")

        if SAVE_EXECUTED:
            new_path = path.replace(".ipynb", "_executed.ipynb")
            with open(new_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

        return True

    except CellExecutionError as e:
        logger.error(f"💥 CELL ERROR in {path}")
        logger.error(str(e))

    except Exception:
        logger.error(f"💥 UNKNOWN ERROR in {path}")
        logger.error(traceback.format_exc())

    return False


# ------------------------------------------------------------------
# Batch Run
# ------------------------------------------------------------------

def main():
    logger.info("========== Notebook Batch Run Started ==========")

    success, fail = 0, 0

    for nb in NOTEBOOKS:
        ok = execute_notebook(nb)
        if ok:
            success += 1
        else:
            fail += 1
            logger.info("➡ Moving to next notebook...")

    logger.info("\n========== SUMMARY ==========")
    logger.info(f"Total   : {len(NOTEBOOKS)}")
    logger.info(f"Success : {success}")
    logger.info(f"Failed  : {fail}")
    logger.info("================================")


if __name__ == "__main__":
    main()