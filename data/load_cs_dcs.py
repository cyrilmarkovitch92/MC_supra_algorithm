from pathlib import Path
import pickle

DATA_DIR = Path(__file__).parent

def load_cross_sections():

    filename = DATA_DIR / "CS_dico.pkl"

    with open(filename, "rb") as f:
        database = pickle.load(f)

    return database