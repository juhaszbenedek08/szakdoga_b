from pathlib import Path

BASE = Path.cwd().parent

LIB = BASE / 'lib'
DATA = BASE / 'data'
DRUGS = DATA / 'drugs'
TARGETS = DATA / 'targets'
AFFINITIES = DATA / 'affinities'
OTHER_DATA = DATA / 'other'
RESULT = DATA / 'results'
MODELS = RESULT / 'models'

SMILES_PATH = DRUGS / 'raw' / 'drug_smiles.pickle'
ROMOL_PATH = DRUGS / 'raw' / 'drug_romol.pickle'
MACCS_RAW_PATH = DRUGS / 'raw' / 'drug_maccs.pickle'
RDKIT_RAW_PATH = DRUGS / 'raw' / 'drug_rdkit.pickle'
MORGAN_RAW_PATH = DRUGS / 'raw' / 'drug_morgan.pickle'
MACCS_PATH = DRUGS / 'maccs.pt'
RDKIT_PATH = DRUGS / 'rdkit.pt'
MORGAN_PATH = DRUGS / 'morgan.pt'

SEQ_PATH = TARGETS / 'raw' / 'target_sequence.pickle'

XLSX_KIBA_PATH = AFFINITIES / 'raw' / 'kiba.xlsx'
PANDAS_KIBA_PATH = AFFINITIES / 'raw' / 'kiba.pandas'
KIBA_PATH = AFFINITIES / 'kiba.pickle'

FOLDING_PATH = OTHER_DATA / 'folding.pickle'


def model_dir(base_path: Path):
    return base_path, base_path / 'logs', base_path / 'model.pt', base_path / 'partial.pt'


STEP_1 = MODELS / 'step_1'

REDUCE_ESM_DIR = MODELS / 'reduce_esm'
PREDICTOR_DIR = MODELS / 'predictor'
FUSE_TARGETS_DIR = MODELS / 'fuse_targets'
FUSE_DRUGS_DIR = MODELS / 'drugs'