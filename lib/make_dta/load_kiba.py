import requests

from lib.make_dta.minimal_dta import MinimalDTA
from lib.util.anchor_util import XLSX_KIBA_PATH, PANDAS_KIBA_PATH, KIBA_PATH
from lib.util.load_util import load
from lib.util.log_util import log_func

import pandas as pd
import numpy as np

from lib.util.pandas_util import load_pandas, save_pandas
from lib.util.pickle_util import load_pickle, save_pickle


@log_func
def download_kiba():
    response = requests.get("https://ndownloader.figstatic.com/files/3950161")
    with open(XLSX_KIBA_PATH, 'wb') as f:
        f.write(response.content)


@log_func
@load(PANDAS_KIBA_PATH, loader=load_pandas, saver=save_pandas)
def load_kiba_pandas():
    if not XLSX_KIBA_PATH.exists():
        download_kiba()
    return pd.read_excel(
        XLSX_KIBA_PATH,
        sheet_name="KIBA",
        header=0,
        index_col=0
    )


THRESHOLD = 3.0


@log_func
@load(KIBA_PATH, load_pickle, save_pickle)
def load_kiba():
    df = load_kiba_pandas()
    target_ids = df.columns.values.tolist()
    drug_ids = []
    known_affinities = {}
    for i, row in enumerate(df.itertuples(index=True, name=None)):
        current_drug = row[0]
        for j, cell in enumerate(row[1:]):
            if not np.isnan(cell):
                known_affinities[i, j] = float(cell) < THRESHOLD
        drug_ids.append(current_drug)

    return MinimalDTA(
        drug_ids=dict(enumerate(drug_ids)),
        target_ids=dict(enumerate(target_ids)),
        known_affinities=known_affinities,
    )
