from functools import wraps
from typing import Any, Callable
import pandas as pd
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from lib.make_dta.load_kiba import load_kiba
from lib.util.anchor_util import ROMOL_PATH, SMILES_PATH, DRUGS
from lib.util.generate_util import batch_generate, safe_generate
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
import numpy as np

from lib.util.load_util import load
from lib.util.log_util import log_func


@batch_generate(
    load_source_reprs=lambda: load_kiba().drug_ids,
    path=SMILES_PATH,
    batch_size=100
)
def load_smiles(batch_dict: dict[int, Any]):
    id_list = list(batch_dict.values())

    drug_provider = new_client.molecule \
        .filter(molecule_chembl_id__in=id_list) \
        .only('molecule_chembl_id', 'molecule_structures')
    drug_records = list(drug_provider)
    drug_df = pd.DataFrame.from_records(drug_records, index='molecule_chembl_id')

    return safe_generate(
        source_reprs=...,
        generator=lambda drug_id: drug_df.loc[drug_id, 'molecule_structures']['canonical_smiles'],
        name='smiles',
        ids=load_kiba().drug_ids,
        with_bar=False
    )


@load(ROMOL_PATH)
def load_romol():
    return safe_generate(
        source_reprs=load_smiles(),
        generator=Chem.MolFromSmiles,
        name='romol',
        ids=load_kiba().drug_ids,
        with_bar=True
    )


def romol_representation(name: str):
    def outer(generator: Callable[[Any], np.ndarray]):
        @log_func
        @load(DRUGS / 'raw' / f'{name}.pickle')
        @wraps(generator)
        def inner():
            return safe_generate(
                source_reprs=load_romol(),
                generator=generator,
                name=name,
                ids=load_kiba().drug_ids,
                with_bar=True
            )

        return inner

    return outer


@romol_representation('maccs')
def load_maccs_raw(molecule):
    return np.array(GenMACCSKeys(molecule).ToList())


@romol_representation('rdkit')
def load_rdkit_raw(molecule):
    return np.array(RDKFingerprint(molecule, fpSize=2048).ToList())


@romol_representation('morgan')
def load_morgan_raw(molecule):
    return np.array(GetMorganFingerprintAsBitVect(molecule, radius=2).ToList())


raw_drug_repr_loaders = [
    load_maccs_raw,
    load_rdkit_raw,
    load_morgan_raw
]
