import numpy as np
import torch
from torch.nn import Sequential, Conv1d, ReLU, Dropout, Linear

from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
from torch.nn.utils import prune
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader

from lib.make_drugs.fine_drugs import load_maccs, load_rdkit, load_morgan
from lib.make_dta.load_kiba import load_kiba
from lib.make_targets.fine_targets import load_esm, load_prottrans, load_seqvec
from lib.prepare_dataset.dataloaders import fused_dataset, fused_flattened_dataset, predictor_dataset
from lib.prepare_dataset.folding import load_folding, Folding
from lib.util.anchor_util import PREDICTOR_DIR
from lib.util.dataset_util import ExclusionRandomSampler, InclusionRandomSampler
from lib.util.device_util import model_device, dtype
from lib.util.experiment_util import experiment, avg_fn
from lib.util.log_util import pretty_tqdm, LearningLogs, logger
from lib.util.network_util import Concater
from lib.util.torch_util import load_torch, save_torch


@experiment(PREDICTOR_DIR)
def predictor():
    model_dir = PREDICTOR_DIR
    name = 'Predictor'

    logger.info(f'Running {name}')

    drug_dataset = fused_dataset(
        load_maccs(),
        load_rdkit(),
        load_morgan()
    )
    drug_repr_num = drug_dataset.repr_num
    target_dataset = fused_flattened_dataset(
        load_esm(),
        load_prottrans(),
        load_seqvec(),
        repr_num=10000
    )
    target_repr_num = target_dataset.repr_num
    total_repr_num = drug_repr_num + target_repr_num
    logger.info(f'Total repr num: {total_repr_num}')

    dataset = predictor_dataset(
        drug_dataset=drug_dataset,
        target_dataset=target_dataset,
        minimal_dta=load_kiba(),
        device=model_device,
        dtype=dtype
    )

    folding: Folding = load_folding()

    total_list = folding.no_test_cartesian
    validate_list = folding.validate_pairs
    validate_set = set(validate_list)

    model = Sequential(
        Concater(dim=1),
        Dropout(),
        Linear(total_repr_num, 1000),
        ReLU(),
        Linear(1000, total_repr_num + 1)
    ).to(model_device, dtype=dtype)

    pos_weight = torch.tensor(10.0, device=model_device, dtype=dtype)
    drug_weight = torch.tensor(1e-3, device=model_device, dtype=dtype)
    target_weight = torch.tensor(1e-3, device=model_device, dtype=dtype)

    def loss_fn(y_p, y):
        return (
            drug_weight * mse_loss(y_p[:, :drug_repr_num], y[0]),
            target_weight * mse_loss(y_p[:, drug_repr_num:-1], y[1]),
            binary_cross_entropy_with_logits(y_p[:, -1], y[2], pos_weight=pos_weight)
        )

    learning_rate = 1e-3
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    batch_size = 2 ** 6

    limit = 4096

    for module in model.modules():
        if isinstance(module, Conv1d):
            prune.random_unstructured(module, 'weight', 0.99)
            prune.remove(module, 'weight')
        if isinstance(module, Linear):
            prune.random_unstructured(module, 'weight', 0.99)
            prune.remove(module, 'weight')

    swa_model = AveragedModel(model, avg_fn=avg_fn(0.9))
    swa_scheduler = SWALR(optimizer, swa_lr=learning_rate, anneal_epochs=3, anneal_strategy='linear')

    logs = LearningLogs(model_dir)

    def train():
        train_dataloader = DataLoader(
            dataset=dataset,
            sampler=ExclusionRandomSampler(
                total=total_list,
                excluded=validate_set,
                generator=torch.default_generator,
                limit=limit,
                with_replacement=True
            ),
            batch_size=batch_size,
        )

        acc_train_drug_loss = torch.tensor(0.0, device=model_device, dtype=dtype)
        acc_train_target_loss = torch.tensor(0.0, device=model_device, dtype=dtype)
        acc_train_affinity_loss = torch.tensor(0.0, device=model_device, dtype=dtype)

        model.train()
        for x, y in pretty_tqdm(train_dataloader, desc='Train', unit_scale=batch_size):
            optimizer.zero_grad()

            y_p = model(x)

            drug_loss, target_loss, affinity_loss = loss_fn(y_p, y)
            acc_train_drug_loss += drug_loss.detach()
            acc_train_target_loss += target_loss.detach()
            acc_train_affinity_loss += affinity_loss.detach()

            loss = drug_loss + target_loss + affinity_loss
            loss.backward()

            optimizer.step()

        swa_model.update_parameters(model)
        swa_scheduler.step()

        logs.log_scalar([name, 'Train', 'Drug'], epoch, float(acc_train_drug_loss))
        logs.log_scalar([name, 'Train', 'Target'], epoch, float(acc_train_target_loss))
        logs.log_scalar([name, 'Train', 'Affintity'], epoch, float(acc_train_affinity_loss))

    def validate():
        validate_dataloader = DataLoader(
            dataset=dataset,
            sampler=InclusionRandomSampler(
                included=validate_list,
                generator=torch.default_generator,
                limit=limit,
                with_replacement=True
            ),
            batch_size=batch_size
        )

        acc_validate_drug_loss = torch.tensor(0.0, device=model_device, dtype=dtype)
        acc_validate_target_loss = torch.tensor(0.0, device=model_device, dtype=dtype)
        acc_validate_affinity_loss = torch.tensor(0.0, device=model_device, dtype=dtype)

        swa_model.eval()
        with torch.no_grad():
            for x, y in pretty_tqdm(validate_dataloader, desc='Validate', unit_scale=batch_size):
                y_p = swa_model(x)

                drug_loss, target_loss, affinity_loss = loss_fn(y_p, y)
                acc_validate_drug_loss += drug_loss.detach()
                acc_validate_target_loss += target_loss.detach()
                acc_validate_affinity_loss += affinity_loss.detach()

        logs.log_scalar([name, 'Validate', 'Drug'], epoch, float(acc_validate_drug_loss))
        logs.log_scalar([name, 'Validate', 'Target'], epoch, float(acc_validate_target_loss))
        logs.log_scalar([name, 'Validate', 'Affintity'], epoch, float(acc_validate_affinity_loss))

        return acc_validate_affinity_loss

    def on_start():
        nonlocal epoch, min_epoch, is_done
        content = load_torch(model_dir / 'partial.pt')
        if content is not None:
            epoch = min_epoch = content['epoch']
            model.load_state_dict(content['model'])
            swa_model.load_state_dict(content['swa_model'])
            optimizer.load_state_dict(content['optimizer'])

    def on_minimum():
        save_torch(
            model_dir / 'partial.pt',
            dict(
                epoch=epoch,
                model=model.state_dict(),
                swa_model=swa_model.state_dict(),
                optimizer=optimizer.state_dict()
            )
        )
        logs.log_comment([name, 'Validate'], epoch, 'New minimum')

    window = np.inf
    max_epoch = 100
    epoch = 0
    min_epoch = 0
    is_done = False

    on_start()

    # is_done = True # for manual stopping

    if not is_done:
        min_score = validate()

        while epoch - min_epoch < window and epoch < max_epoch:
            train()
            epoch += 1
            validate_score = validate()
            if validate_score < min_score:
                min_epoch = epoch
                min_score = validate_score
                on_minimum()

    return swa_model
