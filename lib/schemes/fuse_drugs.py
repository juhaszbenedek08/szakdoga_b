import numpy as np
import torch
from torch.nn import Sequential, ReLU, MSELoss, Dropout, Linear, Sigmoid
from torch.nn.utils import prune
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR

from lib.make_drugs.fine_drugs import load_maccs, load_rdkit, load_morgan
from lib.prepare_dataset.dataloaders import fused_dataset, autoencoder_dataloader
from lib.prepare_dataset.folding import load_folding, Folding
from lib.util.anchor_util import FUSE_DRUGS_DIR
from lib.util.dataset_util import ExclusionRandomSampler, InclusionRandomSampler
from lib.util.device_util import model_device, dtype
from lib.util.experiment_util import experiment, avg_fn
from lib.util.log_util import pretty_tqdm, LearningLogs, logger
from lib.util.network_util import Autoencoder, Trivial
from lib.util.torch_util import load_torch, save_torch


@experiment(FUSE_DRUGS_DIR)
def fuse_drugs():
    model_dir = FUSE_DRUGS_DIR
    name = 'Fuse drugs'

    logger.info(f'Running {name}')

    dataset = fused_dataset(
        load_maccs(),
        load_morgan(),
        load_rdkit()
    )

    folding: Folding = load_folding()

    total_list = folding.no_test_drugs
    validate_list = folding.validate_drugs
    validate_set = set(validate_list)

    model = Autoencoder(
        encoder=Sequential(
            Dropout(0.0),
            Linear(dataset.repr_num, int(dataset.repr_num / 4)),
            Linear(int(dataset.repr_num / 4), dataset.repr_num),
        ),
        decoder=Sequential(
        )
    ).to(model_device, dtype=dtype)

    loss_fn = MSELoss()
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    batch_size = 2 ** 8

    for module in model.modules():
        if isinstance(module, Linear):
            ...
            prune.random_unstructured(module, 'weight', 0.9)
            prune.remove(module, 'weight')
            # module.weight.data = torch.diagflat(torch.ones(module.weight.size(0), device=model_device))

    swa_model = AveragedModel(model, avg_fn=avg_fn(0.9))
    swa_scheduler = SWALR(optimizer, swa_lr=learning_rate, anneal_epochs=3, anneal_strategy='linear')

    logs = LearningLogs(model_dir)

    def train():
        train_dataloader = autoencoder_dataloader(
            dataset=dataset,
            sampler=ExclusionRandomSampler(
                total=total_list,
                excluded=validate_set,
                generator=torch.default_generator
            ),
            batch_size=batch_size,
            device=model_device,
            dtype=dtype
        )

        acc_train_loss = torch.tensor(0.0, device=model_device, dtype=dtype)

        model.train()
        for x, y in pretty_tqdm(train_dataloader, desc='Train', unit_scale=batch_size):
            optimizer.zero_grad()

            h = model.encoder(x)
            y_p = model.decoder(h)

            loss = loss_fn(y_p, y)  # + 1e-6 * L1(h)  + 1e-6 * L2(h)

            acc_train_loss += loss.detach()

            loss.backward()

            optimizer.step()

        swa_model.update_parameters(model)
        swa_scheduler.step()

        logs.log_scalar([name, 'Train'], epoch, float(acc_train_loss))

    def validate():
        validate_dataloader = autoencoder_dataloader(
            dataset=dataset,
            sampler=InclusionRandomSampler(
                included=validate_list,
                generator=torch.default_generator
            ),
            batch_size=batch_size,
            device=model_device,
            dtype=dtype
        )

        acc_validate_loss = torch.tensor(0.0, device=model_device, dtype=dtype)

        swa_model.eval()
        with torch.no_grad():
            for x, y in pretty_tqdm(validate_dataloader, desc='Validate', unit_scale=batch_size):
                y_p = swa_model(x)

                loss = loss_fn(y_p, y)

                acc_validate_loss += loss

        logs.log_scalar([name, 'Validate'], epoch, float(acc_validate_loss))

        return acc_validate_loss

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
