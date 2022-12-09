import numpy as np
import torch
from torch.nn import Sequential, ReLU, MSELoss, Dropout, Conv1d, Linear
from torch.nn.utils import prune
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR

from lib.make_targets.fine_targets import load_esm, load_prottrans, load_seqvec
from lib.prepare_dataset.dataloaders import fused_flattened_dataset, autoencoder_dataloader
from lib.prepare_dataset.folding import load_folding, Folding
from lib.util.anchor_util import FUSE_TARGETS_DIR
from lib.util.dataset_util import InclusionRandomSampler, ExclusionRandomSampler
from lib.util.device_util import model_device, dtype
from lib.util.experiment_util import experiment, avg_fn
from lib.util.log_util import pretty_tqdm, LearningLogs, logger
from lib.util.network_util import RandomReductionLinear, Autoencoder
from lib.util.torch_util import load_torch, save_torch


@experiment(FUSE_TARGETS_DIR)
def fuse_targets():
    model_dir = FUSE_TARGETS_DIR
    name = 'Fuse targets'

    logger.info(f'Running {name}')

    dataset = fused_flattened_dataset(
        load_esm(),
        load_prottrans(),
        load_seqvec(),
        repr_num=65536
    )

    # dataset = fused_aligned_dataset(
    #     load_esm(),
    #     load_prottrans(),
    #     load_seqvec(),
    #     seq_length='nearest'
    # )

    logger.info(f'Repr num : {dataset.repr_num}')
    # logger.info(f'Seq length: {dataset.seq_length}')

    folding: Folding = load_folding()

    total_list = folding.no_test_targets
    validate_list = folding.validate_targets
    validate_set = set(validate_list)

    model = Autoencoder(
        encoder=Sequential(
            Dropout(),
            RandomReductionLinear(
                input_width=dataset.repr_num,
                output_width=10000,
                subset_size=100,
                generator=torch.default_generator
            ),
            ReLU(),
        ),
        decoder=Sequential(
            RandomReductionLinear(
                input_width=10000,
                output_width=dataset.repr_num,
                subset_size=100,
                generator=torch.default_generator,
                with_replacement=True
            ),
        ),
    ).to(model_device, dtype=dtype)

    # model = Autoencoder(
    #     encoder=Sequential(
    #         Dropout(),
    #         Conv1d(
    #             in_channels=dataset.repr_num,
    #             out_channels=2000,
    #             kernel_size=3,
    #             padding='same'
    #         ),
    #         ReLU(),
    #         MaxPool1d(kernel_size=2)
    #     ),
    #     decoder=Sequential(
    #         Upsample(scale_factor=2),
    #         Conv1d(
    #             in_channels=2000,
    #             out_channels=dataset.repr_num,
    #             kernel_size=3,
    #             padding='same'
    #         ),
    #     )
    # ).to(model_device, dtype=dtype)

    loss_fn = MSELoss()

    learning_rate = 1e-3
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    batch_size = 2 ** 4

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
