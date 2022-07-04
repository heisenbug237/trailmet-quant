import os
import torch
from copy import deepcopy
import logging
from collections import OrderedDict

import lapq.data_loaders as datasets
# import distiller
# import distiller.apputils as apputils
# import distiller.apputils.image_classifier as classifier
# import distiller.quantization.ptq_coordinate_search as lapq

msglogger = logging.getLogger()

def load_data(args, fixed_subset=False, sequential=False, load_train=True, load_val=True, load_test=True):
    test_only = not load_train and not load_val

    train_loader, val_loader, test_loader, _ = datasets.load_data(args.dataset, args.arch,
                              os.path.expanduser(args.data), args.batch_size,
                              args.workers, args.validation_split, args.deterministic,
                              args.effective_train_size, args.effective_valid_size, args.effective_test_size,
                              fixed_subset, sequential, test_only)
    if test_only:
        msglogger.info('Dataset sizes:\n\ttest=%d', len(test_loader.sampler))
    else:
        msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                       len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    loaders = (train_loader, val_loader, test_loader)
    flags = (load_train, load_val, load_test)
    loaders = [loaders[i] for i, flag in enumerate(flags) if flag]
    
    if len(loaders) == 1:
        # Unpack the list for convenience
        loaders = loaders[0]
    return loaders


def image_classifier_ptq_lapq(model, criterion, loggers, args):
    args = deepcopy(args)

    effective_test_size_bak = args.effective_test_size
    args.effective_test_size = args.lapq_eval_size
    eval_data_loader = load_data(args, load_train=False, load_val=False, load_test=True, fixed_subset=True)

    args.effective_test_size = effective_test_size_bak
    test_data_loader = load_data(args, load_train=False, load_val=False, load_test=True)

    model = model.eval()
    device = next(model.parameters()).device

    if args.lapq_eval_memoize_dataloader:
        images_batches = []
        targets_batches = []
        for images, targets in eval_data_loader:
            images_batches.append(images.to(device))
            targets_batches.append(targets.to(device))
        memoized_data_loader = [(torch.cat(images_batches), torch.cat(targets_batches))]
    else:
        memoized_data_loader = None

    def eval_fn(model):
        if memoized_data_loader:
            loss = 0
            for images, targets in memoized_data_loader:
                outputs = model(images)
                loss += criterion(outputs, targets).item()
            loss = loss / len(memoized_data_loader)
        else:
            _, _, loss = classifier.test(eval_data_loader, model, criterion, loggers, None, args)
        return loss