# Maintained for the UARD source preview. See THIRD_PARTY_NOTICES.md
# and docs/SOURCE_CHANGES.md for provenance and release modifications.
'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=dataset_opt.get('num_workers', 0), pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    if dataset_opt['name'] in ('QXS', 'SAR2Opt'):
        from uard_io import dataset_from_options
        dataset = dataset_from_options(dataset_opt, augment=(phase == 'train'))
        logging.getLogger('base').info('Dataset [%s - %s]: %d pairs', dataset_opt['name'], phase, len(dataset))
        return dataset
    if dataset_opt['name']=="SAR_EO":
        from data.LRHR_dataset import SAR2EODataset as D
    if dataset_opt['name']=="SEN12":
        from data.LRHR_dataset import SAR2EODataset as D


    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR'),
                opt=dataset_opt
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
