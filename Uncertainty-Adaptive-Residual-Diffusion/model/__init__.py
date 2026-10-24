# Maintained for the UARD source preview. See THIRD_PARTY_NOTICES.md
# and docs/SOURCE_CHANGES.md for provenance and release modifications.
import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
