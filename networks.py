# Maintained for the UARD source preview. See THIRD_PARTY_NOTICES.md
# and docs/SOURCE_CHANGES.md for provenance and release modifications.
import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)



def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt):
    seed = opt['seed']
    model_opt = opt['model']
    # if model_opt['which_model_G'] == 'ddpm':
    #     from .ddpm_modules import diffusion, unet
    if model_opt['which_model_G'] == 'sr3':
        from .sr3_modules import diffusion, unet
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups']=32
    uard_opt = model_opt.get('uard') or {}
    use_uard = uard_opt.get('enabled', False)
    mode = uard_opt.get('mode', 'full')
    if use_uard and opt['stage'] == 0:
        from .sr3_modules.uard import AnchorPretraining
        netG = AnchorPretraining(None, model_opt['diffusion']['image_size'],
                                 channels=model_opt['diffusion']['channels'], opt=opt)
        if opt['phase'] == 'train':
            init_weights(netG, init_type='orthogonal')
        if opt['gpu_ids'] and opt['distributed']:
            netG = nn.DataParallel(netG)
        return netG
    if use_uard:
        if model_opt['unet']['condition_ch'] != uard_opt.get('sar_channels', 3):
            raise ValueError('UARD condition_ch must equal sar_channels')
        if model_opt['unet']['in_channel'] != model_opt['diffusion']['channels'] or model_opt['unet']['out_channel'] != model_opt['diffusion']['channels']:
            raise ValueError('UARD U-Net input/output channels must equal optical channels')
        if model_opt['unet']['inner_channel'] != 64 or list(model_opt['unet']['channel_multiplier']) != [1, 2, 4, 8, 16]:
            raise ValueError('The inherited E3Diff CPEN requires inner_channel=64 and multipliers=[1,2,4,8,16]')

    condition_ch = model_opt['unet']['condition_ch']
    if use_uard:
        conditioning = uard_opt.get('conditioning', {
            'full': 'sar_anchor_uncertainty', 'residual_uniform': 'sar_anchor_uncertainty',
            'anchor_condition': 'sar_anchor', 'uncertainty_only': 'sar_uncertainty'}[mode])
        if 'anchor' in conditioning:
            condition_ch += model_opt['diffusion']['channels']
        if 'uncertainty' in conditioning:
            condition_ch += 1
    model = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size'],
        condition_ch=condition_ch,
        dual_head=use_uard and mode in ('full', 'residual_uniform'),
        portable_pool=uard_opt.get('portable_pool', False)
    )
    diffusion_class = diffusion.GaussianDiffusion
    if use_uard:
        from .sr3_modules.uard import UARDiffusion
        diffusion_class = UARDiffusion
    netG = diffusion_class(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule'][opt['phase']],
        xT_noise_r = opt['xT_noise_r'],
        seed = seed,
        opt=opt
    )
    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG
