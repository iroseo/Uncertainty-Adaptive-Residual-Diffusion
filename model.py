# Maintained for the UARD source preview. See THIRD_PARTY_NOTICES.md
# and docs/SOURCE_CHANGES.md for provenance and release modifications.
import logging
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
import model.lr_scheduler as lr_scheduler



class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        uard_cfg = opt['model'].get('uard') or {}
        if uard_cfg.get('enabled'):
            if opt['stage'] not in (0, 1, 2):
                raise ValueError('UARD stage must be 0, 1 or 2')
            if opt['model']['finetune_norm']:
                raise ValueError('finetune_norm is not supported for UARD')
            if opt['stage'] == 0 and any(opt['loss_w'].get(k, 0) != 0 for k in ('lpips_w', 'fft_w', 'lambda_gan')):
                raise ValueError('Stage 0 only permits the anchor Laplace likelihood')
            if opt['stage'] == 2 and not opt['path']['resume_state']:
                raise ValueError('UARD Stage 2 requires a matching Stage-1 checkpoint in path.resume_state')
            if opt['stage'] == 1 and not opt['path']['resume_state']:
                if not uard_cfg.get('anchor_checkpoint'):
                    raise ValueError('Stage 1 requires model.uard.anchor_checkpoint from Stage 0')
                if not os.path.isfile(uard_cfg['anchor_checkpoint']):
                    raise FileNotFoundError('Set model.uard.anchor_checkpoint to an existing Stage-0 *_gen.pth file')
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()

        # 将freq初始化为0
        # for k, v in self.netG.named_parameters():
        #     # print(k)
        #     if 'freq' in k:
        #         v.data.zero_()
        #         # print(v)
        #         print(f'setting {k} to zeros')



        self.lpips_loss = None
        self.net_disc = None
        self.freq_loss = None
        self.optimizer_disc = None
        if opt['phase'] == 'train' and opt['loss_w']['lpips_w'] > 0:
            import lpips
            self.lpips_loss = lpips.LPIPS(net='vgg', spatial=True).to(self.device).eval()
            self.lpips_loss.requires_grad_(False)
        if opt['phase'] == 'train' and opt['stage'] == 2 and opt['loss_w']['lambda_gan'] > 0:
            import vision_aided_loss
            self.net_disc = vision_aided_loss.Discriminator(
                cv_type='clip', loss_type='multilevel_sigmoid_s', device=str(self.device)).to(self.device)
            self.net_disc.cv_ensemble.requires_grad_(False)
            self.net_disc.train()
            self.optimizer_disc = torch.optim.Adam(
                (p for p in self.net_disc.parameters() if p.requires_grad), lr=opt['train']['optimizer']['lr'])
        if opt['phase'] == 'train' and opt['loss_w']['fft_w'] > 0:
            from model.sr3_modules.focal_frequency_loss import FocalFrequencyLoss
            self.freq_loss = FocalFrequencyLoss(loss_weight=opt['loss_w']['fft_w'], alpha=1.0)

        self.set_new_noise_schedule(
            opt['model']['beta_schedule'][self.opt['phase']], schedule_phase=self.opt['phase'], 
            num_train_timesteps = opt['model']['beta_schedule']['train']['n_timestep'])
        
        uard_cfg = opt['model'].get('uard') or {}
        if uard_cfg.get('enabled') and opt['stage'] == 2 and not opt['path']['resume_state']:
            raise ValueError('UARD Stage 2 requires a matching Stage-1 checkpoint in path.resume_state')
        if uard_cfg.get('enabled') and opt['stage'] == 0:
            if any(opt['loss_w'].get(k, 0) != 0 for k in ('lpips_w', 'fft_w', 'lambda_gan')):
                raise ValueError('Stage 0 only permits the anchor Laplace likelihood')
        self.load_network()
        if uard_cfg.get('enabled') and opt['stage'] in (1, 2):
            network = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
            if not opt['path']['resume_state']:
                anchor_path = uard_cfg.get('anchor_checkpoint')
                if not anchor_path:
                    raise ValueError('Stage 1 requires model.uard.anchor_checkpoint from Stage 0')
                network.load_anchor_checkpoint(anchor_path)
            if not bool(network.anchor_pretrained):
                raise ValueError('Diffusion checkpoint has no validated Stage-0 anchor')

        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = [p for p in self.netG.parameters() if p.requires_grad]

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()

        self.print_network()
        if self.opt['phase'] == 'train':
            self.setup_schedulers()
            if hasattr(self, 'pending_resume'):
                state = self.pending_resume
                self.optG.load_state_dict(state['optimizer'])
                if state.get('scheduler') is not None:
                    self.scheduler.load_state_dict(state['scheduler'])
                if self.net_disc is not None and state.get('discriminator') is not None:
                    self.net_disc.load_state_dict(state['discriminator'])
                    self.optimizer_disc.load_state_dict(state['optimizer_disc'])
                if state.get('rng_cpu') is not None:
                    torch.set_rng_state(state['rng_cpu'].cpu())
                if torch.cuda.is_available() and state.get('rng_cuda') is not None:
                    torch.cuda.set_rng_state_all([v.cpu() for v in state['rng_cuda']])


    def setup_schedulers(self, lastepoch=-1):
        """Set up scheduler."""
        train_opt = self.opt['train']
        scheduler_options = dict(train_opt['scheduler'])
        scheduler_type = scheduler_options.pop('type')
        self.scheduler_type = scheduler_type
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheduler=lr_scheduler.MultiStepRestartLR(self.optG,
                                                    **scheduler_options, last_epoch=lastepoch)
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheduler=lr_scheduler.CosineAnnealingRestartLR(
                        self.optG, **scheduler_options, last_epoch=lastepoch)
        elif scheduler_type == 'CosineAnnealingWarmupRestarts':
            self.scheduler=lr_scheduler.CosineAnnealingWarmupRestarts(
                        self.optG, **scheduler_options, last_epoch=lastepoch)
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            self.scheduler=lr_scheduler.CosineAnnealingRestartCyclicLR(
                        self.optG, **scheduler_options, last_epoch=lastepoch)
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optG, **scheduler_options, last_epoch=lastepoch)
        elif scheduler_type == 'CosineAnnealingLRWithRestart':
            print('..', 'CosineAnnealingLR_With_Restart')
            self.scheduler=lr_scheduler.CosineAnnealingLRWithRestart(self.optG, **scheduler_options, last_epoch=lastepoch)
        elif scheduler_type == 'LinearLR':
            self.scheduler=lr_scheduler.LinearLR(
                        self.optG, train_opt['n_iter'], last_epoch=lastepoch)
        elif scheduler_type == 'VibrateLR':
            self.scheduler=lr_scheduler.VibrateLR(
                        self.optG, train_opt['n_iter'], last_epoch=lastepoch)
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')
        print('Setting up Scheduler finished')

    
    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.
        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warm-up iter numbers. -1 for no warm-up.
                Default： -1.
        """
        if current_iter > 1:
            self.scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optG.param_groups]


    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix, x_start, x_pred, condition_feats, tt = self.netG(self.data)
        
        
        
        self.SR_onestep = x_pred

        t_loss_weight = (1 - tt/self.opt['model']['beta_schedule']['train']['n_timestep']) ** 2

        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)

        if self.opt["loss_w"]["lpips_w"]>0:
            lpips_errs = self.lpips_loss(x_pred, x_start)  # [bs,1,h,w]
            lpips_loss = lpips_errs.sum()/int(b*h*w)* self.opt["loss_w"]['lpips_w']
        else:
            lpips_loss = torch.zeros_like(l_pix)

        if self.opt["loss_w"]["fft_w"]>0:
            freq_loss = self.freq_loss(x_pred, x_start) #*  t_loss_weight.sum()
        else:
            freq_loss = torch.zeros_like(l_pix)
            
        loss = l_pix + lpips_loss + freq_loss # + condition_loss

        if torch.isfinite(loss).all():
            loss.backward()
            self.optG.step()

            # set log
            self.log_dict['l_pix'] = l_pix.item()
            self.log_dict['l_per'] = lpips_loss.sum().item()
            self.log_dict['l_freq'] = freq_loss.sum().item()
            if isinstance(condition_feats, dict):
                for name, value in condition_feats.items():
                    self.log_dict['l_' + name] = value.sum().item() / int(b*c*h*w)

        else:
            raise FloatingPointError('Non-finite generator loss')
        
        if self.net_disc is not None:
            """
            Generator loss: fool the discriminator
            """
            self.optG.zero_grad()
            self.net_disc.requires_grad_(False)
            l_pix, x_start, x_tgt_pred, condition_feats, tt = self.netG(self.data)
            # x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
            lossG = self.net_disc(x_tgt_pred, for_G=True).mean() * self.opt["loss_w"]['lambda_gan']
            lossG.backward()
            self.optG.step()
            self.optG.zero_grad()
            """
            Discriminator loss: fake image vs real image
            """
            self.net_disc.requires_grad_(True)
            self.net_disc.cv_ensemble.requires_grad_(False)
            self.optimizer_disc.zero_grad()
            # real image
            lossD_real = self.net_disc(x_start.detach(), for_real=True).mean() * self.opt["loss_w"]['lambda_gan']
            lossD_real.mean().backward()
            self.optimizer_disc.step()
            self.optimizer_disc.zero_grad()
            # fake image
            lossD_fake = self.net_disc(x_tgt_pred.detach(), for_real=False).mean() * self.opt["loss_w"]['lambda_gan']
            lossD_fake.mean().backward()
            self.optimizer_disc.step()
            self.optimizer_disc.zero_grad()
            lossD = lossD_real + lossD_fake

            self.log_dict['l_G'] = lossG.item()
            self.log_dict['l_D'] = lossD.item()




    def test(self, continous=False, seed=1):
        was_training = self.netG.training
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR, self.SR_onestep = self.netG.module.super_resolution(
                    self.data['SR'], continous, seed, self.data['LR'])
            else:
                self.SR, self.SR_onestep = self.netG.super_resolution(
                    self.data['SR'], continous, seed, self.data['LR'])
        self.netG.train(was_training)

    def sample(self, batch_size=1, continous=False):
        was_training = self.netG.training
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train(was_training)

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train', num_train_timesteps=1000):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device, num_train_timesteps)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device, num_train_timesteps)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            out_dict['SR_onestep'] = self.SR_onestep.detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        network = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        for name, value in getattr(network, 'last_maps', {}).items():
            out_dict[name] = value.detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'stage': self.opt['stage'],
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        opt_state['scheduler'] = self.scheduler.state_dict()
        opt_state['rng_cpu'] = torch.get_rng_state()
        opt_state['rng_cuda'] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        if self.net_disc is not None:
            opt_state['discriminator'] = self.net_disc.state_dict()
            opt_state['optimizer_disc'] = self.optimizer_disc.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        # Resume paths must refer to trusted training outputs; metadata includes floats.
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            net_params = torch.load(gen_path, map_location=self.device, weights_only=False)



            is_uard = bool((self.opt['model'].get('uard') or {}).get('enabled', False))
            network.load_state_dict(net_params, strict=is_uard)
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train' and self.opt['path'].get('resume_training', True):
                # optimizer
                opt = torch.load(opt_path, map_location=self.device, weights_only=False)
                if opt.get('stage', self.opt['stage']) != self.opt['stage']:
                    raise ValueError('Cross-stage transfer requires resume_training=false')
                self.pending_resume = opt
                # self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
