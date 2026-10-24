"""Stage-0 training without the legacy main.py metric/LPIPS/GAN dependencies."""
import argparse
import json
from pathlib import Path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/UARD_anchor_s0.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume', help='Same-stage checkpoint prefix without _gen.pth')
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from model.model import DDPM
    from uard_io import dataset_from_options
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if cfg['stage'] != 0:
        parser.error('Expected Stage-0 configuration')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('Supported devices: cpu or cuda[:index]')
    if device.type == 'cuda':
        torch.cuda.set_device(device.index if device.index is not None else 0)
    torch.manual_seed(args.seed)
    cfg.update(phase='train', gpu_ids=[0] if device.type == 'cuda' else [],
               distributed=False, seed=args.seed, xT_noise_r=0)
    folder = Path(args.output)
    (folder/'checkpoint').mkdir(parents=True, exist_ok=True)
    cfg['path'].update(checkpoint=str(folder/'checkpoint'), resume_state=args.resume,
                       resume_training=bool(args.resume))
    train_options = cfg['datasets']['train']
    train_loader = DataLoader(dataset_from_options(train_options, augment=True),
                              batch_size=train_options['batch_size'], shuffle=True,
                              num_workers=train_options.get('num_workers', 0))
    validation_enabled = bool(cfg['train'].get('validation_enabled', True))
    validation_loader = DataLoader(dataset_from_options(cfg['datasets']['val']), batch_size=1) if validation_enabled else None
    trainer = DDPM(cfg)
    (folder/'resolved_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    step, epoch = trainer.begin_step, trainer.begin_epoch
    total_steps = int(cfg['train']['n_iter'])
    val_freq = int(cfg['train']['val_freq'])
    if total_steps <= step or (validation_enabled and val_freq < 1):
        parser.error('n_iter must exceed resume step and val_freq must be positive')
    best = float('inf')
    summary_path = folder/'summary.json'
    if args.resume and summary_path.exists():
        previous_best = json.loads(summary_path.read_text()).get('best_validation_nll')
        best = float('inf') if previous_best is None else previous_best
    with (folder/'validation.jsonl').open('a', encoding='utf-8') as log:
        while step < total_steps:
            epoch += 1
            for batch in train_loader:
                step += 1
                trainer.feed_data(batch)
                trainer.optimize_parameters()
                trainer.update_learning_rate(step)
                if step % cfg['train']['print_freq'] == 0:
                    print(f'Stage 0: {step}/{total_steps}, NLL={trainer.get_current_log()["l_unc"]:.6f}', flush=True)
                if not validation_enabled and (step % cfg['train']['save_checkpoint_freq'] == 0 or step == total_steps):
                    trainer.save_network(epoch, step)
                    checkpoint = folder/'checkpoint'/f'I{step}_E{epoch}_gen.pth'
                    summary_path.write_text(json.dumps({'validation_enabled': False,
                        'selection_rule': 'fixed training budget; no test-set selection',
                        'last_step': step, 'last_checkpoint': str(checkpoint)}, indent=2), encoding='utf-8')
                    print(f'Saved training checkpoint: {checkpoint}', flush=True)
                if validation_enabled and (step % val_freq == 0 or step == total_steps):
                    trainer.netG.eval()
                    loss_sum, elements = 0.0, 0
                    with torch.no_grad():
                        for val_batch in validation_loader:
                            trainer.feed_data(val_batch)
                            value = trainer.netG(trainer.data)[0]
                            if not torch.isfinite(value):
                                raise FloatingPointError('Non-finite validation likelihood')
                            loss_sum += value.item()
                            elements += trainer.data['HR'].numel()
                    score = loss_sum/elements
                    trainer.netG.train()
                    trainer.save_network(epoch, step)
                    checkpoint = folder/'checkpoint'/f'I{step}_E{epoch}_gen.pth'
                    if score < best:
                        best = score
                        torch.save(trainer.netG.state_dict(), folder/'checkpoint'/'best_gen.pth')
                    entry = {'step': step, 'validation_nll': score}
                    log.write(json.dumps(entry)+'\n')
                    log.flush()
                    summary_path.write_text(json.dumps({'best_validation_nll': best,
                        'best_checkpoint': str(folder/'checkpoint'/'best_gen.pth'),
                        'last_checkpoint': str(checkpoint)}, indent=2), encoding='utf-8')
                    print(entry, flush=True)
                if step == total_steps:
                    break


if __name__ == '__main__':
    main()
