"""Shared QXS/SAR2Opt S0 -> S1 -> S2 runner (Python 3.8+)."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from qxs_data import inspect_dataset

PROJECT = Path(__file__).resolve().parent
DEFAULT_DATA = 'datasets/QXS-SAROPT'
STAGES = ('UARD_QXS_S0_anchor', 'UARD_QXS_S1_full', 'UARD_QXS_S2_full')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def record_validation(checkpoint_dir, epoch, step, psnr, ssim, lpips):
    """Called by main.py only for configs explicitly enabling PSNR selection."""
    psnr, ssim, lpips = map(float, (psnr, ssim, lpips))
    if math.isnan(psnr) or not all(math.isfinite(v) for v in (ssim, lpips)):
        raise ValueError('Non-finite QXS validation metrics')
    directory = Path(checkpoint_dir)
    gen = directory / 'I{}_E{}_gen.pth'.format(step, epoch)
    if not gen.is_file():
        raise FileNotFoundError('Save checkpoint before validation selection: ' + str(gen))
    record = dict(step=step, epoch=epoch, psnr=psnr, ssim=ssim, lpips=lpips,
                  checkpoint=str(gen.resolve()), selection_metric='validation PSNR')
    with (directory.parent / 'validation_history.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
    best_path = directory.parent / 'best_validation.json'
    previous = json.loads(best_path.read_text(encoding='utf-8')) if best_path.exists() else None
    if previous is None or psnr > previous['psnr']:
        write_json(best_path, record)


def latest_checkpoint(directory, step=None):
    candidates = []
    for gen in Path(directory).glob('I*_E*_gen.pth'):
        match = re.fullmatch(r'I(\d+)_E(\d+)_gen\.pth', gen.name)
        if not match:
            continue
        iteration, epoch = map(int, match.groups())
        prefix = str(gen)[:-len('_gen.pth')]
        opt = Path(prefix + '_opt.pth')
        if (step is None or iteration == step) and opt.is_file():
            # Interrupted torch.save files without their ZIP central directory
            # are not usable resume candidates. Do not delete any checkpoint.
            if zipfile.is_zipfile(gen) and zipfile.is_zipfile(opt):
                candidates.append((iteration, epoch, prefix))
    return max(candidates)[2] if candidates else None


def run_command(command, log_path, gpu):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTHONUNBUFFERED'] = '1'
    print('Starting:', ' '.join(command), flush=True)
    with Path(log_path).open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=str(PROJECT), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise


def prepare_run(root, data_root, sar_dir, optical_dir, batch_size, workers, budgets, seed,
                dataset='QXS', image_size=256, expected_counts=None):
    if root.exists():
        raise FileExistsError('Run directory already exists; use --resume-run: ' + str(root))
    report = inspect_dataset(data_root, sar_dir, optical_dir, image_size=image_size)
    profile = 'UARD_QXS' if dataset == 'QXS' else 'UARD_{}_{}'.format(dataset, image_size)
    stage_names = [profile + suffix for suffix in ('_S0_anchor', '_S1_full', '_S2_full')]
    if dataset == 'SAR2Opt':
        from qxs_data import collect_pairs
        seen = {}
        for split, info in report.items():
            pairs = collect_pairs(Path(info['root']) / info['sar_dir'],
                                  Path(info['root']) / info['optical_dir'])
            sar_root = Path(info['root']) / info['sar_dir']
            for sar, _ in pairs:
                sample_id = sar.relative_to(sar_root).with_suffix('').as_posix()
                if sample_id in seen:
                    raise ValueError('Sample ID occurs in {} and {}: {}'.format(seen[sample_id], split, sample_id))
                seen[sample_id] = split
        if expected_counts is not None:
            actual = [report[s]['count'] for s in ('train', 'val', 'test')]
            if actual != list(expected_counts):
                raise ValueError('Expected train/val/test counts {}, got {}'.format(expected_counts, actual))
    root.mkdir(parents=True, exist_ok=False)
    (root / 'configs').mkdir()
    state = dict(dataset=dataset, image_size=image_size, stage_names=stage_names, data_root=str(Path(data_root).resolve()), dataset_report=report,
                 batch_size=batch_size, num_workers=workers, budgets=budgets, seed=seed,
                 completed={}, status='prepared')
    for stage, name in enumerate(stage_names):
        cfg = json.loads((PROJECT / 'config' / '{}_s{}.json'.format(profile, stage)).read_text(encoding='utf-8'))
        stage_root = root / name
        stage_root.mkdir()
        cfg['name'] = '{}_{}'.format(name, root.name)
        cfg['path']['experiments_root'] = str(stage_root)
        # Explicit stage roots keep logs, checkpoints and resume runs together.
        for key, folder in (('log', 'logs'), ('tb_logger', 'tb_logger'),
                            ('results', 'results'), ('checkpoint', 'checkpoint')):
            cfg['path'][key] = str(stage_root / folder)
        cfg['path'].update(resume_state=None, resume_training=False)
        cfg['model']['uard']['anchor_checkpoint'] = None
        cfg['train']['n_iter'] = budgets[stage]
        for split in ('train', 'val'):
            cfg['datasets'][split].update(dataroot=report[split]['root'],
                                          sar_dir=report[split]['sar_dir'],
                                          optical_dir=report[split]['optical_dir'])
        cfg['datasets']['train'].update(batch_size=batch_size, num_workers=workers)
        write_json(root / 'configs' / 's{}.json'.format(stage), cfg)
    write_json(root / 'run_state.json', state)
    return state


def execute_run(root, state, gpu):
    stage_names = state.get('stage_names', STAGES)
    for stage, name in enumerate(stage_names):
        stage_root = root / name
        key = str(stage)
        if key in state['completed']:
            selected = Path(state['completed'][key]['selected_checkpoint'])
            if not selected.is_file():
                raise FileNotFoundError('Completed stage checkpoint is missing: ' + str(selected))
            print('Already completed:', name, flush=True)
            continue
        cfg_path = root / 'configs' / 's{}.json'.format(stage)
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        cfg['gpu_ids'] = [int(gpu)]
        latest = latest_checkpoint(stage_root / 'checkpoint')
        terminal = latest_checkpoint(stage_root / 'checkpoint', step=state['budgets'][stage])
        if stage:
            cfg['model']['uard']['anchor_checkpoint'] = state['completed']['0']['selected_checkpoint']
        if stage == 2:
            cfg['path']['resume_state'] = state['completed']['1']['selected_checkpoint'][:-len('_gen.pth')]
            cfg['path']['resume_training'] = False
        if latest:
            cfg['path'].update(resume_state=latest, resume_training=True)
        write_json(cfg_path, cfg)
        state.update(active_stage=stage, status='running')
        write_json(root / 'run_state.json', state)
        if not terminal:
            if stage == 0:
                command = [sys.executable, '-u', str(PROJECT / 'train_anchor.py'),
                           '--config', str(cfg_path), '--output', str(stage_root),
                           '--device', 'cuda:0', '--seed', str(state['seed'])]
                if latest:
                    command += ['--resume', latest]
            else:
                command = [sys.executable, '-u', str(PROJECT / 'main.py'), '-c', str(cfg_path),
                           '-p', 'train', '-gpu', str(gpu), '--seed', str(state['seed'])]
            try:
                run_command(command, stage_root / 'console.log', gpu)
            except BaseException:
                state['status'] = 'interrupted'
                write_json(root / 'run_state.json', state)
                raise
            terminal = latest_checkpoint(stage_root / 'checkpoint', step=state['budgets'][stage])
            if not terminal:
                raise RuntimeError('Stage ended without a complete terminal checkpoint: ' + name)
        if stage == 0:
            summary = json.loads((stage_root / 'summary.json').read_text(encoding='utf-8'))
            selected = summary['best_checkpoint']
            rule = 'minimum validation Laplace NLL'
        else:
            best = json.loads((stage_root / 'best_validation.json').read_text(encoding='utf-8'))
            selected = best['checkpoint']
            rule = 'maximum validation PSNR'
        if not zipfile.is_zipfile(selected):
            raise RuntimeError('Selected checkpoint is incomplete: ' + selected)
        state['completed'][key] = dict(selected_checkpoint=selected,
                                       terminal_checkpoint=terminal + '_gen.pth', selection_rule=rule)
        write_json(root / 'run_state.json', state)
        print('Completed {} | selected: {}'.format(name, selected), flush=True)

    cfg = json.loads((root / 'configs' / 's2.json').read_text(encoding='utf-8'))
    cfg['name'] = stage_names[2] + '_test_' + root.name
    cfg['path'].update(resume_state=state['completed']['2']['selected_checkpoint'][:-len('_gen.pth')],
                       resume_training=False)
    test = state['dataset_report']['test']
    cfg['datasets']['val'].update(dataroot=test['root'], sar_dir=test['sar_dir'],
                                 optical_dir=test['optical_dir'], data_len=-1)
    write_json(root / 'configs' / 'test_s2.json', cfg)
    state.update(status='complete', test_count=test['count'])
    write_json(root / 'run_state.json', state)
    print('All {} stages completed. Test config:'.format(state['dataset']), root / 'configs' / 'test_s2.json', flush=True)


def main(dataset='QXS', image_size=256):
    sar2opt = dataset == 'SAR2Opt'
    profile = 'UARD_QXS' if not sar2opt else 'UARD_SAR2Opt_512'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='datasets/SAR2Opt' if sar2opt else DEFAULT_DATA)
    parser.add_argument('--sar-dir', default='SAR' if sar2opt else None)
    parser.add_argument('--optical-dir', default='Optical' if sar2opt else None)
    parser.add_argument('--gpu', default='0', help='One physical CUDA GPU index')
    parser.add_argument('--batch-size', type=int, default=2 if sar2opt else 16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--s0-steps', type=int, default=20000 if sar2opt else 100000)
    parser.add_argument('--s1-steps', type=int, default=100000 if sar2opt else 600000)
    parser.add_argument('--s2-steps', type=int, default=50000 if sar2opt else 200000)
    parser.add_argument('--expected-counts', type=int, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'),
                        default=[1305, 145, 627] if sar2opt else None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--run-dir', help='New output folder, default experiments/<dataset>_<timestamp>')
    parser.add_argument('--resume-run', help='Existing same-dataset run folder; reuses its saved settings')
    parser.add_argument('--prepare-only', action='store_true', help='Check data and write configs, without training')
    args = parser.parse_args()
    if not args.prepare_only:
        from release_status import require_available
        require_available("s1")
    if not args.gpu.isdigit() or args.batch_size < 1 or args.num_workers < 0:
        parser.error('Use one nonnegative GPU index, positive batch size and nonnegative workers')
    budgets = [args.s0_steps, args.s1_steps, args.s2_steps]
    if min(budgets) < 1:
        parser.error('Training step budgets must be positive')
    if args.resume_run:
        if args.run_dir:
            parser.error('--run-dir and --resume-run are mutually exclusive')
        root = Path(args.resume_run).resolve()
        state = json.loads((root / 'run_state.json').read_text(encoding='utf-8'))
        if state['dataset'] != dataset or state.get('image_size', 256) != image_size:
            raise ValueError('Run dataset/resolution does not match this entry point')
        print('Resume uses the saved data, step budgets, batch size and seed.', flush=True)
        for split, info in state['dataset_report'].items():
            if not Path(info['root']).is_dir():
                raise FileNotFoundError(info['root'])
    else:
        root = (Path(args.run_dir) if args.run_dir else PROJECT / 'experiments' /
                (profile + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
        state = prepare_run(root, args.data_root, args.sar_dir, args.optical_dir,
                            args.batch_size, args.num_workers, budgets, args.seed,
                            dataset=dataset, image_size=image_size, expected_counts=args.expected_counts)
    print('{} run directory:'.format(dataset), root, flush=True)
    if not args.prepare_only:
        execute_run(root, state, args.gpu)


if __name__ == '__main__':
    main()
