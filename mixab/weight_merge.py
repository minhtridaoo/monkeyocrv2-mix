"""Mix A-core step A1 (run: python -m mixab.weight_merge ...): linear weight interpolation between two fine-tuned DBNet checkpoints.

w(alpha) = (1 - alpha) * w_BClean + alpha * w_B1, inspired by WiSE-FT (Wortsman et al., CVPR 2022);
here both endpoints are fine-tuned from the same B0, so this is not a reproduction of WiSE-FT.

Locked policy (Ke_hoach_phoi_hop_MixA_MixB_20261119.md, section 4.1):
- alpha in {0, 1}: the endpoint state_dict is copied unchanged (no arithmetic).
- 0 < alpha < 1: the same scalar alpha for every floating-point tensor, including BatchNorm
  running statistics; non-floating tensors (e.g. num_batches_tracked) are taken from B1.
- Every floating tensor must be finite and every running_var strictly positive.
- BN statistics are not re-estimated in round 1; merged statistics may not match the merged network.

CPU only. Does not train, read TEST data, or touch historical artifacts. Output files are never
overwritten.
"""
import argparse
from fractions import Fraction
from pathlib import Path

from mixab import require, save_json, sha256

STAGE = 'Mix A-core weight interpolation v1'
DEFAULT_ALPHAS = ('0', '0.25', '0.5', '0.75', '1')
MODULE_PREFIX = 'module.'


def load_state_dict(path, allow_pickle=False):
    """Return (state_dict, provenance) from a raw state_dict or an mmengine checkpoint."""
    import torch

    path = Path(path)
    require(path.is_file(), f'Checkpoint not found: {path}')
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
        loader = 'weights_only'
    except Exception as exc:  # mmengine checkpoints may hold non-tensor metadata objects
        require(allow_pickle, f'{path} needs full unpickling ({type(exc).__name__}); rerun with '
                              '--allow-pickle only for trusted team checkpoints')
        obj = torch.load(path, map_location='cpu', weights_only=False)
        loader = 'full_pickle'
    wrapper = 'raw_state_dict'
    if isinstance(obj, dict) and isinstance(obj.get('state_dict'), dict):
        obj, wrapper = obj['state_dict'], 'mmengine_state_dict'
    require(isinstance(obj, dict) and obj, f'{path} does not contain a state_dict')
    require(all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items()),
            f'{path} state_dict has non-tensor entries')
    stripped = sum(k.startswith(MODULE_PREFIX) for k in obj)
    require(stripped in (0, len(obj)), f'{path} mixes prefixed and unprefixed keys')
    if stripped:  # same rule as mmengine load_checkpoint revise_keys=[(r'^module\.', '')]
        obj = {k[len(MODULE_PREFIX):]: v for k, v in obj.items()}
    provenance = dict(path=str(path), sha256=sha256(path), loader=loader, wrapper=wrapper,
                      stripped_module_prefix=bool(stripped), tensors=len(obj))
    return dict(obj), provenance


def require_compatible(first, second):
    require(set(first) == set(second),
            f'Key sets differ: only-first={sorted(set(first) - set(second))[:5]}, '
            f'only-second={sorted(set(second) - set(first))[:5]}')
    for key in first:
        a, b = first[key], second[key]
        require(a.shape == b.shape, f'Shape differs for {key}: {tuple(a.shape)} vs {tuple(b.shape)}')
        require(a.dtype == b.dtype, f'Dtype differs for {key}: {a.dtype} vs {b.dtype}')


def parse_alpha(text):
    value = Fraction(text)
    require(0 <= value <= 1, f'alpha must lie in [0, 1], got {text}')
    return value


def interpolate(bclean, b1, alpha):
    """Return (state_dict, report). alpha is a Fraction; endpoints are exact copies."""
    import torch

    require_compatible(bclean, b1)
    if alpha in (0, 1):
        source = bclean if alpha == 0 else b1
        merged = {k: v.detach().clone() for k, v in source.items()}
        return merged, dict(mode='endpoint_copy', endpoint='B-Clean' if alpha == 0 else 'B1',
                            non_floating_from=None, non_floating_keys=[], non_floating_differ=[])
    a = float(alpha)
    merged, non_float, differ = {}, [], []
    for key, left in bclean.items():
        right = b1[key]
        if left.is_floating_point():
            # Accumulate in float64, then cast back to the stored dtype.
            value = left.double().mul(1 - a).add(right.double().mul(a))
            merged[key] = value.to(left.dtype)
        else:
            merged[key] = right.detach().clone()
            non_float.append(key)
            if not torch.equal(left, right):
                differ.append(key)
    return merged, dict(mode='linear', endpoint=None, non_floating_from='B1',
                        non_floating_keys=non_float, non_floating_differ=differ)


def validate(state_dict):
    """Finite floating tensors and strictly positive BN running_var."""
    import torch

    running_var = 0
    for key, value in state_dict.items():
        if value.is_floating_point():
            require(bool(torch.isfinite(value).all()), f'Non-finite values in {key}')
        if key.endswith('running_var'):
            running_var += 1
            require(bool((value > 0).all()), f'running_var must be > 0 in {key}')
    return dict(floating=sum(v.is_floating_point() for v in state_dict.values()),
                running_var_tensors=running_var)


def compare(first, second):
    """Tensor-level equality (keys, shape, dtype, values); deliberately ignores file bytes."""
    import torch

    if set(first) != set(second):
        return dict(equal=False, reason='key sets differ')
    for key in first:
        a, b = first[key], second[key]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            return dict(equal=False, reason=f'tensor differs: {key}')
    return dict(equal=True, reason='same keys, shapes, dtypes and values')


def alpha_tag(alpha):
    return f'alpha_{float(alpha):.2f}'.replace('.', 'p')


def build(bclean_path, b1_path, alphas, output, allow_pickle=False):
    import torch

    output = Path(output)
    bclean, bclean_info = load_state_dict(bclean_path, allow_pickle)
    b1, b1_info = load_state_dict(b1_path, allow_pickle)
    require_compatible(bclean, b1)
    alphas = sorted({parse_alpha(a) for a in alphas})
    targets = [output/f'{alpha_tag(a)}.pth' for a in alphas]
    clash = [str(p) for p in targets + [output/'merge_manifest.json'] if p.exists()]
    require(not clash, f'Refusing to overwrite existing outputs: {clash}')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha, target in zip(alphas, targets):
        merged, report = interpolate(bclean, b1, alpha)
        checks = validate(merged)
        if report['mode'] == 'endpoint_copy':
            source = bclean if alpha == 0 else b1
            checks['endpoint_state_dict'] = compare(merged, source)
            require(checks['endpoint_state_dict']['equal'], f'Endpoint copy differs: {target}')
        temp = target.with_name(target.name + '.tmp')
        torch.save(merged, temp)
        temp.replace(target)
        reloaded, _ = load_state_dict(target)
        checks['reload'] = compare(reloaded, merged)
        require(checks['reload']['equal'], f'Saved checkpoint does not reload identically: {target}')
        rows.append(dict(alpha=str(alpha), alpha_float=float(alpha), checkpoint=str(target),
                         checkpoint_sha256=sha256(target), **report, checks=checks))
        print(f'{target.name}: {report["mode"]}, {checks["floating"]} floating tensors, reload equal', flush=True)
    save_json(output/'merge_manifest.json', dict(
        stage=STAGE, formula='(1 - alpha) * B-Clean + alpha * B1', endpoints=dict(bclean=bclean_info, b1=b1_info),
        policy=dict(endpoint='copy state_dict unchanged', floating='same scalar alpha, float64 accumulate',
                    non_floating='from B1', bn_statistics='interpolated, not re-estimated',
                    verification='PENDING_REFERENCE_CHECK until merge_audit gates 3a/3b pass'),
        models=rows))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bclean', required=True, type=Path, help='Selected B-Clean checkpoint')
    parser.add_argument('--b1', required=True, type=Path, help='B1 checkpoint')
    parser.add_argument('--output', required=True, type=Path, help='New, versioned output directory')
    parser.add_argument('--alphas', nargs='+', default=list(DEFAULT_ALPHAS))
    parser.add_argument('--allow-pickle', action='store_true',
                        help='Allow full unpickling (mmengine checkpoints); trusted files only')
    a = parser.parse_args()
    build(a.bclean, a.b1, a.alphas, a.output, a.allow_pickle)


if __name__ == '__main__':
    main()
