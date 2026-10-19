"""Check the existing behavior checkpoint without opening cameras or starting servers."""
import argparse
import hashlib
from pathlib import Path

import config
from python_engine.violence_classifier import create_default_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(2)
        classifier = create_default_classifier(config)
        checkpoint = Path(classifier.checkpoint_path)
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        torch, model, device = classifier._load_model(device_override=args.device)
        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
        inputs = classifier._preprocess(frames, torch, device)
        with torch.inference_mode():
            output = model(inputs)
        if tuple(output.shape) != (1, 2) or not torch.isfinite(output).all():
            raise RuntimeError('The checkpoint did not produce two finite scene-classification outputs.')
        classifier._violence_probability(output)
        after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if before != after:
            raise RuntimeError('Checkpoint contents changed during verification.')
        print(f'PASS: existing X3D-M checkpoint loaded on {device} with PyTorch {torch.__version__}.')
        print('PASS: synthetic 16-frame clip produced two finite outputs; checkpoint is unchanged.')
        print('This checks execution, not action-recognition accuracy. Restart the live engine to load installed dependencies.')
        return 0
    except Exception as error:
        print(f'FAIL: {type(error).__name__}: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
