"""Export the separate, pinned gun/knife checkpoint for local ONNX inference."""
import hashlib
import json
import os
from pathlib import Path

import config


def main():
    import torch
    from ultralytics import YOLO
    from ultralytics.nn import modules, tasks
    from ultralytics.nn.modules import block

    source = Path(config.DATA_DIR) / 'setup_cache' / 'weapon_gun_knife_yolo11n.pt'
    destination = Path(config.DATA_DIR) / 'models' / 'weapon_gun_knife_yolo11n.onnx'
    if not source.is_file():
        raise FileNotFoundError('Download cosgun99/gun-knife-yolo11n revision 572728c best.pt first.')
    # Only installed Torch/Ultralytics classes reported by the pinned model are
    # permitted. Never execute arbitrary checkpoint pickle globals.
    allowed = [set, tasks.DetectionModel]
    allowed += [getattr(torch.nn, name) for name in (
        'BatchNorm2d', 'Sequential', 'Identity', 'Conv2d', 'MaxPool2d',
        'ModuleList', 'SiLU', 'Upsample')]
    allowed += [(getattr(modules, name, None) or getattr(block, name)) for name in (
        'Attention', 'Concat', 'DFL', 'Conv', 'Bottleneck', 'C3k', 'SPPF',
        'Detect', 'DWConv', 'C3k2', 'C2PSA', 'PSABlock')]
    previous = torch.serialization.get_safe_globals()
    try:
        torch.serialization.add_safe_globals(allowed)
        checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous)
    trained = checkpoint.get('ema') or checkpoint['model']
    if trained.names != {0: 'gun', 1: 'knife'}:
        raise ValueError(f'Unexpected weapon model classes: {trained.names}')
    torch.set_num_threads(2)
    wrapper = YOLO('yolo11n.yaml', task='detect')
    wrapper.model = trained.float().eval()
    wrapper.model.task = 'detect'
    wrapper.model.pt_path = str(source)
    wrapper.task = 'detect'
    exported = Path(wrapper.export(format='onnx', imgsz=640, opset=17,
                                   simplify=False, dynamic=False, nms=False, device='cpu'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(exported, destination)
    metadata = {
        'source': 'https://huggingface.co/cosgun99/gun-knife-yolo11n',
        'revision': '572728c', 'classes': {'0': 'gun', '1': 'knife'},
        'checkpoint_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'onnx_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Weapon model ready: {destination}')


if __name__ == '__main__':
    main()
