"""Run with the SoulX interpreter and its repository as the working directory."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    # Keep upstream progress messages off the line-oriented reply stream.
    replies, sys.stdout = sys.stdout, sys.stderr
    sys.path.insert(0, str(Path.cwd()))
    import soundfile as sf
    import torch
    from cli.inference import build_model, load_config, DataProcessor

    options = json.loads(sys.stdin.readline())
    torch.set_num_threads(options['threads'])
    device = options['device']
    config = load_config(options['config'])
    model = build_model(options['checkpoint'], config, device)
    processor = DataProcessor(config.audio.hop_size, config.audio.sample_rate, options['phoneset'], device)
    segments = json.loads(Path(options['prompt_metadata']).read_text())
    if not segments:
        raise ValueError('Prompt metadata contains no segments')
    prompt_meta = segments[0]
    start, end = prompt_meta['time']
    if not 0 <= start < end:
        raise ValueError('Invalid first prompt segment time range')
    # DataProcessor takes audio starting at zero even for a later metadata segment.
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / 'prompt.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(start / 1000), '-i', options['prompt_audio'],
                        '-t', str((end - start) / 1000), '-ar', str(config.audio.sample_rate),
                        '-ac', '1', str(audio)], check=True)
        prompt = processor.process(prompt_meta, str(audio))
    with torch.inference_mode():
        for line in sys.stdin:
            meta, destination = json.loads(line)
            target = processor.process(meta)
            torch.manual_seed(options['seed'])
            wave = model.infer({'prompt': prompt, 'target': target}, control='score',
                               auto_shift=False, pitch_shift=0, n_steps=options['steps'], cfg=config.infer.cfg)
            sf.write(destination, wave.squeeze().cpu().numpy(), config.audio.sample_rate, subtype='FLOAT')
            print('done', file=replies, flush=True)


if __name__ == '__main__':
    main()
