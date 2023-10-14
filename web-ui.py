import os
import asyncio
import numpy as np
import soundfile as sf
import gradio as gr
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

# Constants
TEMP_FILE_NAME = "mixture.wav"
OUTPUT_FOLDER_NAME = "output"
PREFIX = "mixture_"

# Windows Event Loop Policy
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def save_to_temp(audio_data, sample_rate, path):
    if audio_data is None:
        raise gr.Error("Please upload an audio file before proceeding.")
    sf.write(str(path), audio_data, sample_rate)  # Convert path to string


def run_inference(temp_file_path, params):
    command = [
        'python', 'inference.py', '--large_gpu', 
        '--weight_MDXv3', str(params['weight_MDXv3']),
        '--weight_VOCFT', str(params['weight_VOCFT']),
        '--weight_HQ3', str(params['weight_HQ3']),
        '--chunk_size', str(params['chunk_size']),
        '--input_audio', temp_file_path,
        '--overlap_demucs', str(params['overlap_demucs']),
        '--overlap_MDX', str(params['overlap_MDX']),
        '--overlap_MDXv3', str(params['overlap_MDXv3']),
        '--output_format', params['output_format'],
        '--bigshifts', str(params['BigShifts_MDX']),
        '--output_folder', params['output_folder']
    ]
    if params['vocals_only']:
        command.append('--vocals_only')
        command.append('true')

    subprocess.run(command, check=True)


def process_output(output_folder):
    output_folder = Path(output_folder)  # Ensure it's a Path object
    output_folder.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    for file in output_folder.iterdir():
        if file.name.startswith(PREFIX):
            new_name = file.name[len(PREFIX):]
            new_path = output_folder / new_name
            if new_path.exists():
                new_path.unlink()  # Delete the existing file
            file.rename(new_path)

    paths = {
        "vocals": output_folder / "vocals.wav",
        "instrum": output_folder / "instrum.wav",
        "instrum2": output_folder / "instrum2.wav",
        "bass": output_folder / "bass.wav",
        "drums": output_folder / "drums.wav",
        "other": output_folder / "other.wav"
    }
    
    return paths


def separate_audio(input_audio, BigShifts_MDX, overlap_MDX, overlap_MDXv3, weight_MDXv3, weight_VOCFT, weight_HQ3, overlap_demucs, output_format, vocals_instru_only, chunk_size):

    # Check if the user has provided an audio file
    if input_audio is None:
        raise gr.Error("Please upload an audio file in WAV format.")
    
    sample_rate, audio_data = input_audio
    
    temp_file_path = Path.cwd() / TEMP_FILE_NAME
    save_to_temp(audio_data, sample_rate, temp_file_path)
    
    output_folder_path = Path.cwd() / OUTPUT_FOLDER_NAME
    
    params = {
        "weight_MDXv3": weight_MDXv3,
        "weight_VOCFT": weight_VOCFT,
        "weight_HQ3": weight_HQ3,
        "chunk_size": chunk_size,
        "overlap_demucs": overlap_demucs,
        "overlap_MDX": overlap_MDX,
        "overlap_MDXv3": overlap_MDXv3,
        "output_format": output_format,
        "BigShifts_MDX": BigShifts_MDX,
        "vocals_only": vocals_instru_only,
        "output_folder": str(output_folder_path)
    }
    run_inference(temp_file_path, params)
    paths = process_output(params["output_folder"])
    
    # Clean up by removing the temporary audio file
    temp_file_path.unlink()
    
    return paths["vocals"], paths["instrum"], paths.get("instrum2", None), paths.get("bass", None), paths.get("drums", None), paths.get("other", None)


# Gradio interface setup
theme = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="cyan",
)

with gr.Blocks(theme=theme) as demo:
    input_audio = gr.Audio(label="Upload Audio", interactive=True)
    BigShifts_MDX = gr.Slider(minimum=1, maximum=41, step=1, value=6, label="BigShifts MDX")
    overlap_MDX = gr.Slider(minimum=0, maximum=0.95, step=0.05, value=0, label="Overlap MDX")
    overlap_MDXv3 = gr.Slider(minimum=2, maximum=40, step=2, value=8, label="Overlap MDXv3")
    weight_MDXv3 = gr.Slider(minimum=0, maximum=10, step=1, value=8, label="Weight MDXv3")
    weight_VOCFT = gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Weight VOCFT")
    weight_HQ3 = gr.Slider(minimum=0, maximum=10, step=1, value=2, label="Weight HQ3")
    overlap_demucs = gr.Slider(minimum=0, maximum=0.95, step=0.05, value=0.6, label="Overlap Demucs")
    output_format = gr.Dropdown(choices=["PCM_16", "FLOAT"], value="PCM_16", label="Output Format")
    vocals_instru_only = gr.Checkbox(value=False, label="Vocals Only", interactive=True)
    chunk_size = gr.Slider(minimum=100000, maximum=1000000, step=100000, value=500000, label="Chunk Size")
    
    b1 = gr.Button("Start Audio Separation", variant="primary")
    
    vocals = gr.Audio(label="Vocals")
    instrumental = gr.Audio(label="Instrumental")
    instrumental2 = gr.Audio(label="Instrumental 2")
    bass = gr.Audio(label="Bass")
    drums = gr.Audio(label="Drums")
    other = gr.Audio(label="Other")

    b1.click(separate_audio, inputs=[input_audio, BigShifts_MDX, overlap_MDX, overlap_MDXv3, weight_MDXv3, weight_VOCFT, weight_HQ3, overlap_demucs, output_format, vocals_instru_only, chunk_size], outputs=[vocals, instrumental, instrumental2, bass, drums, other])

demo.queue().launch(debug=True, share=False)