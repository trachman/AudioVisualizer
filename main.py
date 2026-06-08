import argparse
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tkinter as tk
from scipy import fft
from matplotlib.widgets import Slider
from pathlib import Path
import time

def parse_wav_file():
    parser = argparse.ArgumentParser(
        description="Load data or audio files into the chart application.")
    parser.add_argument(
        '-f',
        '--file', 
        type=Path, 
        nargs='?',
        required=True, 
        help="Path to the input .wav file")
    args = parser.parse_args()
    if not args.file.is_file():
        print(f"Error: The file path '{args.file}' does not exist.")
        return
    if args.file.suffix.lower() != '.wav':
        raise argparse.ArgumentTypeError(
            f"Unsupported file format. Please provide a .wav file instead.")
    return args.file

# remove the toolbar
mpl.rcParams['toolbar'] = 'None'

# Read the WAV data into a NumPy array and extract the sample rate
filename = parse_wav_file()
data, sample_rate_hz = sf.read(filename, dtype='float32')
if len(data.shape) > 1:
    # Convert stereo to mono if necessary
    data = np.mean(data, axis=1)

# Compute the Real FFT
# rfft automatically drops the negative mirror frequencies
rfft_data = fft.rfft(data)
abs_rfft_data = np.abs(rfft_data)
frequencies = fft.rfftfreq(len(data), 1 / sample_rate_hz)

# Compute short time fourier transforms over windows
SONG_DURATION = len(data) / sample_rate_hz
WINDOW_LENGTH = 0.15
current_time = 0
NUM_SAMPLES = int(WINDOW_LENGTH * sample_rate_hz)
window_data = data[0 : NUM_SAMPLES + 1]
window_rfft_data = np.abs(fft.rfft(window_data))
fft_log_db = 20 * np.log10(window_rfft_data + 1e-10)
window_freq = fft.rfftfreq(len(window_data), 1 / sample_rate_hz)

# Plot + Play
plt.ion()
fig, ax = plt.subplots(figsize=(8, 3), facecolor="#2e2e2e", layout="constrained")
manager = plt.get_current_fig_manager()
if hasattr(manager, "window"):
    root = manager.window
    note_icon = tk.PhotoImage(file="note.png")
    root.iconphoto(False, note_icon)
fig.canvas.manager.set_window_title("Audio Visualizer")
def on_close(event):
    sd.stop()
fig.canvas.mpl_connect('close_event', on_close)
(line,) = ax.plot(window_freq, fft_log_db, color="purple")
fill_collection = None
ax.set_facecolor("#1e1e1e")
ax.set_xlabel("Frequency (Hz)", color='white')
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_xlim(0, np.max(window_freq))
ax.set_ylim(0, 75)
ax.grid(False) # No gridlines
ax.tick_params(colors="white")
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Time Slider
time_slider_axes = plt.axes([0.25, 0.93, 0.5, 0.06])
time_slider = Slider(
        ax=time_slider_axes,
        label='Time',
        valmin=0,
        valmax=SONG_DURATION,
        valinit=0,
        color='skyblue')
time_slider.valtext.set_color("white")
time_slider.label.set_color("white")
current_index = 0
current_time = time_slider.val # 0
def update_time(val):
    global current_index, current_time
    if val == current_time:
        return
    sd.stop()
    current_time = val
    current_index = int(val * sample_rate_hz)
    sd.play(data[current_index:], sample_rate_hz)
time_slider.on_changed(update_time)

# Band Pass Filters
def update_data(high_pass, low_pass):
    global data, frequencies, rfft_data, current_time
    copy_of_rfft = rfft_data.copy()
    low_freq_indices = frequencies < low_pass
    high_freq_indices = frequencies > high_pass
    copy_of_rfft[low_freq_indices] = 0.0
    copy_of_rfft[high_freq_indices] = 0.0
    data = fft.irfft(copy_of_rfft, n=len(data))
    sd.stop()
    sd.play(data[current_index:], sample_rate_hz)
    current_time = current_index / sample_rate_hz # reset the time

# Low Pass Filter
low_pass_slider_axes = plt.axes([0.25, 0.86, 0.5, 0.06])
low_pass_slider = Slider(
        ax=low_pass_slider_axes,
        label='Low Pass Filter (Hz)',
        valmin=0,
        valmax=np.max(frequencies),
        valinit=0,
        color='skyblue')
low_pass_slider.valtext.set_color("white")
low_pass_slider.label.set_color("white")
# High Pass Filter
high_pass_slider_axes = plt.axes([0.25, 0.79, 0.5, 0.06])
high_pass_slider = Slider(
        ax=high_pass_slider_axes,
        label='High Pass Filter (Hz)',
        valmin=0,
        valmax=np.max(frequencies),
        valinit=np.max(frequencies),
        color='skyblue')
high_pass_slider.valtext.set_color("white")
high_pass_slider.label.set_color("white")

def update_low_pass(val):
    update_data(high_pass_slider.val, val)
def update_high_pass(val):
    update_data(val, low_pass_slider.val)
low_pass_slider.on_changed(update_low_pass)
high_pass_slider.on_changed(update_high_pass)

try:
    sd.play(data, sample_rate_hz)
    while sd.get_stream().active:
        start_time = time.perf_counter()
        current_index = int(current_time * sample_rate_hz)
        window_data = data[current_index : current_index + NUM_SAMPLES]
        if not len(window_data):
            break
        window_rfft_data = np.abs(fft.rfft(window_data))
        fft_log_db = 20 * np.log10(window_rfft_data + 1e-10)
        n_fft, n_freq = len(fft_log_db), len(window_freq)
        if n_fft < n_freq:
            fft_log_db = np.append(fft_log_db, np.zeros(n_freq - n_fft, dtype=fft_log_db.dtype))
        line.set_ydata(fft_log_db)
        if fill_collection:
            fill_collection.remove()
        fill_collection = ax.fill_between(window_freq, fft_log_db, y2=0, color="purple", alpha=0.5)
        fig.canvas.draw()
        fig.canvas.flush_events()
        duration = time.perf_counter() - start_time
        current_time += duration
        time_slider.set_val(current_time)
except KeyboardInterrupt:
    sd.stop()

# Turn off interactive mode when done so window stays open
# plt.ioff()
# plt.show()
sd.stop()


