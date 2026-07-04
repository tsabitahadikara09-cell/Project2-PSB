import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

st.set_page_config(page_title="Time-Frequency Analysis - STFT", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
[data-testid="stAppViewContainer"] {background:#0f1118;}
[data-testid="stSidebar"] {background:#262631; border-right:2px solid #4b4b55;}
[data-testid="stHeader"] {background:#0f1118;}
.block-container {padding-top:2rem; padding-left:2rem; padding-right:2rem; max-width:1600px;}
h1 {color:white; font-size:34px !important; font-weight:800 !important;}
h2,h3,label,p,span,div {color:white;}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {color:white !important;}
.stButton>button {background:#ff4b4b; color:white; border:0; border-radius:7px; padding:0.5rem 1.2rem;}
.stButton>button:hover {background:#ff6b6b; color:white; border:0;}
div[data-testid="stNumberInput"] input {background:#080b11; color:white; border:1px solid #383b45;}
.section-box {border-top:1px solid #555865; padding-top:20px; margin-top:24px;}
.plot-card {background:white; border-radius:6px; padding:8px; margin-bottom:18px;}
.small-note {font-size:13px; color:#ddd;}
hr {border-color:#444753;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- Utility ----------
def make_signal(n, fs, mode):
    t = np.arange(n) / fs
    if mode == "Sinyal ECG":
        # ECG-like synthetic signal, no external data needed
        y = 0.04*np.sin(2*np.pi*1.2*t)
        rr = max(int(fs*0.75), 1)
        for r in range(int(0.12*fs), n, rr):
            idx = np.arange(n)
            y += 1.0*np.exp(-0.5*((idx-r)/(0.012*fs+1))**2)
            y -= 0.22*np.exp(-0.5*((idx-(r-0.025*fs))/(0.010*fs+1))**2)
            y += 0.30*np.exp(-0.5*((idx-(r+0.035*fs))/(0.018*fs+1))**2)
        y = y / (np.max(np.abs(y)) + 1e-12)
        return t, y

    # Four time-varying sinus blocks to mimic Delphi/STFT task
    y = np.zeros(n)
    breaks = [0, int(0.25*n), int(0.50*n), int(0.75*n), n]
    freqs = [50, 150, 250, 350]
    for i in range(4):
        a, b = breaks[i], breaks[i+1]
        y[a:b] = np.sin(2*np.pi*freqs[i]*np.arange(a, b)/fs)
    # add slight second tone in last half so spectrum has small side components
    if n > 4:
        y[int(0.50*n):] += 0.22*np.sin(2*np.pi*65*np.arange(int(0.50*n), n)/fs)
    y = y/(np.max(np.abs(y))+1e-12)
    return t, y

def get_window(name, length):
    length = max(int(length), 2)
    n = np.arange(length)
    if name == "Rectangular":
        return np.ones(length)
    if name == "Bartlett":
        return np.bartlett(length)
    if name == "Hanning":
        return np.hanning(length)
    if name == "Hamming":
        return np.hamming(length)
    if name == "Blackman":
        return np.blackman(length)
    return np.hanning(length)

def spectrum(y, fs):
    n = len(y)
    Y = np.abs(np.fft.rfft(y)) / max(n, 1)
    f = np.fft.rfftfreq(n, d=1/fs)
    return f, Y

def stft_calc(y, fs, win_len, overlap, num_windows, window_name):
    n = len(y)
    win_len = int(np.clip(win_len, 8, n))
    overlap = int(np.clip(overlap, 0, win_len-1))
    hop = max(win_len - overlap, 1)
    starts = [i*hop for i in range(max(1, int(num_windows)))]
    starts = [s for s in starts if s + win_len <= n]
    if not starts:
        starts = [0]
    w = get_window(window_name, win_len)
    mags = []
    times = []
    for s in starts:
        seg = y[s:s+win_len] * w
        f, m = spectrum(seg, fs)
        mags.append(m)
        times.append((s + win_len/2) / fs)
    return np.array(times), f, np.array(mags).T

def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.55, linestyle="-")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def fig_signal(x, y, fs, title, window_slice=None):
    fig, ax = plt.subplots(figsize=(8.8, 2.6), dpi=120)
    if window_slice is None:
        ax.plot(np.arange(len(y)), y, color="red", linewidth=1.0)
    else:
        ax.plot(np.arange(len(y)), y, color="lightgray", linewidth=0.75, alpha=0.65)
        s, e, yw = window_slice
        ax.plot(np.arange(s, e), yw, color="black", linewidth=1.0)
    ax.set_xlim(0, len(y)-1)
    ax.set_ylim(-1.1, 1.1)
    style_axes(ax, title, "n sample", "Amplitude")
    fig.tight_layout()
    return fig

def fig_spectrum(y, fs, title):
    f, Y = spectrum(y, fs)
    fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=120)
    ax.plot(f, Y, color="red", linewidth=1.0)
    ax.set_xlim(0, fs/2)
    style_axes(ax, title, "Frekuensi (Hz)", "Magnitude")
    fig.tight_layout()
    return fig

def fig_spec2d(times, freqs, Z, fs):
    fig, ax = plt.subplots(figsize=(6.1, 4.2), dpi=120)
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="jet",
                   extent=[times[0], times[-1] if len(times)>1 else times[0]+0.01, freqs[0], freqs[-1]])
    ax.set_ylim(0, fs/2)
    ax.set_xlabel("Waktu (s)")
    ax.set_ylabel("Frekuensi (Hz)")
    ax.set_title("2D Spectrogram", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    return fig

def fig_spec3d(times, freqs, Z, fs):
    fig = plt.figure(figsize=(5.6, 4.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    step_f = max(1, len(freqs)//80)
    T, F = np.meshgrid(times, freqs[::step_f])
    ZZ = Z[::step_f, :]
    ax.plot_surface(T, F, ZZ, cmap="jet", linewidth=0, antialiased=False)
    ax.set_xlabel("Waktu (s)", fontsize=8)
    ax.set_ylabel("Frekuensi (Hz)", fontsize=8)
    ax.set_zlabel("Magnitude", fontsize=8)
    ax.set_title("3D Spectrogram", fontsize=11)
    ax.set_ylim(0, fs/2)
    ax.view_init(elev=28, azim=-55)
    fig.tight_layout()
    return fig

# ---------- Sidebar ----------
st.sidebar.markdown("## Data")
signal_mode = st.sidebar.radio("Pilih Sinyal:", ["Sinyal Buatan", "Sinyal ECG"], index=0)
n_data = st.sidebar.number_input("Jumlah Data:", min_value=128, max_value=4096, value=1024, step=1)
fs = st.sidebar.number_input("Frek. Sampling:", min_value=100, max_value=5000, value=1024, step=1)
st.sidebar.button("Proses Data")

st.sidebar.markdown('<div class="section-box"></div>', unsafe_allow_html=True)
st.sidebar.markdown("## Set Panjang Data")
mode_data = st.sidebar.radio("Pilih Mode:", ["Full Data", "Ambil Data ke :"], index=0)
c1, c2 = st.sidebar.columns(2)
start_cut = c1.number_input("Mulai", min_value=0, max_value=int(n_data)-1, value=min(280, int(n_data)-2), step=1)
end_cut = c2.number_input("Sampai", min_value=1, max_value=int(n_data), value=min(579, int(n_data)-1), step=1)
st.sidebar.button("Proses Potong Data")

st.sidebar.markdown('<div class="section-box"></div>', unsafe_allow_html=True)
st.sidebar.markdown("## Windowing")
window_name = st.sidebar.radio("Pilih Window:", ["Rectangular", "Bartlett", "Hanning", "Hamming", "Blackman"], index=2)
overlap = st.sidebar.number_input("Irisan (Overlap):", min_value=0, max_value=int(n_data)-1, value=50, step=1)
win_len = st.sidebar.number_input("Lebar Window:", min_value=8, max_value=int(n_data), value=100, step=1)
num_win = st.sidebar.number_input("Jumlah Window:", min_value=1, max_value=100, value=21, step=1)
run_stft = st.sidebar.button("STFT")

# ---------- Data processing ----------
t, y = make_signal(int(n_data), int(fs), signal_mode)
if mode_data == "Ambil Data ke :":
    s = int(min(start_cut, end_cut-1))
    e = int(max(start_cut+1, end_cut))
    y_proc = y[s:e]
else:
    s, e = 0, len(y)
    y_proc = y

wmax = max(0, (len(y)-int(win_len)) // max(int(win_len)-int(overlap), 1))
selected_w = st.session_state.get("selected_w", 0)
selected_w = int(np.clip(selected_w, 0, max(wmax, 0)))

# ---------- Main ----------
st.title("Time-Frequency Analysis - STFT")

col1, col2 = st.columns([2.25, 1.1], gap="medium")
with col1:
    st.pyplot(fig_signal(t, y, fs, "Sinyal Input"), use_container_width=True)
with col2:
    st.pyplot(fig_spectrum(y, fs, "Spectrum Sinyal Input"), use_container_width=True)

st.markdown("---")
st.markdown("**Pilih Window (w = 0 sampai 9)**")
selected_w = st.slider("", 0, 9, min(selected_w, 9), label_visibility="collapsed")
st.session_state["selected_w"] = selected_w
hop = max(int(win_len) - int(overlap), 1)
ws = min(selected_w * hop, max(0, len(y)-int(win_len)))
we = min(ws + int(win_len), len(y))
w = get_window(window_name, we-ws)
yw = y[ws:we] * w
pad = np.zeros_like(y)
pad[ws:we] = yw

c3, c4 = st.columns([2.25, 1.1], gap="medium")
with c3:
    st.pyplot(fig_signal(t, y, fs, f"Sinyal Hasil Windowing (w = {selected_w}, Data = {ws}->{we-1})", (ws, we, yw)), use_container_width=True)
with c4:
    st.pyplot(fig_spectrum(pad, fs, "Spectrum Sinyal Hasil Windowing"), use_container_width=True)

st.markdown("### 2D Spectrogram & 3D Spectrogram")
times, freqs, Z = stft_calc(y_proc, int(fs), int(win_len), int(overlap), int(num_win), window_name)
c5, c6 = st.columns([1.2, 1], gap="medium")
with c5:
    st.pyplot(fig_spec2d(times, freqs, Z, int(fs)), use_container_width=True)
with c6:
    st.pyplot(fig_spec3d(times, freqs, Z, int(fs)), use_container_width=True)

st.caption("Nilai di panel kiri dapat diubah: jumlah data, frekuensi sampling, mode potong data, jenis window, overlap, lebar window, dan jumlah window.")
