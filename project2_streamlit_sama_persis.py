import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# TIME-FREQUENCY ANALYSIS - STFT v1.0
# Versi Streamlit dibuat menyerupai tampilan Delphi pada gambar
# ============================================================

st.set_page_config(page_title="Time-Frequency Analysis - STFT", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background-color: #f5f5f5;}
.block-container {padding-top: 0.2rem; padding-left: 0.3rem; padding-right: 0.3rem; max-width: 100%;}
[data-testid="stSidebar"] {background-color: #00f000; border-right: 4px solid #00bb00;}
[data-testid="stSidebar"] * {font-size: 12px !important; color: black !important;}
.stButton>button {height: 28px; padding: 0px 18px; border: 2px solid #777; border-radius: 0px; background:#efefef; color:black; font-size:12px;}
div[data-testid="stNumberInput"] input {height: 24px; font-size: 12px; background: white; color: black; border-radius: 0px;}
.main-title {font-size: 15px; font-weight: 500; color: black; margin: 0 0 3px 2px;}
.green-box {border: 3px solid #00ff00; padding: 6px; margin-bottom: 7px; background: #00f000;}
.box-title {font-weight: 700; margin-bottom: 2px;}
hr {margin: 0.2rem 0;}
</style>
""", unsafe_allow_html=True)

# ---------------- Fungsi hitung ----------------
def generate_signal(n_data: int, fs: int):
    n = np.arange(n_data)
    x = np.zeros(n_data)
    # dibuat sama seperti tampilan: frekuensi berubah tiap 256 sampel
    parts = [(0, 256, 50), (256, 512, 150), (512, 768, 250), (768, n_data, 350)]
    for a, b, f in parts:
        b = min(b, n_data)
        idx = np.arange(a, b)
        x[a:b] = np.sin(2 * np.pi * f * idx / fs)
    return n, x

def generate_ecg(n_data: int, fs: int):
    t = np.arange(n_data) / fs
    x = 0.05*np.sin(2*np.pi*0.5*t)
    rr = max(int(0.8*fs), 1)
    idx = np.arange(n_data)
    for p in range(70, n_data, rr):
        x += 1.0*np.exp(-0.5*((idx-p)/(0.012*fs+1e-9))**2)
        x += 0.25*np.exp(-0.5*((idx-(p+int(0.18*fs)))/(0.045*fs+1e-9))**2)
        x -= 0.15*np.exp(-0.5*((idx-(p-int(0.04*fs)))/(0.018*fs+1e-9))**2)
    return idx, x

def make_window(kind: str, N: int):
    if N <= 1:
        return np.ones(max(N, 1))
    n = np.arange(N)
    if kind == "Rectangular":
        return np.ones(N)
    if kind == "Bartlett":
        return 1 - 2*np.abs(n-(N-1)/2)/(N-1)
    if kind == "Hanning":
        return 0.5 - 0.5*np.cos(2*np.pi*n/(N-1))
    if kind == "Hamming":
        return 0.54 - 0.46*np.cos(2*np.pi*n/(N-1))
    if kind == "Blackman":
        return 0.42 - 0.5*np.cos(2*np.pi*n/(N-1)) + 0.08*np.cos(4*np.pi*n/(N-1))
    return np.ones(N)

def fft_spectrum(x, fs):
    N = 1
    while N < len(x):
        N *= 2
    X = np.fft.fft(x, n=N)
    half = N//2
    f = np.arange(half+1)*fs/N
    mag = np.abs(X[:half+1])/N
    return f, mag

def stft_calc(x, fs, kind, overlap, win_len, fft_len=256):
    hop = win_len - overlap
    if hop <= 0:
        hop = 1
    if win_len > len(x):
        win_len = len(x)
    nwin = max(1, int(np.floor((len(x)-win_len)/hop))+1)
    w = make_window(kind, win_len)
    S = np.zeros((fft_len//2+1, nwin))
    centers = []
    for k in range(nwin):
        start = k*hop
        seg = x[start:start+win_len]
        if len(seg) < win_len:
            seg = np.pad(seg, (0, win_len-len(seg)))
        segw = seg*w
        X = np.fft.fft(segw, n=fft_len)
        S[:, k] = np.abs(X[:fft_len//2+1]) / fft_len
        centers.append(start + win_len/2)
    freqs = np.arange(fft_len//2+1)*fs/fft_len
    return S, freqs, np.array(centers), hop

# ---------------- Sidebar menyerupai panel hijau Delphi ----------------
with st.sidebar:
    st.markdown('<div class="green-box"><div class="box-title">Data</div>', unsafe_allow_html=True)
    data_type = st.radio("", ["Sinyal Buatan", "Sinyal ECG"], index=0, label_visibility="collapsed")
    n_data = st.number_input("Jumlah Data", min_value=64, max_value=4096, value=1024, step=1)
    fs = st.number_input("Frek. Sampling", min_value=64, max_value=10000, value=1024, step=1)
    st.button("Proses", key="p1")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="green-box"><div class="box-title">Set Panjang Data</div>', unsafe_allow_html=True)
    range_type = st.radio("", ["Full Data", "Ambil Data ke :"], index=0, label_visibility="collapsed")
    cmin, cmax = st.columns(2)
    with cmin:
        min_idx = st.number_input("", min_value=0, max_value=int(n_data)-1, value=280, step=1, label_visibility="collapsed")
    with cmax:
        max_idx = st.number_input("", min_value=0, max_value=int(n_data)-1, value=579, step=1, label_visibility="collapsed")
    st.button("Proses", key="p2")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="green-box"><div class="box-title">Windowing</div>', unsafe_allow_html=True)
    window_type = st.radio("", ["Rectangular", "Bartlett", "Hanning", "Hamming", "Blackman"], index=2, label_visibility="collapsed")
    irisan = st.number_input("Irisan", min_value=0, max_value=1000, value=50, step=1)
    win_len = st.number_input("Lebar Window", min_value=2, max_value=2048, value=100, step=1)
    jum_win_label = st.number_input("Jumlah Window", min_value=1, max_value=1000, value=21, step=1)
    st.button("Proses", key="p3")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="green-box">', unsafe_allow_html=True)
    st.button("STFT", key="stft")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Data ----------------
if data_type == "Sinyal Buatan":
    n, x = generate_signal(int(n_data), int(fs))
else:
    n, x = generate_ecg(int(n_data), int(fs))

if range_type == "Full Data":
    x_sel = x.copy()
    n_sel = np.arange(len(x_sel))
else:
    if max_idx < min_idx:
        max_idx = min_idx
    x_sel = x[int(min_idx):int(max_idx)+1]
    n_sel = np.arange(int(min_idx), int(max_idx)+1)

win_len = min(int(win_len), len(x_sel))
w = make_window(window_type, win_len)
windowed = np.zeros_like(x_sel)
windowed[:win_len] = x_sel[:win_len] * w

f_in, mag_in = fft_spectrum(x_sel, int(fs))
f_win, mag_win = fft_spectrum(windowed[:win_len], int(fs))
S, freqs, centers, hop = stft_calc(x_sel, int(fs), window_type, int(irisan), win_len, 256)

# ---------------- Plot style ----------------
def style_axes(ax):
    ax.grid(True, color="#777", alpha=0.45, linewidth=0.6)
    ax.tick_params(axis='both', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333")
        spine.set_linewidth(0.8)

def plot_line(xv, yv, title, xlabel, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(5.4, 1.55), dpi=120)
    ax.plot(xv, yv, color="red", linewidth=0.8)
    ax.set_title(title, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.margins(x=0)
    style_axes(ax)
    fig.tight_layout(pad=0.7)
    return fig

def plot_windowed():
    fig, ax = plt.subplots(figsize=(5.4, 1.55), dpi=120)
    colors = ["red", "blue", "black"]
    # menampilkan beberapa potongan window seperti gambar referensi
    starts = [0, int(win_len*0.45), int(win_len*0.9)]
    for i, s in enumerate(starts):
        e = min(s+win_len, len(x_sel))
        if e > s:
            seg = x_sel[s:e] * w[:e-s]
            ax.plot(np.arange(s, e), seg, color=colors[i % len(colors)], linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"Sinyal Hasil Windowing           w = 0, Data = (0->{win_len-1})", fontsize=7)
    ax.set_xlabel("n sample", fontsize=7)
    ax.set_ylabel("Amplitude", fontsize=7)
    ax.set_xlim(0, len(x_sel)-1)
    ax.set_ylim(-1.1, 1.1)
    style_axes(ax)
    fig.tight_layout(pad=0.7)
    return fig

def plot_stft_2d():
    fig, ax = plt.subplots(figsize=(7.1, 3.05), dpi=120)
    extent = [0, len(x_sel), freqs[0], freqs[-1]]
    vmax = np.percentile(S, 99) if np.max(S) > 0 else 1
    im = ax.imshow(S, aspect="auto", origin="lower", extent=extent, cmap="jet", vmin=0, vmax=vmax)
    ax.set_xlabel("n(sample)", fontsize=7)
    ax.set_ylabel("Frekuensi(Hz)", fontsize=7)
    style_axes(ax)
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055, pad=0.12)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.5)
    return fig

def plot_stft_3d():
    # downsample agar cepat dan menyerupai bentuk tangga 3D
    Z = S
    fmask = freqs <= min(900, freqs[-1])
    Z = Z[fmask, :]
    F = freqs[fmask]
    t = np.linspace(0, len(x_sel), Z.shape[1])
    step_f = max(1, len(F)//45)
    step_t = max(1, len(t)//30)
    F2 = F[::step_f]
    T2 = t[::step_t]
    Z2 = Z[::step_f, ::step_t]
    Tm, Fm = np.meshgrid(T2, F2)
    fig = plt.figure(figsize=(3.9, 2.55), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Tm, Fm, Z2, cmap="jet", linewidth=0, antialiased=False)
    ax.set_xlabel("n (sample)", fontsize=6, labelpad=-2)
    ax.set_ylabel("f (Hz)", fontsize=6, labelpad=-2)
    ax.set_zlabel("", fontsize=6)
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=28, azim=-55)
    fig.tight_layout(pad=0.2)
    return fig

# ---------------- Layout utama menyerupai screenshot ----------------
st.markdown('<div class="main-title">⚙️ Time-Frequency Analysis - STFT v1.0 @ Fauzan Arrofiiqi</div>', unsafe_allow_html=True)

left, right = st.columns([2.1, 1.15], gap="small")
with left:
    st.pyplot(plot_line(n_sel, x_sel, "Sinyal Input", "n sample", "Amplitude", ylim=(-1.1, 1.1)), use_container_width=True)
with right:
    st.pyplot(plot_line(f_in, mag_in, "Spectrum Sinyal Input", "Frekuensi (Hz)", "Magnitude"), use_container_width=True)

left2, right2 = st.columns([2.1, 1.15], gap="small")
with left2:
    st.pyplot(plot_windowed(), use_container_width=True)
with right2:
    st.pyplot(plot_line(f_win, mag_win, "Spectrum Sinyal Hasil Windowing", "Frekuensi (Hz)", "Magnitude"), use_container_width=True)

bottom1, bottom2 = st.columns([2.1, 1.15], gap="small")
with bottom1:
    st.pyplot(plot_stft_2d(), use_container_width=True)
with bottom2:
    st.pyplot(plot_stft_3d(), use_container_width=True)

st.caption("Versi Streamlit dibuat mengikuti layout gambar: panel hijau kiri, sinyal input, spektrum, windowing, STFT 2D, dan STFT 3D.")
