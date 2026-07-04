import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

st.set_page_config(page_title="Time-Frequency Analysis - STFT", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{--navy:#252633;--dark:#0e1117;--red:#ff4b4b;}
html, body, [data-testid="stAppViewContainer"]{background:#0e1117;color:white;font-family:Segoe UI,Arial,sans-serif;}
[data-testid="stHeader"]{background:#0e1117;height:48px;}
[data-testid="stToolbar"]{right:1rem;}
.block-container{padding-top:4.5rem;padding-left:5rem;padding-right:5rem;max-width:1500px;}
[data-testid="stSidebar"]{background:#252633 !important;border-right:1px solid #454650;min-width:300px;max-width:300px;}
[data-testid="stSidebar"] *{color:white !important;font-size:13px;}
[data-testid="stSidebar"] .block-container{padding-top:3.5rem;padding-left:1.35rem;padding-right:1.35rem;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-size:18px!important;margin-top:16px;margin-bottom:14px;}
[data-testid="stSidebar"] hr{margin:22px 0;border-color:#4b4c58;}
h1{font-size:42px!important;line-height:1.15;margin-bottom:18px;font-weight:800;color:#fff;}
h2{font-size:20px!important;margin-top:12px;margin-bottom:10px;color:#fff;}
.stRadio > label{font-weight:700;margin-bottom:6px;}
.stRadio div[role="radiogroup"] label{padding:1px 0;}
.stNumberInput label{font-weight:600;}
.stNumberInput{margin-bottom:9px;}
.stNumberInput input{background:#0b0f17!important;color:white!important;border:1px solid #1a1d26!important;border-radius:7px!important;height:40px!important;font-size:13px!important;}
.stNumberInput button{background:#0b0f17!important;border-color:#0b0f17!important;color:white!important;}
.stButton>button{background:#252633;color:white;border:1px solid #5b5c67;border-radius:7px;padding:.55rem .8rem;font-weight:700;}
.stButton>button:hover{background:#ff4b4b;color:white;border-color:#ff4b4b;}
.stSlider label{font-weight:700;}
.stSlider [data-testid="stThumbValue"]{color:#ff4b4b!important;}
.stSlider [data-baseweb="slider"] > div > div{background:#ff4b4b!important;}
.plot-card{background:#fff;border-radius:7px;padding:8px 8px 2px 8px;margin-bottom:18px;box-shadow:none;}
.small-spacer{height:18px;border-bottom:1px solid #343741;margin-bottom:26px;}
.sidebar-section{font-weight:800;font-size:18px;margin-top:8px;margin-bottom:16px;}
</style>
""", unsafe_allow_html=True)

# ---------- signal functions ----------
def generate_signal(n, fs, kind):
    t = np.arange(n) / fs
    if kind == "Sinyal ECG":
        # ECG-like synthetic waveform, no external data needed
        x = np.zeros(n)
        period = max(int(0.8 * fs), 1)
        for beat in range(0, n, period):
            for c, a, w in [(0.18, 0.18, 0.025), (0.36, -0.18, 0.012), (0.39, 1.0, 0.009), (0.42, -0.25, 0.014), (0.62, 0.32, 0.045)]:
                center = beat + int(c * period)
                idx = np.arange(n)
                x += a * np.exp(-0.5 * ((idx - center) / max(w * period, 1)) ** 2)
        x += 0.03*np.sin(2*np.pi*50*t)
        if np.max(np.abs(x)) > 0:
            x = x / np.max(np.abs(x))
        return x

    # Same style as Delphi/STFT demo: 4 frequency blocks
    x = np.zeros(n)
    blocks = [(0, n//4, 50), (n//4, n//2, 150), (n//2, 3*n//4, 250), (3*n//4, n, 350)]
    for a, b, f in blocks:
        x[a:b] = np.sin(2*np.pi*f*t[a:b])
    return x

def window_values(name, length):
    if length <= 1:
        return np.ones(max(length, 1))
    k = np.arange(length)
    if name == "Bartlett":
        return np.bartlett(length)
    if name == "Hanning":
        return np.hanning(length)
    if name == "Hamming":
        return np.hamming(length)
    if name == "Blackman":
        return np.blackman(length)
    return np.ones(length)

def spectrum(x, fs):
    y = np.abs(np.fft.rfft(x)) / max(len(x), 1)
    f = np.fft.rfftfreq(len(x), 1/fs)
    return f, y

def stft_manual(x, fs, win_len, overlap, jumlah, window_name):
    win_len = max(8, min(int(win_len), len(x)))
    overlap = max(0, min(int(overlap), win_len-1))
    step = max(1, win_len - overlap)
    max_start = max(0, len(x) - win_len)
    starts = [min(i*step, max_start) for i in range(max(1, int(jumlah)))]
    starts = sorted(list(dict.fromkeys(starts)))
    win = window_values(window_name, win_len)
    rows = []
    times = []
    for s in starts:
        seg = x[s:s+win_len] * win
        mag = np.abs(np.fft.rfft(seg)) / win_len
        rows.append(mag)
        times.append((s + win_len/2)/fs)
    S = np.array(rows).T
    freqs = np.fft.rfftfreq(win_len, 1/fs)
    return np.array(times), freqs, S, starts

# ---------- plot helpers ----------
def fig_line(width=7.6, height=2.35):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(True, linestyle=':', linewidth=0.55, color='0.70')
    ax.tick_params(labelsize=8, colors='black')
    for s in ax.spines.values():
        s.set_color('0.25')
    return fig, ax

def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def plot_signal(x):
    fig, ax = fig_line(8.0, 2.35)
    ax.plot(np.arange(len(x)), x, color='red', linewidth=0.9)
    ax.set_title('Sinyal Input', fontsize=10)
    ax.set_xlabel('n sample', fontsize=8)
    ax.set_ylabel('Amplitude', fontsize=8)
    ax.set_ylim(-1.15, 1.15)
    return fig

def plot_spec(x, fs, title='Spectrum Sinyal Input'):
    fig, ax = fig_line(4.05, 2.35)
    f, y = spectrum(x, fs)
    ax.plot(f, y, color='red', linewidth=0.9)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('Frekuensi (Hz)', fontsize=7)
    ax.set_ylabel('Magnitude', fontsize=7)
    ax.set_xlim(0, fs/2)
    return fig

def plot_windowed(x, win_len, overlap, w_index, window_name):
    fig, ax = fig_line(8.0, 2.35)
    ax.plot(np.arange(len(x)), x, color='0.83', linewidth=0.7, alpha=0.55)
    step = max(1, win_len-overlap)
    colors = ['red','blue','black']
    starts = []
    for k in range(3):
        s = min((w_index+k)*step, max(0, len(x)-win_len))
        starts.append(s)
        seg = x[s:s+win_len] * window_values(window_name, win_len)
        ax.plot(np.arange(s, s+len(seg)), seg, color=colors[k], linewidth=1.1, label=f'w={k}')
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_title('Sinyal Hasil Windowing (Overlap w=0, 1, 2)', fontsize=10)
    ax.set_xlabel('n sample', fontsize=8)
    ax.set_ylabel('Amplitude', fontsize=8)
    ax.set_ylim(-1.15, 1.15)
    ax.legend(fontsize=7, loc='upper right', frameon=True)
    return fig, starts[0]

def plot_stft2(t, f, S, fs):
    fig, ax = plt.subplots(figsize=(6.2, 3.95), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    if S.size == 0:
        S = np.zeros((10,10)); t=np.linspace(0,1,10); f=np.linspace(0,fs/2,10)
    im = ax.imshow(S, origin='lower', aspect='auto', cmap='jet', extent=[t.min(), t.max(), f.min(), min(f.max(), fs/2)], vmin=0, vmax=max(float(np.max(S)), 1e-9))
    ax.set_ylim(0, fs/2)
    ax.set_xlabel('Waktu (s)', fontsize=10)
    ax.set_ylabel('Frekuensi (Hz)', fontsize=10)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=8)
    return fig

def plot_stft3(t, f, S, fs):
    fig = plt.figure(figsize=(6.2, 3.95), dpi=120)
    fig.patch.set_facecolor('#0e1117')
    ax = fig.add_subplot(111, projection='3d')
    if S.size == 0:
        S = np.zeros((10,10)); t=np.linspace(0,1,10); f=np.linspace(0,fs/2,10)
    # reduce rows to keep fast
    fmask = f <= fs/2
    f2 = f[fmask]
    S2 = S[fmask, :]
    T, F = np.meshgrid(t, f2)
    surf = ax.plot_surface(T, F, S2, cmap='jet', linewidth=0, antialiased=False, rstride=1, cstride=1)
    ax.set_xlabel('Waktu (s)', fontsize=8, color='white')
    ax.set_ylabel('Frekuensi (Hz)', fontsize=8, color='white')
    ax.set_zlabel('Magnitude', fontsize=8, color='white')
    ax.tick_params(colors='white', labelsize=7)
    ax.set_facecolor('#0e1117')
    ax.view_init(elev=28, azim=-55)
    fig.colorbar(surf, shrink=0.78, aspect=12, pad=0.08)
    return fig

# ---------- sidebar ----------
with st.sidebar:
    st.markdown('<div class="sidebar-section">Data</div>', unsafe_allow_html=True)
    signal_kind = st.radio('Pilih Sinyal:', ['Sinyal Buatan', 'Sinyal ECG'], index=0)
    n = st.number_input('Jumlah Data:', min_value=128, max_value=4096, value=1024, step=1)
    fs = st.number_input('Frek. Sampling:', min_value=100, max_value=5000, value=1024, step=1)
    st.button('Proses Data')
    st.markdown('---')

    st.markdown('<div class="sidebar-section">Set Panjang Data</div>', unsafe_allow_html=True)
    mode = st.radio('Pilih Mode:', ['Full Data', 'Ambil Data ke :'], index=0)
    c1, c2 = st.columns(2)
    with c1:
        start_cut = st.number_input('Mulai', min_value=0, max_value=int(n)-1, value=min(280, int(n)-1), step=1)
    with c2:
        end_cut = st.number_input('Sampai', min_value=1, max_value=int(n), value=min(579, int(n)), step=1)
    st.button('Proses Potong Data')
    st.markdown('---')

    st.markdown('<div class="sidebar-section">Windowing</div>', unsafe_allow_html=True)
    window_name = st.radio('Pilih Window:', ['Rectangular','Bartlett','Hanning','Hamming','Blackman'], index=3)
    overlap = st.number_input('Irisan (Overlap):', min_value=0, max_value=1000, value=0, step=1)
    win_len = st.number_input('Lebar Window:', min_value=8, max_value=int(n), value=min(100, int(n)), step=1)
    jumlah_win = st.number_input('Jumlah Window:', min_value=1, max_value=100, value=20, step=1)
    st.button('STFT')

# ---------- main ----------
st.title('Time-Frequency Analysis - STFT')

n = int(n); fs = int(fs); win_len = int(win_len); overlap = int(overlap); jumlah_win = int(jumlah_win)
x = generate_signal(n, fs, signal_kind)
if mode == 'Ambil Data ke :':
    start_cut = int(max(0, min(start_cut, n-2)))
    end_cut = int(max(start_cut+1, min(end_cut, n)))
    x_proc = np.zeros_like(x)
    x_proc[start_cut:end_cut] = x[start_cut:end_cut]
else:
    x_proc = x.copy()

# Top row
col1, col2 = st.columns([2.05, 1.0], gap='medium')
with col1:
    show_fig(plot_signal(x_proc))
with col2:
    show_fig(plot_spec(x_proc, fs, 'Spectrum Sinyal Input'))

st.markdown('<div class="small-spacer"></div>', unsafe_allow_html=True)

# slider for selected window
max_w = max(0, jumlah_win-1)
w_index = st.slider('Pilih Window (w = 0 sampai %d)' % max_w, min_value=0, max_value=max_w, value=0, step=1)

col3, col4 = st.columns([2.05, 1.0], gap='medium')
with col3:
    figw, first_start = plot_windowed(x_proc, win_len, overlap, w_index, window_name)
    show_fig(figw)
with col4:
    seg = x_proc[first_start:first_start+win_len] * window_values(window_name, min(win_len, len(x_proc)-first_start))
    show_fig(plot_spec(seg if len(seg) else np.zeros(win_len), fs, 'Spectrum Sinyal Hasil Windowing'))

col5, col6 = st.columns([1.0, 1.0], gap='medium')
t, f, S, starts = stft_manual(x_proc, fs, win_len, overlap, jumlah_win, window_name)
with col5:
    st.subheader('2D Spectrogram')
    show_fig(plot_stft2(t, f, S, fs))
with col6:
    st.subheader('3D Spectrogram')
    show_fig(plot_stft3(t, f, S, fs))
