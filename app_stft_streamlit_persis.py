import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


st.set_page_config(
    page_title="Time-Frequency Analysis - STFT",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0b0f17;
}

[data-testid="stSidebar"] {
    background-color: #242633;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.block-container {
    padding-top: 4rem;
    padding-left: 4rem;
    padding-right: 4rem;
    max-width: 1500px;
}

h1 {
    color: white;
    font-size: 42px !important;
    font-weight: 800 !important;
}

h3 {
    color: white;
}

hr {
    border: 1px solid #333842;
}

.stButton > button {
    background-color: #ff4b5c;
    color: white;
    border-radius: 8px;
    border: none;
    height: 42px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #ff6b78;
    color: white;
}

.stNumberInput input {
    background-color: #080d15 !important;
    color: white !important;
    border-radius: 8px;
}

.stSlider label {
    color: white !important;
}

.plot-card {
    background-color: white;
    padding: 10px;
    border-radius: 7px;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)


def generate_signal(n_data, fs):
    n = np.arange(n_data)
    t = n / fs
    x = np.zeros(n_data)

    batas1 = n_data // 4
    batas2 = n_data // 2
    batas3 = 3 * n_data // 4

    x[:batas1] = np.sin(2 * np.pi * 50 * t[:batas1])
    x[batas1:batas2] = np.sin(2 * np.pi * 150 * t[batas1:batas2])
    x[batas2:batas3] = np.sin(2 * np.pi * 250 * t[batas2:batas3])
    x[batas3:] = np.sin(2 * np.pi * 350 * t[batas3:])

    return n, t, x


def generate_ecg_like(n_data, fs):
    n = np.arange(n_data)
    t = n / fs
    x = (
        0.8 * np.sin(2 * np.pi * 1.2 * t)
        + 0.25 * np.sin(2 * np.pi * 2.4 * t)
        + 0.15 * np.sin(2 * np.pi * 50 * t)
    )
    return n, t, x


def get_window(window_type, width):
    if window_type == "Rectangular":
        return np.ones(width)
    if window_type == "Bartlett":
        return np.bartlett(width)
    if window_type == "Hanning":
        return np.hanning(width)
    if window_type == "Hamming":
        return np.hamming(width)
    if window_type == "Blackman":
        return np.blackman(width)
    return np.ones(width)


def spectrum(x, fs):
    y = np.abs(np.fft.rfft(x)) / len(x)
    f = np.fft.rfftfreq(len(x), d=1 / fs)
    return f, y


def compute_stft(x, fs, width, overlap, jumlah_window, window_type):
    width = int(width)
    overlap = int(overlap)
    step = max(1, width - overlap)

    win = get_window(window_type, width)
    specs = []
    times = []

    for i in range(jumlah_window):
        start = i * step
        end = start + width

        if end > len(x):
            segment = np.zeros(width)
            available = len(x) - start
            if available > 0:
                segment[:available] = x[start:]
        else:
            segment = x[start:end]

        segment_windowed = segment * win
        mag = np.abs(np.fft.rfft(segment_windowed)) / width
        specs.append(mag)
        times.append((start + width / 2) / fs)

    specs = np.array(specs).T
    freqs = np.fft.rfftfreq(width, d=1 / fs)

    return np.array(times), freqs, specs


def fig_signal(n, x, title, color="red", width=8, height=2.6):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    ax.plot(n, x, color=color, linewidth=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("n sample", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_ylim(-1.15, 1.15)
    fig.tight_layout()
    return fig


def fig_spectrum(f, mag, title, width=4.2, height=2.6):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    ax.plot(f, mag, color="red", linewidth=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Frekuensi (Hz)", fontsize=7)
    ax.set_ylabel("Magnitude", fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_xlim(0, 500)
    fig.tight_layout()
    return fig


def fig_windowed(n, x, fs, width_win, overlap, w_index, window_type, jumlah_window):
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=120)

    ax.plot(n, x, color="lightgray", linewidth=0.7, alpha=0.45)

    step = max(1, width_win - overlap)
    win = get_window(window_type, width_win)

    colors = ["red", "blue", "black"]
    labels = ["w=0", "w=1", "w=2"]

    for i in range(min(3, jumlah_window)):
        start = (w_index + i) * step
        end = start + width_win
        if start >= len(x):
            continue

        seg = np.zeros(width_win)
        available = min(width_win, len(x) - start)
        seg[:available] = x[start:start + available]

        seg_win = seg * win
        nn = np.arange(start, start + width_win)

        ax.plot(
            nn,
            seg_win,
            color=colors[i],
            linewidth=1,
            label=labels[i]
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"Sinyal Hasil Windowing (Overlap w={w_index}, {w_index+1}, {w_index+2})",
        fontsize=11
    )
    ax.set_xlabel("n sample", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_ylim(-1.15, 1.15)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def fig_specgram(times, freqs, spec):
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=120)
    im = ax.pcolormesh(times, freqs, spec, shading="auto", cmap="jet")
    ax.set_title("")
    ax.set_xlabel("Waktu (s)")
    ax.set_ylabel("Frekuensi (Hz)")
    ax.set_ylim(0, 500)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def fig_3d(times, freqs, spec):
    fig = plt.figure(figsize=(6.6, 4.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    T, F = np.meshgrid(times, freqs)
    ax.plot_surface(T, F, spec, cmap="jet", linewidth=0, antialiased=True)

    ax.set_xlabel("Waktu (s)", fontsize=8)
    ax.set_ylabel("Frekuensi (Hz)", fontsize=8)
    ax.set_zlabel("Magnitude", fontsize=8)
    ax.view_init(elev=28, azim=-55)
    fig.tight_layout()
    return fig


# =========================
# SIDEBAR
# =========================

st.sidebar.markdown("## Data")

pilih_sinyal = st.sidebar.radio(
    "Pilih Sinyal:",
    ["Sinyal Buatan", "Sinyal ECG"]
)

jumlah_data = st.sidebar.number_input(
    "Jumlah Data:",
    min_value=128,
    max_value=4096,
    value=1024,
    step=1
)

fs = st.sidebar.number_input(
    "Frek. Sampling:",
    min_value=100,
    max_value=5000,
    value=1024,
    step=1
)

st.sidebar.button("Proses Data")

st.sidebar.markdown("---")
st.sidebar.markdown("## Set Panjang Data")

mode_data = st.sidebar.radio(
    "Pilih Mode:",
    ["Full Data", "Ambil Data ke :"]
)

col_start, col_end = st.sidebar.columns(2)

with col_start:
    mulai = st.number_input(
        "Mulai",
        min_value=0,
        max_value=int(jumlah_data) - 1,
        value=280,
        step=1
    )

with col_end:
    sampai = st.number_input(
        "Sampai",
        min_value=1,
        max_value=int(jumlah_data),
        value=579,
        step=1
    )

st.sidebar.button("Proses Potong Data")

st.sidebar.markdown("---")
st.sidebar.markdown("## Windowing")

window_type = st.sidebar.radio(
    "Pilih Window:",
    ["Rectangular", "Bartlett", "Hanning", "Hamming", "Blackman"],
    index=2
)

overlap = st.sidebar.number_input(
    "Irisan (Overlap):",
    min_value=0,
    max_value=1000,
    value=50,
    step=1
)

lebar_window = st.sidebar.number_input(
    "Lebar Window:",
    min_value=16,
    max_value=1024,
    value=100,
    step=1
)

jumlah_window = st.sidebar.number_input(
    "Jumlah Window:",
    min_value=1,
    max_value=100,
    value=20,
    step=1
)

w_index = st.sidebar.slider(
    "Pilih Window:",
    min_value=0,
    max_value=max(0, int(jumlah_window) - 1),
    value=0,
    step=1
)

st.sidebar.button("Proses Windowing")
st.sidebar.button("STFT")


# =========================
# DATA
# =========================

jumlah_data = int(jumlah_data)
fs = int(fs)
overlap = int(overlap)
lebar_window = int(lebar_window)
jumlah_window = int(jumlah_window)

if pilih_sinyal == "Sinyal Buatan":
    n, t, x = generate_signal(jumlah_data, fs)
else:
    n, t, x = generate_ecg_like(jumlah_data, fs)

if mode_data == "Ambil Data ke :":
    mulai = int(mulai)
    sampai = int(sampai)
    if sampai <= mulai:
        sampai = mulai + 1
    x_process = np.zeros_like(x)
    x_process[mulai:sampai] = x[mulai:sampai]
else:
    x_process = x.copy()


# =========================
# MAIN CONTENT
# =========================

st.title("Time-Frequency Analysis - STFT")

f_in, mag_in = spectrum(x, fs)

col1, col2 = st.columns([2.2, 1])

with col1:
    st.pyplot(fig_signal(n, x, "Sinyal Input", width=8, height=2.7), use_container_width=True)

with col2:
    st.pyplot(fig_spectrum(f_in, mag_in, "Spectrum Sinyal Input", width=4.1, height=2.7), use_container_width=True)

st.markdown("---")

col3, col4 = st.columns([2.2, 1])

with col3:
    st.pyplot(
        fig_windowed(
            n,
            x_process,
            fs,
            lebar_window,
            overlap,
            w_index,
            window_type,
            jumlah_window
        ),
        use_container_width=True
    )

with col4:
    step = max(1, lebar_window - overlap)
    start = w_index * step
    end = start + lebar_window

    segment = np.zeros(lebar_window)
    if start < len(x_process):
        available = min(lebar_window, len(x_process) - start)
        segment[:available] = x_process[start:start + available]

    segment = segment * get_window(window_type, lebar_window)
    f_win, mag_win = spectrum(segment, fs)

    st.pyplot(
        fig_spectrum(
            f_win,
            mag_win,
            "Spectrum Sinyal Hasil Windowing",
            width=4.1,
            height=2.7
        ),
        use_container_width=True
    )

times, freqs, spec = compute_stft(
    x_process,
    fs,
    lebar_window,
    overlap,
    jumlah_window,
    window_type
)

col5, col6 = st.columns([1.15, 1])

with col5:
    st.markdown("### 2D Spectrogram")
    st.pyplot(fig_specgram(times, freqs, spec), use_container_width=True)

with col6:
    st.markdown("### 3D Spectrogram")
    st.pyplot(fig_3d(times, freqs, spec), use_container_width=True)
