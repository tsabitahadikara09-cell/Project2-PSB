import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


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
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNGSI DATA
# ============================================================

def generate_signal(n_data, fs):
    n = np.arange(n_data)
    t = n / fs
    x = np.zeros(n_data)

    b1 = n_data // 4
    b2 = n_data // 2
    b3 = 3 * n_data // 4

    x[:b1] = np.sin(2 * np.pi * 50 * t[:b1])
    x[b1:b2] = np.sin(2 * np.pi * 150 * t[b1:b2])
    x[b2:b3] = np.sin(2 * np.pi * 250 * t[b2:b3])
    x[b3:] = np.sin(2 * np.pi * 350 * t[b3:])

    return n, t, x


def generate_ecg_like(n_data, fs):
    n = np.arange(n_data)
    t = n / fs

    x = 0.05 * np.sin(2 * np.pi * 0.5 * t)

    rr = int(fs * 0.8)
    if rr < 1:
        rr = 1

    for r in range(int(0.3 * fs), n_data, rr):
        qrs_width = max(1, int(0.025 * fs))
        p_width = max(1, int(0.05 * fs))
        t_width = max(1, int(0.12 * fs))

        x += 1.0 * np.exp(-((n - r) ** 2) / (2 * qrs_width ** 2))
        x += 0.18 * np.exp(-((n - (r - int(0.18 * fs))) ** 2) / (2 * p_width ** 2))
        x += 0.30 * np.exp(-((n - (r + int(0.25 * fs))) ** 2) / (2 * t_width ** 2))

    x = x - np.mean(x)
    max_val = np.max(np.abs(x))

    if max_val != 0:
        x = x / max_val

    return n, t, x


def read_ecg_file(uploaded_file):
    try:
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
        values = []

        for line in raw.splitlines():
            line = line.strip()

            if line == "":
                continue

            line = line.replace(",", " ")
            parts = line.split()

            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except:
                    pass

            if len(nums) == 1:
                values.append(nums[0])
            elif len(nums) >= 2:
                values.append(nums[1])

        if len(values) < 10:
            return None

        x = np.array(values, dtype=float)
        x = x - np.mean(x)

        max_val = np.max(np.abs(x))
        if max_val != 0:
            x = x / max_val

        return x

    except:
        return None


# ============================================================
# FUNGSI WINDOW, FFT, STFT
# ============================================================

def get_window(window_type, width):
    width = int(width)

    if width < 1:
        width = 1

    if window_type == "Rectangular":
        return np.ones(width)
    elif window_type == "Bartlett":
        return np.bartlett(width)
    elif window_type == "Hanning":
        return np.hanning(width)
    elif window_type == "Hamming":
        return np.hamming(width)
    elif window_type == "Blackman":
        return np.blackman(width)

    return np.ones(width)


def spectrum(x, fs):
    if len(x) < 2:
        x = np.array([0, 0])

    y = np.abs(np.fft.rfft(x)) / len(x)
    f = np.fft.rfftfreq(len(x), d=1 / fs)

    return f, y


def compute_stft(x, fs, width, overlap, jumlah_window, window_type):
    width = int(width)
    overlap = int(overlap)
    jumlah_window = int(jumlah_window)

    if width < 2:
        width = 2

    if overlap >= width:
        overlap = width - 1

    if overlap < 0:
        overlap = 0

    step = width - overlap
    win = get_window(window_type, width)

    specs = []
    times = []

    for i in range(jumlah_window):
        start = i * step
        segment = np.zeros(width)

        if start < len(x):
            available = min(width, len(x) - start)
            segment[:available] = x[start:start + available]

        segment_windowed = segment * win
        mag = np.abs(np.fft.rfft(segment_windowed)) / width

        specs.append(mag)
        times.append((start + width / 2) / fs)

    specs = np.array(specs).T
    freqs = np.fft.rfftfreq(width, d=1 / fs)

    return np.array(times), freqs, specs


# ============================================================
# FUNGSI GAMBAR
# ============================================================

def fig_signal(n, x, title, width=8, height=2.7):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)

    ax.plot(n, x, color="red", linewidth=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("n sample", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_ylim(-1.15, 1.15)

    fig.tight_layout()
    return fig


def fig_spectrum(f, mag, title, width=4.2, height=2.7):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)

    ax.plot(f, mag, color="red", linewidth=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Frekuensi (Hz)", fontsize=7)
    ax.set_ylabel("Magnitude", fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_xlim(0, min(500, max(f) if len(f) > 0 else 500))

    fig.tight_layout()
    return fig


def fig_windowed(n, x, width_win, overlap, w_index, window_type, jumlah_window):
    fig, ax = plt.subplots(figsize=(8, 2.7), dpi=120)

    ax.plot(n, x, color="lightgray", linewidth=0.7, alpha=0.45)

    if overlap >= width_win:
        overlap = width_win - 1

    step = max(1, width_win - overlap)
    win = get_window(window_type, width_win)

    colors = ["red", "blue", "black"]
    labels = [f"w={w_index}", f"w={w_index + 1}", f"w={w_index + 2}"]

    for i in range(min(3, jumlah_window)):
        start = (w_index + i) * step

        if start >= len(x):
            continue

        segment = np.zeros(width_win)
        available = min(width_win, len(x) - start)
        segment[:available] = x[start:start + available]

        segment_windowed = segment * win
        nn = np.arange(start, start + width_win)

        ax.plot(
            nn,
            segment_windowed,
            color=colors[i],
            linewidth=1,
            label=labels[i]
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"Sinyal Hasil Windowing (Overlap w={w_index}, {w_index + 1}, {w_index + 2})",
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
    fig, ax = plt.subplots(figsize=(6.3, 4.1), dpi=120)

    im = ax.pcolormesh(times, freqs, spec, shading="auto", cmap="jet")
    ax.set_xlabel("Waktu (s)")
    ax.set_ylabel("Frekuensi (Hz)")
    ax.set_ylim(0, min(500, max(freqs) if len(freqs) > 0 else 500))

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def fig_3d(times, freqs, spec):
    fig = plt.figure(figsize=(6.3, 4.1), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    T, F = np.meshgrid(times, freqs)

    ax.plot_surface(
        T,
        F,
        spec,
        cmap="jet",
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel("Waktu (s)", fontsize=8)
    ax.set_ylabel("Frekuensi (Hz)", fontsize=8)
    ax.set_zlabel("Magnitude", fontsize=8)
    ax.view_init(elev=28, azim=-55)

    fig.tight_layout()
    return fig


# ============================================================
# SIDEBAR - DATA
# ============================================================

st.sidebar.markdown("## Data")

pilih_sinyal = st.sidebar.radio(
    "Pilih Sinyal:",
    ["Sinyal Buatan", "Sinyal ECG"]
)

uploaded_ecg = None

if pilih_sinyal == "Sinyal ECG":
    uploaded_ecg = st.sidebar.file_uploader(
        "Upload Data ECG:",
        type=["txt", "csv", "dat"]
    )

jumlah_data_input = st.sidebar.number_input(
    "Jumlah Data:",
    min_value=50,
    max_value=10000,
    value=1024,
    step=1
)

fs = st.sidebar.number_input(
    "Frek. Sampling:",
    min_value=1,
    max_value=10000,
    value=1024,
    step=1
)

st.sidebar.button("Proses Data")


# ============================================================
# LOAD DATA DULU SUPAYA JUMLAH DATA AKTUAL DIKETAHUI
# ============================================================

fs = int(fs)
jumlah_data_input = int(jumlah_data_input)

if pilih_sinyal == "Sinyal Buatan":
    n, t, x = generate_signal(jumlah_data_input, fs)
    jumlah_data = len(x)

else:
    if uploaded_ecg is not None:
        ecg_data = read_ecg_file(uploaded_ecg)

        if ecg_data is not None:
            x_full = ecg_data

            if jumlah_data_input < len(x_full):
                x = x_full[:jumlah_data_input]
            else:
                x = x_full

            jumlah_data = len(x)
            n = np.arange(jumlah_data)
            t = n / fs
            st.sidebar.success(f"Data ECG berhasil dibaca: {len(x_full)} data")

        else:
            st.sidebar.error("File ECG tidak bisa dibaca")
            n, t, x = generate_ecg_like(jumlah_data_input, fs)
            jumlah_data = len(x)

    else:
        n, t, x = generate_ecg_like(jumlah_data_input, fs)
        jumlah_data = len(x)
        st.sidebar.info("Belum upload ECG, memakai contoh ECG")


# ============================================================
# SIDEBAR - SET PANJANG DATA
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("## Set Panjang Data")

mode_data = st.sidebar.radio(
    "Pilih Mode:",
    ["Full Data", "Ambil Data ke :"]
)

max_data = int(jumlah_data)

default_mulai = min(280, max_data - 2)
if default_mulai < 0:
    default_mulai = 0

default_sampai = min(579, max_data)
if default_sampai <= default_mulai:
    default_sampai = default_mulai + 1

col_start, col_end = st.sidebar.columns(2)

with col_start:
    mulai = st.number_input(
        "Mulai",
        min_value=0,
        max_value=max(1, max_data - 2),
        value=default_mulai,
        step=1
    )

with col_end:
    sampai = st.number_input(
        "Sampai",
        min_value=int(mulai) + 1,
        max_value=max_data,
        value=min(default_sampai, max_data),
        step=1
    )

st.sidebar.button("Proses Potong Data")


# ============================================================
# SIDEBAR - WINDOWING
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("## Windowing")

window_type = st.sidebar.radio(
    "Pilih Window:",
    ["Rectangular", "Bartlett", "Hanning", "Hamming", "Blackman"],
    index=2
)

lebar_window = st.sidebar.number_input(
    "Lebar Window:",
    min_value=8,
    max_value=max(8, max_data),
    value=min(100, max_data),
    step=1
)

overlap = st.sidebar.number_input(
    "Irisan (Overlap):",
    min_value=0,
    max_value=max(0, int(lebar_window) - 1),
    value=min(50, max(0, int(lebar_window) - 1)),
    step=1
)

jumlah_window = st.sidebar.number_input(
    "Jumlah Window:",
    min_value=1,
    max_value=200,
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


# ============================================================
# PROSES POTONG DATA
# ============================================================

mulai = int(mulai)
sampai = int(sampai)
lebar_window = int(lebar_window)
overlap = int(overlap)
jumlah_window = int(jumlah_window)
w_index = int(w_index)

if mode_data == "Ambil Data ke :":
    mulai = max(0, min(mulai, len(x) - 1))
    sampai = max(mulai + 1, min(sampai, len(x)))

    x_process = np.zeros_like(x)
    x_process[mulai:sampai] = x[mulai:sampai]

else:
    x_process = x.copy()


# ============================================================
# MAIN DISPLAY
# ============================================================

st.title("Time-Frequency Analysis - STFT")

f_in, mag_in = spectrum(x, fs)

col1, col2 = st.columns([2.2, 1])

with col1:
    st.pyplot(
        fig_signal(n, x, "Sinyal Input", width=8, height=2.7),
        use_container_width=True
    )

with col2:
    st.pyplot(
        fig_spectrum(f_in, mag_in, "Spectrum Sinyal Input", width=4.1, height=2.7),
        use_container_width=True
    )

st.markdown("---")

col3, col4 = st.columns([2.2, 1])

with col3:
    st.pyplot(
        fig_windowed(
            n,
            x_process,
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
    st.pyplot(
        fig_specgram(times, freqs, spec),
        use_container_width=True
    )

with col6:
    st.markdown("### 3D Spectrogram")
    st.pyplot(
        fig_3d(times, freqs, spec),
        use_container_width=True
    )
