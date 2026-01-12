import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from collections import deque

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


CSV = "dataset_russia_subjects_normalized.csv"
FEATS = ["t2m", "swvl1", "tp", "lai_hv"]
OPS_RU = ["Полив", "Удобрение", "Опрыскивание", "Культивация"]
MAX_SIDE = 220


def culture_mult(c):
    c = (c or "").lower()
    return {
        "пшеница": np.array([1, 1, 1, 1.0]),
        "рис": np.array([1, 0.8, 1.1, 1.0]),
        "картофель": np.array([1, 1, 1, 1.1]),
        "кукуруза": np.array([1.1, 1, 1, 1.0]),
        "подсолнечник": np.array([1, 1.1, 1, 0.9]),
    }.get(c, np.array([1, 1, 1, 1.0]))


def generate_cell(mu, rng):
    return mu + rng.normal(0, 0.6, 4)


def label_cell(x):
    sig = lambda z: 1 / (1 + np.exp(-z))
    t, m, r, l = sig(x[0]), sig(x[1]), sig(x[2]), sig(0.7 * x[3])
    stress = (abs(m - 0.55) + abs(t - 0.55) + max(0, 0.6 - r)) / 3
    s = (
        np.array(
            [
                0.7 * (1 - m) + 0.2 * (1 - r) + 0.3 * stress,
                0.85 * (1 - l) + 0.15 * stress + 0.1 * r,
                0.6 * r + 0.4 * l,
                0.7 * r + 0.3 * stress,
            ]
        )
        + np.random.normal(0, 0.02, 4)
    )
    return (s > np.array([0.55, 0.48, 0.55, 0.55])).astype(int)


def build_action_net():
    m = Sequential(
        [
            Dense(128, activation="relu", input_dim=4),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(4, activation="sigmoid"),
        ]
    )
    m.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    return m


def fit_scalers(df):
    s = {"__g__": StandardScaler().fit(df[FEATS])}
    for r, sub in df.groupby("region_id"):
        s[r] = StandardScaler().fit(sub[FEATS])
    return s


@st.cache_resource
def train_action_net():
    df = pd.read_csv(CSV)
    regs = sorted(df["region_id"].dropna().unique())
    scalers = fit_scalers(df)
    rng = np.random.default_rng(1)
    cultures = ["Пшеница", "Рис", "Картофель", "Кукуруза", "Подсолнечник"]

    X, Y = [], []
    for _ in range(12000):
        r = rng.choice(regs)
        c = rng.choice(cultures)
        sub = df[df.region_id == r]
        mu = sub[FEATS].mean().values
        sc = scalers.get(r, scalers["__g__"])
        x = sc.transform([generate_cell(mu, rng)])[0] * culture_mult(c)
        X.append(x)
        Y.append(label_cell(x))

    X, Y = np.array(X), np.array(Y)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=1)

    m = build_action_net()
    m.fit(Xtr, Ytr, epochs=10, batch_size=256, verbose=0)
    p = (m.predict(Xte, verbose=0) > 0.5).astype(int)

    return df, regs, scalers, m, {
        "hamming": hamming_loss(Yte, p),
        "f1": f1_score(Yte, p, average="micro"),
    }


def grid_size(w, h, spm):
    H0, W0 = int(h * spm), int(w * spm)
    s = max(1, int(np.ceil(max(H0 / MAX_SIDE, W0 / MAX_SIDE))))
    return max(1, H0 // s), max(1, W0 // s), s / spm


# =========================
# "УМНАЯ ЗМЕЙКА"
# =========================
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def in_bounds(r, c, H, W):
    return 0 <= r < H and 0 <= c < W


def bfs_path(start, goal, H, W):
    """Кратчайший путь по сетке без препятствий."""
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc, H, W):
                continue
            if (nr, nc) in prev:
                continue
            prev[(nr, nc)] = (r, c)
            if (nr, nc) == goal:
                path = [goal]
                cur = goal
                while prev[cur] is not None:
                    cur = prev[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append((nr, nc))
    return [start]


def smart_snake_route(ops_mask, order=(0, 1, 2, 3), start=(0, 0)):
    """
    ops_mask: bool array (H,W,4) где ops_mask[r,c,i]=True если нужна операция i
    order: порядок операций (0..3) — чтобы одинаковые делались подряд
    start: стартовая клетка
    Возвращает path [(r,c), ...]
    """
    H, W, K = ops_mask.shape
    cur = start
    path = [cur]

    def add_move(to_cell):
        nonlocal cur, path
        seg = bfs_path(cur, to_cell, H, W)
        path.extend(seg[1:])
        cur = to_cell

    for op in order:
        # собираем строки, где есть эта операция: (r, cmin, cmax)
        rows = []
        for r in range(H):
            cols = np.where(ops_mask[r, :, op])[0]
            if len(cols) == 0:
                continue
            rows.append((r, int(cols.min()), int(cols.max())))

        rows.sort(key=lambda x: x[0])

        # идём по строкам "змейкой", но только по сегменту [cmin..cmax]
        for idx, (r, cmin, cmax) in enumerate(rows):
            left_to_right = (idx % 2 == 0)
            entry = (r, cmin) if left_to_right else (r, cmax)

            # перейти к входу строки кратчайшим путём
            add_move(entry)

            # пройти по сегменту
            if left_to_right:
                for c in range(cmin, cmax + 1):
                    if (r, c) != cur:
                        path.append((r, c))
                        cur = (r, c)
            else:
                for c in range(cmax, cmin - 1, -1):
                    if (r, c) != cur:
                        path.append((r, c))
                        cur = (r, c)

    return path


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("Автоматизация сельскохозяйственных операций")

df, regions, scalers, action_model, metrics = train_action_net()

with st.expander("Метрики модели операций"):
    st.write(metrics)

c1, c2, c3 = st.columns(3)
w = c1.number_input("Ширина поля (м)", 1, 1_000_000, 200, 10)
h = c2.number_input("Длина поля (м)", 1, 1_000_000, 200, 10)
spm = c3.number_input("Посевов в 1 метре", 1, 200, 1)

c4, c5 = st.columns(2)
reg = c4.selectbox("Регион", regions)
cul = c5.selectbox("Культура", ["Пшеница", "Рис", "Картофель", "Кукуруза", "Подсолнечник"])

H, W, cell = grid_size(float(w), float(h), int(spm))

# порядок выполнения одинаковых операций подряд
order = st.multiselect(
    "Порядок операций (слева направо)",
    options=list(range(4)),
    default=[0, 1, 2, 3],
    format_func=lambda i: OPS_RU[i],
)

start_r = st.number_input("Старт (r)", 0, max(0, H - 1), 0, 1)
start_c = st.number_input("Старт (c)", 0, max(0, W - 1), 0, 1)

if st.button("Сгенерировать"):
    rng = np.random.default_rng(2)
    sub = df[df.region_id == reg]
    mu = sub[FEATS].mean().values
    sc = scalers.get(reg, scalers["__g__"])

    raw = mu + rng.normal(0, 0.6, (H, W, 4))
    x = sc.transform(raw.reshape(-1, 4)) * culture_mult(cul)

    act = (action_model.predict(x, verbose=0) > 0.5).astype(int).reshape(H, W, 4)
    mask_any = act.any(axis=-1)

    # Карты операций
    fig1, axs = plt.subplots(2, 2, figsize=(9, 9))
    for i in range(4):
        axs.flat[i].imshow(act[..., i])
        axs.flat[i].set_title(OPS_RU[i])
        axs.flat[i].axis("off")
    st.pyplot(fig1, clear_figure=True)

    # Умная змейка: по операциям и по сегментам
    ops_mask = act.astype(bool)  # (H,W,4)
    p = smart_snake_route(ops_mask, order=tuple(order), start=(int(start_r), int(start_c)))

    # Визуализация маршрута поверх "есть работа"
    fig2, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(mask_any)

    rr = [i[0] for i in p]
    cc = [i[1] for i in p]
    ax.plot(cc, rr, linewidth=3)
    ax.scatter([cc[0]], [rr[0]], s=120)
    ax.scatter([cc[-1]], [rr[-1]], s=120, marker="X")

    ax.axis("off")
    st.pyplot(fig2, clear_figure=True)

    st.caption(f"Длина маршрута (в клетках): {max(0, len(p)-1)} | Размер сетки: {H}×{W} | Клетка ~ {cell:.2f} м")
