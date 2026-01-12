import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN
from tensorflow.keras.optimizers import Adam


# =======================
# ДАННЫЕ И НАСТРОЙКИ
# =======================

CSV = "dataset_russia_subjects_normalized.csv"
FEATS = ["t2m", "swvl1", "tp", "lai_hv"]
OPS_RU = ["Полив", "Удобрение", "Опрыскивание", "Культивация"]


# =======================
# КУЛЬТУРЫ
# =======================

def culture_mult(c):
    c = (c or "").lower()
    return {
        "пшеница": np.array([1,1,1,1.0]),
        "рис": np.array([1,0.8,1.1,1.0]),
        "картофель": np.array([1,1,1,1.1]),
        "кукуруза": np.array([1.1,1,1,1.0]),
        "подсолнечник": np.array([1,1.1,1,0.9])
    }.get(c, np.ones(4))


# =======================
# ГЕНЕРАЦИЯ И МЕТКИ
# =======================

def generate_cell(mu, rng):
    return mu + rng.normal(0, 0.6, 4)


def label_cell(x):
    sig = lambda z: 1 / (1 + np.exp(-z))
    t, m, r, l = sig(x[0]), sig(x[1]), sig(x[2]), sig(x[3])

    stress = (abs(m-0.55) + abs(t-0.55) + max(0,0.6-r)) / 3

    s = np.array([
        0.7*(1-m) + 0.3*stress,
        0.8*(1-l) + 0.2*stress,
        0.6*r + 0.4*l,
        0.7*r + 0.3*stress
    ])

    return (s > 0.5).astype(int)


# =======================
# НЕЙРОСЕТЬ ДЕЙСТВИЙ
# =======================

def build_action_net():
    m = Sequential([
        Dense(128, activation='relu', input_dim=4),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
    return m


@st.cache_resource
def train_action_net():
    df = pd.read_csv(CSV)
    regions = df["region_id"].unique()

    scalers = {r: StandardScaler().fit(df[df.region_id == r][FEATS])
               for r in regions}

    X, Y = [], []
    rng = np.random.default_rng(1)
    cultures = ["Пшеница","Рис","Картофель","Кукуруза","Подсолнечник"]

    for _ in range(10000):
        r = rng.choice(regions)
        c = rng.choice(cultures)
        sub = df[df.region_id == r]
        mu = sub[FEATS].mean().values

        x = scalers[r].transform([generate_cell(mu, rng)])[0]
        x *= culture_mult(c)

        X.append(x)
        Y.append(label_cell(x))

    X, Y = np.array(X), np.array(Y)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2)

    model = build_action_net()
    model.fit(Xtr, Ytr, epochs=8, batch_size=256, verbose=0)

    return df, regions, scalers, model


# =======================
# НЕЙРОСЕТЬ ТРАКТОРА (RNN)
# =======================

def build_tractor_rnn():
    m = Sequential([
        SimpleRNN(32, activation="tanh", input_shape=(None, 2)),
        Dense(4, activation="softmax")  # 0↑ 1↓ 2← 3→
    ])
    m.compile(optimizer=Adam(0.001), loss="sparse_categorical_crossentropy")
    return m


def generate_snake_path(H, W):
    path = []
    for r in range(H):
        cols = range(W) if r % 2 == 0 else range(W-1, -1, -1)
        for c in cols:
            path.append((r, c))
    return path


@st.cache_resource
def train_tractor_net():
    X, y = [], []
    rng = np.random.default_rng(2)

    for _ in range(2000):
        H, W = rng.integers(5, 20), rng.integers(5, 20)
        path = generate_snake_path(H, W)

        for i in range(len(path)-1):
            r, c = path[i]
            r2, c2 = path[i+1]

            X.append([[r/H, c/W]])
            if r2 > r: y.append(1)
            elif r2 < r: y.append(0)
            elif c2 > c: y.append(3)
            else: y.append(2)

    X = np.array(X)
    y = np.array(y)

    model = build_tractor_rnn()
    model.fit(X, y, epochs=5, batch_size=128, verbose=0)

    return model


# =======================
# STREAMLIT
# =======================

st.set_page_config(layout="wide")
st.title("Интеллектуальное управление сельхозтехникой")

df, regions, scalers, action_model = train_action_net()
tractor_model = train_tractor_net()

w = st.number_input("Ширина поля (ячейки)", 1, 500, 30)
h = st.number_input("Высота поля (ячейки)", 1, 500, 20)
reg = st.selectbox("Регион", regions)
cul = st.selectbox("Культура", ["Пшеница","Рис","Картофель","Кукуруза","Подсолнечник"])

if st.button("Сгенерировать карту и маршрут"):
    rng = np.random.default_rng(3)
    mu = df[df.region_id == reg][FEATS].mean().values
    sc = scalers[reg]

    raw = mu + rng.normal(0, 0.6, (h, w, 4))
    X = sc.transform(raw.reshape(-1, 4)) * culture_mult(cul)

    act = (action_model.predict(X, verbose=0) > 0.5).astype(int)
    act = act.reshape(h, w, 4)

    fig, axs = plt.subplots(2, 2, figsize=(8,8))
    for i in range(4):
        axs.flat[i].imshow(act[...,i])
        axs.flat[i].set_title(OPS_RU[i])
        axs.flat[i].axis("off")
    st.pyplot(fig)

    # маршрут
    path = generate_snake_path(h, w)
    rr = [p[0] for p in path]
    cc = [p[1] for p in path]

    fig2, ax = plt.subplots(figsize=(8,6))
    ax.imshow(act.any(axis=-1), cmap="Greens")
    ax.plot(cc, rr, linewidth=2, color="red")
    ax.scatter(cc[0], rr[0], c="blue", s=80)
    ax.scatter(cc[-1], rr[-1], c="black", s=80, marker="X")
    ax.set_title("Маршрут трактора")
    ax.axis("off")
    st.pyplot(fig2)
