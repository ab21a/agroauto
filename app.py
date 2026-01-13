import time
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
        "пшеница": np.array([1,1,1,1.0]),
        "рис": np.array([1,0.8,1.1,1.0]),
        "картофель": np.array([1,1,1,1.1]),
        "кукуруза": np.array([1.1,1,1,1.0]),
        "подсолнечник": np.array([1,1.1,1,0.9])
    }.get(c, np.array([1,1,1,1.0]))


def generate_cell(mu, rng):
    return mu + rng.normal(0, 0.6, 4)


def label_cell(x):
    sig = lambda z: 1/(1+np.exp(-z))
    t,m,r,l = sig(x[0]), sig(x[1]), sig(x[2]), sig(0.7*x[3])
    stress = (abs(m-0.55)+abs(t-0.55)+max(0,0.6-r))/3
    s = np.array([
        0.7*(1-m) + 0.2*(1-r) + 0.3*stress,
        0.85*(1-l) + 0.15*stress + 0.1*r,
        0.6*r + 0.4*l,
        0.7*r + 0.3*stress
    ]) + np.random.normal(0,0.02,4)
    return (s > np.array([0.55,0.48,0.55,0.55])).astype(int)


def build_action_net():
    m = Sequential([
        Dense(128, activation='relu', input_dim=4),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(4, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
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
    cultures = ["Пшеница","Рис","Картофель","Кукуруза","Подсолнечник"]

    X,Y = [],[]
    for _ in range(12000):
        r = rng.choice(regs)
        c = rng.choice(cultures)
        sub = df[df.region_id==r]
        mu = sub[FEATS].mean().values
        sc = scalers.get(r, scalers["__g__"])
        x = sc.transform([generate_cell(mu,rng)])[0] * culture_mult(c)
        X.append(x)
        Y.append(label_cell(x))

    X,Y = np.array(X), np.array(Y)
    Xtr,Xte,Ytr,Yte = train_test_split(X,Y,test_size=0.2,random_state=1)

    m = build_action_net()
    m.fit(Xtr,Ytr,epochs=10,batch_size=256,verbose=0)
    p = (m.predict(Xte,verbose=0)>0.5).astype(int)

    return df, regs, scalers, m, {
        "hamming": hamming_loss(Yte,p),
        "f1": f1_score(Yte,p,average="micro")
    }


def grid_size(w,h,spm):
    H0,W0=int(h*spm),int(w*spm)
    s=max(1,int(np.ceil(max(H0/MAX_SIDE,W0/MAX_SIDE))))
    return max(1,H0//s),max(1,W0//s),s/spm

DIRS4 = [(-1,0),(1,0),(0,-1),(0,1)]

def in_bounds_ext(r,c,H,W):
    return 0 <= r < H and (-1 <= c < W)

def manhattan_path_ext(a, b):
    (r1,c1),(r2,c2) = a,b
    path=[(r1,c1)]
    r,c=r1,c1
    while r != r2:
        r += 1 if r2>r else -1
        path.append((r,c))
    while c != c2:
        c += 1 if c2>c else -1
        path.append((r,c))
    return path


def smart_snake_route_for_mask(mask2d):
    H,W = mask2d.shape
    if not mask2d.any():
        return [(0,0)]

    segments = []
    for r in range(H):
        cols = np.where(mask2d[r])[0]
        if len(cols)==0:
            continue
        segments.append((r, int(cols.min()), int(cols.max())))
    segments.sort(key=lambda x: x[0])

    path = [(0,0)]
    cur = (0,0)

    for i,(r,cmin,cmax) in enumerate(segments):
        left_to_right = (i % 2 == 0)
        entry = (r, cmin) if left_to_right else (r, cmax)

        segmove = manhattan_path_ext(cur, entry)
        path.extend(segmove[1:])
        cur = entry

        if left_to_right:
            for c in range(cmin, cmax+1):
                if (r,c) != cur:
                    path.append((r,c))
                    cur = (r,c)
        else:
            for c in range(cmax, cmin-1, -1):
                if (r,c) != cur:
                    path.append((r,c))
                    cur = (r,c)

    return path


def add_refills_to_route(base_route, work_mask, capacity, refill_left_col=-1):
    H,W = work_mask.shape
    if capacity <= 0:
        capacity = 1

    if not work_mask.any():
        return base_route, 0, 0

    pending = work_mask.copy()
    fuel = capacity
    refills = 0
    full_route = [base_route[0]]
    cur = base_route[0]

    def go_to(point):
        nonlocal cur, full_route
        seg = manhattan_path_ext(cur, point)
        full_route.extend(seg[1:])
        cur = point

    total_work = int(pending.sum())
    done_count = 0

    for nxt in base_route[1:]:
        go_to(nxt)

        r,c = cur
        if 0 <= c < W and pending[r,c]:
            if fuel == 0:
                # заехать пополниться слева, вернуться
                go_to((r, refill_left_col))
                refills += 1
                fuel = capacity
                go_to((r, c))

            pending[r,c] = False
            fuel -= 1
            done_count += 1

            if done_count >= total_work:
                break

    return full_route, refills, done_count


def padded_mask_for_display(mask2d):
    H,W = mask2d.shape
    pad = np.zeros((H, W+1), dtype=mask2d.dtype)
    pad[:, 1:] = mask2d
    return pad


def shift_route_for_display(route):
    rr = [r for (r,c) in route]
    cc = [c+1 for (r,c) in route]
    return rr, cc

def draw_frame(ax, img, rr, cc, step, title=""):
    ax.clear()
    ax.imshow(img)
    ax.plot(cc[:step+1], rr[:step+1], linewidth=3)        # след
    ax.scatter([cc[0]], [rr[0]], s=120)                   # старт
    ax.scatter([cc[step]], [rr[step]], s=180, marker="s") # трактор
    ax.scatter([cc[-1]], [rr[-1]], s=120, marker="X")     # финиш
    ax.axis("off")
    if title:
        ax.set_title(title)

def animate_route_streamlit(img, rr, cc, speed_fps=20, stride=3, title=""):
    placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(9,9))
    delay = 1.0 / max(1, int(speed_fps))
    n = len(rr)

    step_list = list(range(0, n, max(1, int(stride))))
    if step_list[-1] != n-1:
        step_list.append(n-1)

    for step in step_list:
        draw_frame(ax, img, rr, cc, step, title=title)
        placeholder.pyplot(fig, clear_figure=False)
        time.sleep(delay)

st.set_page_config(layout="wide")
st.title("Автоматизация сельскохозяйственных операций")

df, regions, scalers, action_model, metrics = train_action_net()

with st.expander("Метрики модели операций"):
    st.write(metrics)

c1,c2,c3 = st.columns(3)
w = c1.number_input("Ширина поля (м)",1,1_000_000,200,10)
h = c2.number_input("Длина поля (м)",1,1_000_000,200,10)
spm = c3.number_input("Посевов в 1 метре",1,200,1)

c4,c5 = st.columns(2)
reg = c4.selectbox("Регион", regions)
cul = c5.selectbox("Культура", ["Пшеница","Рис","Картофель","Кукуруза","Подсолнечник"])

H, W, cell = grid_size(float(w), float(h), int(spm))

st.subheader("Запасы (ёмкости) для операций (1 ед. = обработка 1 клетки)")
cc1,cc2,cc3,cc4 = st.columns(4)
cap = [
    cc1.number_input(f"Ёмкость: {OPS_RU[0]} (клеток)", 1, 1_000_000, 300, 10),
    cc2.number_input(f"Ёмкость: {OPS_RU[1]} (клеток)", 1, 1_000_000, 220, 10),
    cc3.number_input(f"Ёмкость: {OPS_RU[2]} (клеток)", 1, 1_000_000, 180, 10),
    cc4.number_input(f"Ёмкость: {OPS_RU[3]} (клеток)", 1, 1_000_000, 260, 10),
]
st.caption("Считаем, что запасы находятся слева от поля (служебная полоса). Когда запас кончается, трактор едет на c=-1 и возвращается.")

st.subheader("Анимация")
a1,a2,a3 = st.columns(3)
do_anim = a1.checkbox("Анимировать движение трактора", value=True)
fps = a2.number_input("Скорость (fps)", 1, 60, 20, 1)
stride = a3.number_input("Шаг (ускорение, 1=каждая точка)", 1, 50, 3, 1)

if st.button("Сгенерировать"):
    rng = np.random.default_rng(2)
    sub = df[df.region_id==reg]
    mu = sub[FEATS].mean().values
    sc = scalers.get(reg, scalers["__g__"])

    raw = mu + rng.normal(0,0.6,(H,W,4))
    x = sc.transform(raw.reshape(-1,4)) * culture_mult(cul)
    act = (action_model.predict(x,verbose=0)>0.5).astype(int).reshape(H,W,4)

    mask_any = act.any(axis=-1)

    # Карты операций
    fig1,axs = plt.subplots(2,2,figsize=(9,9))
    for i in range(4):
        axs.flat[i].imshow(act[...,i])
        axs.flat[i].set_title(OPS_RU[i])
        axs.flat[i].axis("off")
    st.pyplot(fig1,clear_figure=True)

    st.subheader("Маршруты трактора по операциям (с выездами за пополнением)")
    tabs = st.tabs([OPS_RU[i] for i in range(4)])

    for op in range(4):
        with tabs[op]:
            op_mask = act[...,op].astype(bool)
            total_cells = int(op_mask.sum())

            if total_cells == 0:
                st.info("Нет клеток для этой операции.")
                continue

            # умная змейка
            base = smart_snake_route_for_mask(op_mask)

            # пополнения
            route, refills, done_count = add_refills_to_route(
                base_route=base,
                work_mask=op_mask,
                capacity=int(cap[op]),
                refill_left_col=-1
            )

            # картинка для отображения
            disp_mask = padded_mask_for_display(op_mask.astype(int))

            rr, cc = shift_route_for_display(route)

            if do_anim:
                animate_route_streamlit(
                    img=disp_mask,
                    rr=rr,
                    cc=cc,
                    speed_fps=int(fps),
                    stride=int(stride),
                    title=f"{OPS_RU[op]} (ёмкость={int(cap[op])}, пополнений={refills})"
                )
            else:
                fig, ax = plt.subplots(figsize=(9,9))
                ax.imshow(disp_mask)
                ax.plot(cc, rr, linewidth=3)
                ax.scatter([cc[0]],[rr[0]],s=120)
                ax.scatter([cc[-1]],[rr[-1]],s=120,marker="X")
                ax.axis("off")
                st.pyplot(fig, clear_figure=True)

            route_len_cells = max(0, len(route)-1)
            st.caption(
                f"Операция: {OPS_RU[op]} | "
                f"Клеток работы: {total_cells} | "
                f"Сделано: {done_count} | "
                f"Пополнений: {refills} | "
                f"Длина маршрута (клеток): {route_len_cells} | "
                f"Клетка ~ {cell:.2f} м | "
                f"Оценка пробега ~ {route_len_cells * cell:.1f} м"
            )

    st.subheader("Общая карта: где есть хотя бы одна операция")
    fig2, ax2 = plt.subplots(figsize=(9,9))
    ax2.imshow(mask_any)
    ax2.axis("off")
    st.pyplot(fig2, clear_figure=True)

    st.caption(f"Размер сетки: {H}×{W}, Клетка ~ {cell:.2f} м")
