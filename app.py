import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)

try:
    from catboost import CatBoostClassifier, Pool

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ---------- КОНСТАНТЫ ----------
MODEL_PATH = "model.cbm"
MEDIANS_PATH = "medians.json"

# Архив хранится на диске, поэтому переживает перезагрузку страницы
ARCHIVE_DIR = Path("archive")
ARCHIVE_INDEX = ARCHIVE_DIR / "index.json"
ARCHIVE_DATA = ARCHIVE_DIR / "datasets"
MAX_ARCHIVE = 50

MODE_MANUAL = "Ввести вручную"
MODE_CSV = "Загрузить CSV-файл"
NAN_MEDIAN = "Заполнить медианами из medians.json"
NAN_DROP = "Удалить строки с пропусками"

LOW_T, HIGH_T = 0.4, 0.6
GREEN, YELLOW, RED = "#2ecc71", "#f1c40f", "#e74c3c"

FEATURES = [
    "male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP",
    "diaBP", "BMI", "heartRate", "glucose"
]

FEATURE_LABELS = {
    "male": "Пол (0 - Женский, 1 - Мужской)",
    "age": "Возраст, лет",
    "education": "Образование (1-4)",
    "currentSmoker": "Курит в настоящее время (0/1)",
    "cigsPerDay": "Сигарет в день",
    "BPMeds": "Принимает лекарства от давления (0/1)",
    "prevalentStroke": "Был инсульт (0/1)",
    "prevalentHyp": "Гипертония (0/1)",
    "diabetes": "Диабет (0/1)",
    "totChol": "Общий холестерин, мг/дл",
    "sysBP": "Систолическое давление",
    "diaBP": "Диастолическое давление",
    "BMI": "Индекс массы тела (ИМТ)",
    "heartRate": "Пульс (уд/мин)",
    "glucose": "Уровень глюкозы, мг/дл"
}
SHORT_LABELS = {f: FEATURE_LABELS[f].split(" (")[0] for f in FEATURES}

FEATURE_DESCRIPTIONS = {
    "male": "Пол пациента. 0 — женщина, 1 — мужчина.",
    "age": "Возраст пациента в годах.",
    "education": "Уровень образования (1 — начальное, 2 — среднее, 3 — высшее, 4 — ученое).",
    "currentSmoker": "Курит ли пациент в настоящее время (0 — нет, 1 — да).",
    "cigsPerDay": "Среднее количество выкуриваемых сигарет в день.",
    "BPMeds": "Принимает ли пациент препараты от артериального давления (0 — нет, 1 — да).",
    "prevalentStroke": "Был ли у пациента инсульт в прошлом (0 — нет, 1 — да).",
    "prevalentHyp": "Есть ли у пациента гипертония (0 — нет, 1 — да).",
    "diabetes": "Есть ли у пациента диабет (0 — нет, 1 — да).",
    "totChol": "Общий уровень холестерина в крови (мг/дл).",
    "sysBP": "Систолическое артериальное давление (верхнее число).",
    "diaBP": "Диастолическое артериальное давление (нижнее число).",
    "BMI": "Индекс массы тела (вес в кг / рост в метрах в квадрате).",
    "heartRate": "Частота сердечных сокращений (ударов в минуту).",
    "glucose": "Уровень глюкозы в крови (мг/дл)."
}

TEST_EXAMPLE = {
    "male": 1, "age": 50, "education": 2, "currentSmoker": 1, "cigsPerDay": 10,
    "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 1, "diabetes": 0,
    "totChol": 240, "sysBP": 140, "diaBP": 90, "BMI": 28.5, "heartRate": 75, "glucose": 85
}

# Как рисовать поля формы
BINARY_FORMAT = {
    "male": {0: "Женский", 1: "Мужской"},
    "currentSmoker": {0: "Нет", 1: "Да"},
    "BPMeds": {0: "Нет", 1: "Да"},
    "prevalentStroke": {0: "Нет", 1: "Да"},
    "prevalentHyp": {0: "Нет", 1: "Да"},
    "diabetes": {0: "Нет", 1: "Да"},
}
EDU_LABELS = {1: "1 — начальное", 2: "2 — среднее", 3: "3 — высшее", 4: "4 — учёная степень"}
NUMERIC_SPECS = {  # (min, max, step)
    "age": (20, 100, 1), "cigsPerDay": (0, 60, 1), "totChol": (100, 600, 1),
    "sysBP": (80, 220, 1), "diaBP": (40, 140, 1), "BMI": (15.0, 50.0, 0.1),
    "heartRate": (40, 200, 1), "glucose": (40, 400, 1),
}

CSS = """
<style>
/* Подсветка архивных записей при наведении */
[class*="st-key-arch_"] {
    border-radius: 12px;
    transition: background .15s ease, border-color .15s ease,
                box-shadow .15s ease, transform .15s ease;
}
[class*="st-key-arch_"]:hover {
    background: rgba(255, 75, 75, .10);
    border-color: rgba(255, 75, 75, .75) !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, .18);
    transform: translateX(3px);
}
/* Лёгкий "подъём" кнопок в боковой панели */
[data-testid="stSidebar"] button { transition: transform .12s ease, box-shadow .12s ease; }
[data-testid="stSidebar"] button:hover { transform: translateY(-1px); box-shadow: 0 2px 8px rgba(0,0,0,.18); }
</style>
"""


# ---------- ЗАГРУЗКА ----------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not CATBOOST_AVAILABLE:
        raise RuntimeError("Библиотека catboost не установлена.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл модели '{path}' не найден.")
    model = CatBoostClassifier()
    model.load_model(path)
    return model


@st.cache_data(show_spinner=False)
def load_medians(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ---------- ЗНАЧЕНИЯ ФОРМЫ ----------
def coerce(feat, value):
    """Приводит значение к типу и диапазону соответствующего виджета."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(TEST_EXAMPLE[feat])
    if feat in NUMERIC_SPECS:
        lo, hi, _ = NUMERIC_SPECS[feat]
        v = min(max(v, lo), hi)
        return round(v, 1) if feat == "BMI" else int(round(v))
    if feat == "education":
        return int(min(max(round(v), 1), 4))
    return int(min(max(round(v), 0), 1))


def default_values():
    medians = load_medians(MEDIANS_PATH)
    return {f: coerce(f, medians.get(f, TEST_EXAMPLE[f])) for f in FEATURES}


def set_inputs(values, name=None):
    """Записывает значения прямо в ключи виджетов (иначе форма их не подхватит)."""
    for f in FEATURES:
        if f in values:
            st.session_state[f"in_{f}"] = coerce(f, values[f])
    if name is not None:
        st.session_state["in_name"] = name


def reset_form():
    set_inputs(default_values())
    st.toast("Форма сброшена к медианным значениям", icon="🧹")


def load_test_example():
    set_inputs(TEST_EXAMPLE)
    st.toast("Загружен тестовый пример", icon="🧪")


# ---------- АРХИВ (на диске) ----------
def archive_load():
    try:
        if ARCHIVE_INDEX.exists():
            data = json.loads(ARCHIVE_INDEX.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except (OSError, ValueError):
        pass
    return []


def archive_save():
    try:
        ARCHIVE_DATA.mkdir(parents=True, exist_ok=True)
        tmp = ARCHIVE_INDEX.with_suffix(".tmp")
        tmp.write_text(json.dumps(st.session_state.archive, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(ARCHIVE_INDEX)
    except OSError as exc:
        st.toast(f"Не удалось сохранить архив на диск: {exc}", icon="⚠️")


def _remove_files(rec):
    if rec.get("file"):
        try:
            (ARCHIVE_DATA / rec["file"]).unlink(missing_ok=True)
        except OSError:
            pass


def archive_add(record, raw_df=None):
    """Добавляет запись в начало архива. Повторный анализ тех же данных не плодит дубликаты."""
    records = st.session_state.archive
    for old in [r for r in records if r.get("fingerprint") == record["fingerprint"]]:
        _remove_files(old)
        records.remove(old)

    if raw_df is not None:
        try:
            ARCHIVE_DATA.mkdir(parents=True, exist_ok=True)
            fname = f"{record['id']}.csv"
            raw_df.to_csv(ARCHIVE_DATA / fname, index=False)
            record["file"] = fname
        except OSError as exc:
            st.toast(f"Не удалось сохранить набор данных: {exc}", icon="⚠️")

    records.insert(0, record)
    while len(records) > MAX_ARCHIVE:
        _remove_files(records.pop())
    archive_save()


def delete_record(rec_id):
    records = st.session_state.archive
    rec = next((r for r in records if r["id"] == rec_id), None)
    if rec:
        _remove_files(rec)
        records.remove(rec)
        archive_save()
        st.toast("Запись удалена из архива", icon="🗑️")


def clear_archive():
    for rec in st.session_state.archive:
        _remove_files(rec)
    st.session_state.archive = []
    archive_save()
    st.toast("Архив очищен", icon="🗑️")


def restore_record(rec_id):
    rec = next((r for r in st.session_state.archive if r["id"] == rec_id), None)
    if rec is None:
        return
    st.session_state.last_result = None

    if rec["type"] == "patient":
        set_inputs(rec["inputs"], rec["name"])
        st.session_state.radio_mode = MODE_MANUAL
        st.toast(f"Параметры пациента «{rec['name']}» восстановлены", icon="↩️")
        return

    try:
        raw = pd.read_csv(ARCHIVE_DATA / rec["file"])
    except (OSError, ValueError, KeyError):
        st.toast("Файл с этим набором данных не найден на диске", icon="⚠️")
        return
    st.session_state.csv_source = {"name": rec["name"], "df": raw, "from_archive": True}
    st.session_state.csv_sig = None
    st.session_state.uploader_nonce += 1  # сбрасываем виджет загрузки файла
    st.session_state.radio_mode = MODE_CSV
    st.session_state.check_model = bool(rec.get("check_model"))
    if rec.get("nan_strategy"):
        st.session_state.nan_strategy = rec["nan_strategy"]
    st.toast(f"Набор данных «{rec['name']}» восстановлен ({len(raw)} строк)", icon="↩️")


# ---------- ИНИЦИАЛИЗАЦИЯ ----------
def init_state():
    ss = st.session_state
    defaults = default_values()
    for f in FEATURES:
        ss.setdefault(f"in_{f}", defaults[f])
    ss.setdefault("in_name", "Пациент А")
    ss.setdefault("radio_mode", MODE_MANUAL)
    ss.setdefault("check_model", False)
    ss.setdefault("csv_source", None)
    ss.setdefault("csv_sig", None)
    ss.setdefault("uploader_nonce", 0)
    ss.setdefault("last_result", None)
    ss.setdefault("archive_filter", "Все")
    if "archive" not in ss:
        ss.archive = archive_load()


# ---------- РУЧНОЙ ВВОД ----------
def render_field(feat):
    key, label, hint = f"in_{feat}", SHORT_LABELS[feat], FEATURE_DESCRIPTIONS[feat]
    if feat == "education":
        st.selectbox(label, [1, 2, 3, 4], format_func=EDU_LABELS.get, key=key, help=hint)
    elif feat in BINARY_FORMAT:
        st.radio(label, [0, 1], format_func=BINARY_FORMAT[feat].get, horizontal=True, key=key, help=hint)
    else:
        lo, hi, step = NUMERIC_SPECS[feat]
        st.number_input(label, min_value=lo, max_value=hi, step=step, key=key, help=hint)


def manual_input_form():
    st.subheader("Клинические показатели пациента")
    st.text_input("Имя пациента (для архива)", key="in_name",
                  help="Под этим именем запись сохранится в архиве.")

    cols = st.columns(2)
    for col, feats in zip(cols, (FEATURES[:8], FEATURES[8:])):
        with col:
            for f in feats:
                render_field(f)

    vals = {f: st.session_state[f"in_{f}"] for f in FEATURES}

    # Мягкие проверки на правдоподобие
    if vals["diaBP"] >= vals["sysBP"]:
        st.warning("Диастолическое давление не должно быть выше или равно систолическому. Проверьте ввод.")
    if vals["currentSmoker"] == 0 and vals["cigsPerDay"] > 0:
        st.info("Указаны сигареты в день, но пациент отмечен как некурящий.")
    if vals["currentSmoker"] == 1 and vals["cigsPerDay"] == 0:
        st.info("Пациент отмечен как курящий, но число сигарет в день равно 0.")

    name = (st.session_state.in_name or "").strip() or "Пациент"
    return {
        "X": pd.DataFrame([vals], columns=FEATURES),
        "names": [name],
        "target": None,
        "check_model": False,
        "kind": "patient",
        "label": name,
    }


# ---------- CSV ----------
def prepare_csv(raw: pd.DataFrame, check_model: bool):
    """Готовит данные к анализу. Имена и таргет идут вместе со строками, поэтому не рассинхронизируются."""
    df = raw.copy()

    rows_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < rows_before:
        st.warning(f"Удалено дубликатов: {rows_before - len(df)}")

    name_col = next((c for c in df.columns if str(c).lower() in ("name", "имя")), None)
    names = df[name_col].astype(str) if name_col else pd.Series([f"Пациент {i + 1}" for i in range(len(df))])

    target = None
    if check_model:
        if "TenYearCHD" not in df.columns:
            st.error("Вы включили проверку модели, но в файле нет колонки 'TenYearCHD'.")
            return None
        target = pd.to_numeric(df["TenYearCHD"], errors="coerce")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"В файле не хватает колонок: {', '.join(missing)}")
        return None

    raw_feats = df[FEATURES]
    X = raw_feats.apply(pd.to_numeric, errors="coerce")
    coerced = int(X.isna().sum().sum() - raw_feats.isna().sum().sum())
    if coerced > 0:
        st.warning(f"Нечисловых значений превращено в пропуски: {coerced}")

    # Строки без таргета нельзя использовать для проверки
    if target is not None:
        bad = target.isna() | ~target.isin([0, 1])
        if bad.any():
            st.warning(f"Строк без корректного TenYearCHD (0/1) убрано: {int(bad.sum())}")
            keep = ~bad
            X, names, target = X[keep], names[keep], target[keep]

    nan_strategy = None
    nan_count = int(X.isna().sum().sum())
    if nan_count > 0:
        st.warning(f"Обнаружены пропуски: {nan_count} ячеек.")
        nan_strategy = st.radio("Что делать с пропусками (NaN)?", [NAN_MEDIAN, NAN_DROP], key="nan_strategy",
                                help="Медианы берутся из medians.json (посчитаны на обучающей выборке).")
        if nan_strategy == NAN_MEDIAN:
            X = X.fillna(load_medians(MEDIANS_PATH))
        else:
            keep = X.notna().all(axis=1)
            if (~keep).any():
                st.info(f"Удалено строк с пропусками: {int((~keep).sum())}")
            X, names = X[keep], names[keep]
            if target is not None:
                target = target[keep]

    X = X.reset_index(drop=True)
    names = names.reset_index(drop=True)
    if target is not None:
        target = target.reset_index(drop=True).astype(int)

    if X.empty:
        st.error("После очистки не осталось данных.")
        return None

    return {"X": X, "names": names.tolist(), "target": target, "nan_strategy": nan_strategy}


def csv_input_block():
    st.subheader("Загрузите CSV-файл")
    st.caption("Один пациент — одна строка. Колонки должны совпадать с признаками модели (15 признаков). "
               "Необязательная колонка `name`/`имя` попадёт в таблицу результатов.")

    template = pd.DataFrame([TEST_EXAMPLE], columns=FEATURES)
    st.download_button(
        "📥 Скачать шаблон CSV (без таргета)",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="framingham_template.csv", mime="text/csv",
        help="Пример файла с правильными названиями колонок."
    )

    check_model = st.checkbox(
        "Проверить работу модели (нужна колонка TenYearCHD)", key="check_model",
        help="Модель предскажет риски, сравнит их с реальными ответами и покажет метрики точности.")
    if check_model:
        st.info("⚠️ Для проверки метрик CSV должен содержать колонку `TenYearCHD` с ответами 0 или 1.")

    uploaded = st.file_uploader("CSV с клиническими данными", type=["csv"],
                                key=f"uploader_{st.session_state.uploader_nonce}")
    src = st.session_state.csv_source

    if uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        if st.session_state.csv_sig != sig:
            try:
                raw = pd.read_csv(uploaded)
            except Exception:
                st.error("Не удалось прочитать файл. Проверьте, что это корректный CSV.")
                return None
            st.session_state.csv_source = src = {"name": uploaded.name, "df": raw, "from_archive": False}
            st.session_state.csv_sig = sig
            st.session_state.last_result = None
    elif src and not src["from_archive"]:
        # файл убрали крестиком в загрузчике
        st.session_state.csv_source = src = None
        st.session_state.csv_sig = None

    if src is None:
        return None

    if src["from_archive"]:
        c1, c2 = st.columns([4, 1])
        c1.info(f"📂 Набор данных из архива: **{src['name']}** ({len(src['df'])} строк)")

        def _detach():
            st.session_state.csv_source = None

        c2.button("✖ Убрать", on_click=_detach, use_container_width=True,
                  help="Убрать набор данных из архива и загрузить другой файл.")

    prepared = prepare_csv(src["df"], check_model)
    if prepared is None:
        return None

    st.success(f"Готово к анализу строк: {len(prepared['X'])}")
    st.dataframe(prepared["X"], use_container_width=True, height=260)

    prepared.update({
        "check_model": check_model, "kind": "dataset", "label": src["name"], "raw": src["df"],
    })
    return prepared


# ---------- АНАЛИЗ ----------
def risk_category(probs):
    return np.select([probs < LOW_T, probs <= HIGH_T], ["Низкий", "Серая зона"], "Высокий")


def run_analysis(model, payload):
    X = payload["X"]
    probs = model.predict_proba(X)[:, 1]
    shap = None
    if len(X) == 1:
        try:
            sv = model.get_feature_importance(Pool(X), type="ShapValues")
            shap = np.asarray(sv)[0, :-1]
        except Exception:
            shap = None
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": payload["label"], "X": X, "names": payload["names"],
        "target": payload["target"], "check_model": payload["check_model"],
        "probs": probs, "shap": shap,
    }


def save_to_archive(result, payload):
    probs = result["probs"]
    base = {"id": uuid.uuid4().hex[:8], "time": result["time"]}

    if payload["kind"] == "patient":
        inputs = {f: float(result["X"].iloc[0][f]) for f in FEATURES}
        cat = risk_category(probs)[0]
        record = {
            **base, "type": "patient", "name": payload["label"], "inputs": inputs,
            "verdict": f"{probs[0] * 100:.0f}% · {cat}",
            "fingerprint": "p|" + json.dumps({"n": payload["label"], "i": inputs}, sort_keys=True),
        }
        archive_add(record)
    else:
        raw = payload["raw"]
        fp = int(pd.util.hash_pandas_object(raw, index=False).sum())
        high = float((probs > HIGH_T).mean())
        record = {
            **base, "type": "dataset", "name": payload["label"], "rows": int(len(raw)),
            "check_model": bool(payload["check_model"]), "nan_strategy": payload.get("nan_strategy"),
            "verdict": f"{len(probs)} пац. · ср. {probs.mean() * 100:.0f}% · высокий риск {high * 100:.0f}%",
            "fingerprint": f"d|{payload['label']}|{len(raw)}|{fp}",
        }
        archive_add(record, raw_df=raw)


# ---------- РЕЗУЛЬТАТЫ ----------
def make_gauge(p):
    color = GREEN if p < LOW_T else (YELLOW if p <= HIGH_T else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": " %", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, LOW_T * 100], "color": "rgba(46,204,113,.25)"},
                {"range": [LOW_T * 100, HIGH_T * 100], "color": "rgba(241,196,15,.25)"},
                {"range": [HIGH_T * 100, 100], "color": "rgba(231,76,60,.25)"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def render_single(result):
    p = float(result["probs"][0])
    X = result["X"]

    left, right = st.columns([1, 1.2])
    with left:
        if p < LOW_T:
            st.success("### ✅ Риск ИБС не обнаружен")
        elif p <= HIGH_T:
            st.warning("### ⚠️ Серая зона\nЕсть риски, но небольшие. Рекомендуется ежегодное посещение врача.")
        else:
            st.error("### 🚨 Повышенный риск ИБС в ближайшие 10 лет")
        st.plotly_chart(make_gauge(p), use_container_width=True)

    with right:
        st.markdown("**Что повлияло на этот прогноз**")
        if result["shap"] is None:
            st.info("Разбор факторов для этого пациента недоступен.")
        else:
            contrib = pd.DataFrame({
                "Признак": [f"{SHORT_LABELS[f]} = {X.iloc[0][f]:g}" for f in FEATURES],
                "Вклад": result["shap"],
            })
            top = contrib.reindex(contrib["Вклад"].abs().sort_values(ascending=False).index).head(8).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top["Вклад"], y=top["Признак"], orientation="h",
                marker_color=[RED if v > 0 else GREEN for v in top["Вклад"]],
            ))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Вклад в прогноз (SHAP)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟥 повышает риск · 🟩 снижает риск. Показаны 8 самых влиятельных показателей.")


def make_distribution(result):
    probs, target = result["probs"], result["target"]
    if target is not None and result["check_model"]:
        dist_df = pd.DataFrame({
            "Вероятность ИБС": probs,
            "Фактический класс": pd.Series(target).map({0: "Здоров (0)", 1: "ИБС (1)"}).values,
        })
        fig = px.histogram(dist_df, x="Вероятность ИБС", color="Фактический класс", barmode="overlay",
                           opacity=0.7, color_discrete_map={"Здоров (0)": GREEN, "ИБС (1)": RED})
    else:
        fig = px.histogram(pd.DataFrame({"Вероятность ИБС": probs}), x="Вероятность ИБС",
                           color_discrete_sequence=["#4c78a8"])
    fig.update_traces(xbins=dict(start=0, end=1, size=0.05))
    fig.add_vrect(x0=0, x1=LOW_T, fillcolor=GREEN, opacity=0.08, layer="below", line_width=0)
    fig.add_vrect(x0=LOW_T, x1=HIGH_T, fillcolor=YELLOW, opacity=0.10, layer="below", line_width=0)
    fig.add_vrect(x0=HIGH_T, x1=1, fillcolor=RED, opacity=0.08, layer="below", line_width=0)
    fig.add_vline(x=float(probs.mean()), line_dash="dash",
                  annotation_text=f"среднее {probs.mean():.2f}", annotation_position="top")
    fig.update_layout(title="Распределение вероятности ИБС", xaxis_range=[0, 1],
                      yaxis_title="Количество пациентов", bargap=0.05)
    return fig


def render_metrics(result, preds):
    y, probs = result["target"], result["probs"]
    st.info("Проверка полезна, чтобы понять, не устарела ли модель и нужно ли её переобучать.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y, preds):.3f}")
    if y.nunique() == 2:
        c2.metric("ROC-AUC", f"{roc_auc_score(y, probs):.3f}")
    else:
        c2.metric("ROC-AUC", "—", help="В файле только один класс, AUC не определён.")
    c3.metric("Recall (найдено больных)", f"{recall_score(y, preds, zero_division=0):.3f}")
    c4.metric("Precision", f"{precision_score(y, preds, zero_division=0):.3f}")

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y, preds, labels=[0, 1])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Прогноз: нет ИБС", "Прогноз: ИБС"], y=["Факт: нет ИБС", "Факт: ИБС"],
                        title="Матрица ошибок")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, probs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Модель"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Случайное угадывание",
                                     line=dict(dash="dash")))
            fig.update_layout(title="ROC-кривая", xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

    report = classification_report(y, preds, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)


def render_batch(result, preds):
    probs, X, names = result["probs"], result["X"], result["names"]
    cats = risk_category(probs)

    results = X.copy()
    results.insert(0, "Имя", names)
    results["Вероятность ИБС"] = probs.round(3)
    results["Категория"] = cats
    results["Предсказание"] = preds
    if result["target"] is not None and result["check_model"]:
        results["Факт (TenYearCHD)"] = result["target"].values
    results = results.sort_values("Вероятность ИБС", ascending=False).reset_index(drop=True)

    tabs = ["📊 Обзор", "📋 Таблица"]
    if result["target"] is not None and result["check_model"]:
        tabs.append("🎯 Метрики модели")
    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        k1, k2, k3, k4 = st.columns(4)
        for col, label, value in (
                (k1, "Пациентов", f"{len(probs)}"),
                (k2, "Средняя вероятность ИБС", f"{probs.mean() * 100:.1f} %"),
                (k3, "Высокий риск", f"{(cats == 'Высокий').sum()} ({(cats == 'Высокий').mean() * 100:.0f} %)"),
                (k4, "Серая зона", f"{(cats == 'Серая зона').sum()} ({(cats == 'Серая зона').mean() * 100:.0f} %)")):
            with col, st.container(border=True):
                st.metric(label, value)

        c1, c2 = st.columns([1.6, 1])
        with c1:
            st.plotly_chart(make_distribution(result), use_container_width=True)
            st.caption("Зелёная, жёлтая и красная зоны — пороги вердикта (до 0.4, 0.4–0.6, выше 0.6).")
        with c2:
            cat_df = pd.Series(cats).value_counts().reindex(["Низкий", "Серая зона", "Высокий"]).fillna(0)
            fig = px.pie(names=cat_df.index, values=cat_df.values, hole=0.55, title="Категории риска",
                         color=cat_df.index,
                         color_discrete_map={"Низкий": GREEN, "Серая зона": YELLOW, "Высокий": RED})
            st.plotly_chart(fig, use_container_width=True)

    with tab_objs[1]:
        st.dataframe(
            results, use_container_width=True,
            column_config={"Вероятность ИБС": st.column_config.ProgressColumn(
                "Вероятность ИБС", min_value=0.0, max_value=1.0, format="%.3f")},
        )
        st.download_button("⬇️ Скачать результаты (CSV)", data=results.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv",
                           help="Таблица со всеми показателями, вероятностью и категорией риска.")

    if len(tab_objs) > 2:
        with tab_objs[2]:
            render_metrics(result, preds)


@st.cache_data(show_spinner=False)
def get_feature_importance(_model):
    fi = _model.get_feature_importance()
    return pd.DataFrame({"Признак": FEATURES, "Важность": fi}).sort_values("Важность", ascending=False)


def render_feature_importance(model):
    with st.expander("🔎 Какие признаки важны модели в целом"):
        try:
            fi_df = get_feature_importance(model)
            fig = px.bar(fi_df.sort_values("Важность"), x="Важность", y="Признак", orientation="h")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"Важность признаков недоступна: {exc}")


def render_results(model, result):
    st.subheader("Результаты анализа")
    st.caption(f"{result['label']} · {result['time']}")

    preds = (result["probs"] >= 0.5).astype(int)
    if len(result["probs"]) == 1:
        render_single(result)
    else:
        render_batch(result, preds)
    render_feature_importance(model)


# ---------- БОКОВАЯ ПАНЕЛЬ ----------
def render_sidebar():
    with st.sidebar:
        st.header("Управление")
        st.button("🧹 Очистить форму", on_click=reset_form, use_container_width=True,
                  help="Сбрасывает все поля ввода к медианным значениям.")
        st.button("🧪 Тестовый пример", on_click=load_test_example, use_container_width=True,
                  help="Заполняет форму примером пациента с известными показателями.")

        st.divider()
        st.subheader("🗂️ Архив")
        archive = st.session_state.archive

        if not archive:
            st.caption("Архив пока пуст. Каждый запущенный анализ сохраняется сюда автоматически.")
        else:
            st.radio("Показать", ["Все", "Пациенты", "Наборы данных"], horizontal=True,
                     key="archive_filter", label_visibility="collapsed",
                     help="Фильтр записей архива по типу.")
            flt = st.session_state.archive_filter
            shown = [r for r in archive if flt == "Все"
                     or (flt == "Пациенты" and r["type"] == "patient")
                     or (flt == "Наборы данных" and r["type"] == "dataset")]

            for rec in shown:
                with st.container(border=True, key=f"arch_{rec['id']}"):
                    icon = "🧑‍⚕️" if rec["type"] == "patient" else "📊"
                    st.markdown(f"**{icon} {rec['name']}**")
                    st.caption(f"{rec['time']} · {rec['verdict']}")
                    c1, c2 = st.columns([3, 1])
                    hint = ("Вернуть в форму те же параметры пациента." if rec["type"] == "patient"
                            else "Снова загрузить этот набор данных с теми же настройками.")
                    c1.button("↩️ Вернуться", key=f"rest_{rec['id']}", on_click=restore_record,
                              args=(rec["id"],), use_container_width=True, help=hint)
                    c2.button("🗑️", key=f"del_{rec['id']}", on_click=delete_record, args=(rec["id"],),
                              use_container_width=True, help="Удалить только эту запись из архива.")

            with st.popover("🗑️ Очистить архив", use_container_width=True,
                            help="Удаляет все записи архива вместе с сохранёнными наборами данных."):
                st.warning("Все записи будут удалены без возможности восстановления.")
                st.button("Да, удалить всё", on_click=clear_archive, type="primary", use_container_width=True,
                          help="Подтвердить полную очистку архива.")

        st.caption("Архив хранится в папке `archive/` рядом с приложением.")

        with st.expander("📖 Справка по показателям"):
            for feat, desc in FEATURE_DESCRIPTIONS.items():
                st.markdown(f"**{FEATURE_LABELS[feat]}:** {desc}")


# ---------- ГЛАВНАЯ ФУНКЦИЯ ----------
def main():
    st.set_page_config(page_title="Прогноз ИБС (Framingham)", page_icon="🫀", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    st.title("🫀 Прогноз ИБС на 10 лет (Framingham)")
    st.caption("Учебное демо. Не является медицинским заключением.")

    model = None
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        st.warning(str(exc))

    mode = st.radio("Способ ввода данных:", [MODE_MANUAL, MODE_CSV], horizontal=True, key="radio_mode")
    payload = manual_input_form() if mode == MODE_MANUAL else csv_input_block()

    st.divider()

    if st.button("🔍 Запустить анализ", type="primary", use_container_width=True, key="btn_predict",
                 help="Считает вероятность ИБС и сохраняет данные в архив."):
        if model is None:
            st.error("Модель не загружена.")
        elif payload is None:
            st.error("Сначала введите данные или загрузите корректный CSV.")
        else:
            try:
                with st.spinner("Считаю..."):
                    result = run_analysis(model, payload)
                st.session_state.last_result = result
                save_to_archive(result, payload)
            except Exception as exc:
                st.error(f"Непредвиденная ошибка: {exc}")

    # Результат хранится в session_state, поэтому не пропадает при перерисовке
    # (клик по «Скачать», смена стратегии пропусков и т.д.)
    if model is not None and st.session_state.last_result is not None:
        render_results(model, st.session_state.last_result)

    # Сайдбар рисуем последним, чтобы только что добавленная запись сразу появилась в архиве
    render_sidebar()


if __name__ == "__main__":
    main()
