import os
import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

from typing import Tuple, List, Optional
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


APP_TITLE = "Динамическое ценообразование: PD → ставка"
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "loan_applications.csv")


def _df_fingerprint(df: pd.DataFrame) -> str:
    sample = df.head(1000).to_csv(index=False).encode("utf-8", errors="ignore")
    return hashlib.md5(sample).hexdigest()


def clean_and_prepare_loans(df: pd.DataFrame) -> pd.DataFrame:
    loans = df[df["status"].isin(["reimbursed", "issued"])].copy()
    if "id" in loans.columns:
        loans = loans.drop_duplicates(subset=["id"])

    # Бизнес-правило: дефолт, если просрочка > 10 дней
    if "overdue_days" in loans.columns:
        loans["is_bad"] = loans["overdue_days"] > 10
    else:
        raise ValueError("В наборе данных отсутствует столбец 'overdue_days'.")

    # Удаляем старый канал
    if "source" in loans.columns:
        loans = loans[~loans["source"].isin(["online-old"])]

    # Удаляем нерелевантные столбцы, как в ноутбуке
    cols_to_drop = [
        "overdue_days",
        "status",
        "client_id",
        "id",
        "loan_given_duration",
        "loan_given_amount",
        "monthly_income_usd",
        "created_at",
        "updated_at",
        "source",
        "exported_at",
        "otp_is_verified",
        "otp_verified_at",
        "is_overdue",
        "partner_specific",
        "input",
        "deleted_at",
        "is_deleted",
        "threshold",
        "scoring_is_approved",
        "prediction",
        "contract_num",
        "remaining_principal_amount",
        "currency",
        "created_by_admin_id",
        "prediction-2",
        "credit_history_report_id",
        "partner_id",
        "is_client_bad_borrower",
    ]
    loans = loans.drop(columns=[c for c in cols_to_drop if c in loans.columns], errors="ignore")

    # Нормализация district, как в ноутбуке
    if "district" in loans.columns:
        loans["district"] = loans["district"].replace(["Hissar"], "Hissor")
        loans["district"] = loans["district"].replace(["Tursunzade"], "Tursunzoda")
        loans["district"] = loans["district"].replace(["Vahdat"], "Vakhdat")
        loans["district"] = loans["district"].replace(["Unknown", np.nan], "Dushanbe")

    return loans


def train_dynamic_model(loans: pd.DataFrame) -> Tuple[CatBoostClassifier, List[str], List[str], float]:
    # Удаляем сумму и срок клиента, чтобы модель PD не зависела от них
    dynamic = loans.copy()
    for c in ["client_loan_amount", "client_loan_duration"]:
        if c in dynamic.columns:
            dynamic = dynamic.drop(columns=[c])

    if "is_bad" not in dynamic.columns:
        raise ValueError("В данных отсутствует столбец 'is_bad' после подготовки.")

    X = dynamic.drop(columns=["is_bad"])  # признаки
    y = dynamic["is_bad"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    cat_features = [col for col in X.columns if X[col].dtype == "object"]

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        depth=10,
        custom_metric=["AUC"],
        cat_features=cat_features,
        use_best_model=True,
        verbose=False,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

    y_pred_test = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_pred_test))

    return model, cat_features, list(X.columns), auc


@st.cache_data(show_spinner=False)
def load_dataframe_from_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), low_memory=False)


@st.cache_data(show_spinner=False)
def load_default_dataframe(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    return None


@st.cache_resource(show_spinner=True)
def cached_load_model(model_path: str, meta_path: str) -> Tuple[CatBoostClassifier, List[str], List[str], Optional[float]]:
    model = CatBoostClassifier()
    model.load_model(model_path)
    auc = None
    feature_names: List[str] = []
    cat_features: List[str] = []
    if os.path.exists(meta_path):
        import json
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        feature_names = meta.get('feature_names', [])
        cat_features = meta.get('cat_features', [])
        auc = meta.get('auc')
    return model, cat_features, feature_names, auc


def compute_rate(pd_value: float, r_funds: float, lgd: float, rr: float, pricing_spread: float) -> float:
    if not (0 <= rr < 1):
        raise ValueError("RR должен быть в диапазоне [0, 1).")
    expected_loss = pd_value * lgd
    return float(r_funds + expected_loss / (1.0 - rr) + pricing_spread)


def ui_sidebar_data() -> pd.DataFrame:
    st.sidebar.header("Данные для обучения")
    df: Optional[pd.DataFrame] = None

    if os.path.exists(DEFAULT_DATA_PATH):
        st.sidebar.info("Найден локальный файл loan_applications.csv — будет использован по умолчанию.")
        df = load_default_dataframe(DEFAULT_DATA_PATH)

    uploaded = st.sidebar.file_uploader("Или загрузите CSV с заявками", type=["csv"])
    if uploaded is not None:
        content = uploaded.read()
        df = load_dataframe_from_bytes(content)

    if df is None:
        st.sidebar.warning("Нет данных: поместите loan_applications.csv рядом с приложением или загрузите CSV.")

    return df


def ui_sidebar_pricing() -> Tuple[float, float, float, float]:
    st.sidebar.header("Параметры ценообразования")
    r_funds = st.sidebar.number_input("Стоимость фондирования r_funds", value=0.08, min_value=0.0, max_value=1.0, step=0.005, format="%.3f")
    lgd = st.sidebar.number_input("LGD (доля потерь)", value=0.45, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
    rr = st.sidebar.number_input("RR (нормативный капитал)", value=0.10, min_value=0.0, max_value=0.99, step=0.01, format="%.2f")
    pricing_spread = st.sidebar.number_input("Маржа (pricing_spread)", value=0.03, min_value=0.0, max_value=1.0, step=0.005, format="%.3f")
    return r_funds, lgd, rr, pricing_spread


def ui_single_prediction(model: CatBoostClassifier, feature_names: List[str], cat_features: List[str], pricing_params: Tuple[float, float, float, float]):
    st.subheader("Одиночный расчёт")

    numeric_defaults = {
        "age": 30,
        "credit_history_count": 1,
        "dependents": 0,
    }

    # Базовые варианты для категориальных признаков, если их нет в обучающем датасете
    category_defaults = {
        "gender": ["Male", "Female"],
        "education": ["Higher Education", "Secondary Education", "Other Education"],
        "marital_status": ["Single", "Married"],
        "district": ["Dushanbe", "Hissor", "Tursunzoda", "Vakhdat"],
    }

    inputs = {}
    for name in feature_names:
        if name in ["client_loan_amount", "client_loan_duration"]:
            # Эти признаки исключены из модели PD
            continue

        if name in cat_features:
            # Предлагаем список значений; если обучающий набор известен — можно расширить
            options = category_defaults.get(name, None)
            if options is None:
                # Если неизвестны категории — оставляем текстовый ввод
                value = st.text_input(name, value="")
            else:
                value = st.selectbox(name, options)
            inputs[name] = value
        else:
            default_val = numeric_defaults.get(name, 0)
            value = st.number_input(name, value=float(default_val), step=1.0)
            inputs[name] = value

    if st.button("Рассчитать PD и ставку"):
        row = pd.DataFrame([inputs], columns=feature_names)
        # Убедимся, что категориальные признаки — строкового типа
        for c in cat_features:
            if c in row.columns:
                row[c] = row[c].astype(str)

        proba = float(model.predict_proba(row)[0, 1])
        r_funds, lgd, rr, pricing_spread = pricing_params
        rate = compute_rate(proba, r_funds, lgd, rr, pricing_spread)

        st.success(f"PD: {proba:.4f}")
        st.info(f"Ставка (годовая): {rate:.4f}")


def ui_batch_prediction(model: CatBoostClassifier, feature_names: List[str], cat_features: List[str], pricing_params: Tuple[float, float, float, float]):
    st.subheader("Пакетный расчёт")
    uploaded = st.file_uploader("Загрузите CSV для расчёта (со столбцами признаков)", type=["csv"], key="batch")
    if uploaded is None:
        st.caption("Ожидается CSV с теми же признаками, что и у обученной модели.")
        return

    df = load_dataframe_from_bytes(uploaded.read())
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error(f"В CSV отсутствуют столбцы: {missing}")
        return

    # Приводим типы категориальных признаков к строкам
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype(str)

    proba = model.predict_proba(df[feature_names])[:, 1]
    r_funds, lgd, rr, pricing_spread = pricing_params
    rates = np.array([compute_rate(p, r_funds, lgd, rr, pricing_spread) for p in proba])

    out = df.copy()
    out["PD"] = proba
    out["Rate"] = rates

    st.dataframe(out.head(50))
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Скачать результаты (CSV)", data=csv, file_name="rates_output.csv", mime="text/csv")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    df_raw = ui_sidebar_data()
    pricing_params = ui_sidebar_pricing()

    if df_raw is None:
        st.stop()

    with st.expander("Первые строки исходных данных", expanded=False):
        st.dataframe(df_raw.head())

    try:
        loans = clean_and_prepare_loans(df_raw)
    except Exception as e:
        st.error(f"Ошибка подготовки данных: {e}")
        st.stop()

    st.write(f"Размер выборки после подготовки: {loans.shape}")

    # Загрузка обученной модели из ноутбука
    model_path = os.path.join(os.path.dirname(__file__), "cb_dynamic_model.cbm")
    meta_path = os.path.join(os.path.dirname(__file__), "cb_dynamic_metadata.json")
    if not os.path.exists(model_path):
        st.error("Модель не найдена. Запустите последнюю ячейку в Attempt2.ipynb для сохранения модели.")
        st.stop()

    with st.spinner("Загрузка обученной CatBoost модели..."):
        model, cat_features, feature_names, auc = cached_load_model(model_path, meta_path)

    if auc is not None:
        st.success(f"Загружена модель. ROC AUC: {auc:.4f}")
    else:
        st.success("Загружена модель.")

    tab_single, tab_batch, tab_info = st.tabs(["Одиночный расчёт", "Пакетный расчёт", "О модели"])

    with tab_single:
        ui_single_prediction(model, feature_names, cat_features, pricing_params)

    with tab_batch:
        ui_batch_prediction(model, feature_names, cat_features, pricing_params)

    with tab_info:
        st.markdown("""
        - Модель: CatBoostClassifier (как в ноутбуке Attempt2.ipynb)
        - Признаки: все после очистки и удаления `client_loan_amount`, `client_loan_duration`
        - Формула ставки: `Rate = r_funds + (PD*LGD)/(1 - RR) + pricing_spread`
        """)


if __name__ == "__main__":
    main()


