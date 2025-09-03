import os
import pandas as pd
import streamlit as st

from typing import Tuple, List, Optional
from catboost import CatBoostClassifier


APP_TITLE = "Динамическое ценообразование: PD → ставка"


# Убрали загрузку/предпросмотр данных и любое обучение в приложении


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


def ui_sidebar_pricing() -> Tuple[float, float, float, float]:
    st.sidebar.header("Параметры ценообразования")
    r_funds = st.sidebar.number_input("Стоимость фондирования r_funds", value=0.08, min_value=0.0, max_value=1.0, step=0.005, format="%.3f")
    lgd = st.sidebar.number_input("LGD (доля потерь)", value=0.45, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
    rr = st.sidebar.number_input("RR (нормативный капитал)", value=0.10, min_value=0.0, max_value=0.99, step=0.01, format="%.2f")
    pricing_spread = st.sidebar.number_input("Маржа (pricing_spread)", value=0.03, min_value=0.0, max_value=1.0, step=0.005, format="%.3f")
    return r_funds, lgd, rr, pricing_spread


def ui_single_prediction(model: CatBoostClassifier, feature_names: List[str], cat_features: List[str], pricing_params: Tuple[float, float, float, float], top_placeholder):
    st.subheader("Одиночный расчёт")

    numeric_defaults = {
        "age": 30,
        "credit_history_count": 1,
        "dependents": 0,
    }

    # Статические справочники для категориальных признаков [[memory:3748319]]
    category_defaults = {
        "gender": ["Male", "Female"],
        "education": [
            "Higher Education",
            "Secondary Education",
            "Incomplete Secondary Education",
            "Other Education",
            "Primary Education",
            "No Education",
        ],
        "marital_status": ["Single", "Married", "Divorced", "Widowed"],
        "district": [
            "Dushanbe",
            "Hissor",
            "Tursunzoda",
            "Vakhdat",
            "Yovon",
            "Panjakent",
            "Kanibadam",
            "Bobojon_Gafurov",
            "Khujand",
            "Istaravshan",
            "Isfara",
            "Kulob",
            "Qurghonteppa",
            "Rasht",
        ],
    }

    inputs = {}
    for name in feature_names:
        if name in ["client_loan_amount", "client_loan_duration"]:
            # Эти признаки исключены из модели PD
            continue

        if name in cat_features:
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
        # Большой акцент наверху: ставка в процентах
        top_placeholder.markdown(
            f"""
            <div style='padding: 18px; background: #F0F7FF; border: 1px solid #CDE3FF; border-radius: 12px; text-align: center; margin-bottom: 12px;'>
              <div style='font-size: 14px; color: #1D4ED8; font-weight: 600; letter-spacing: .3px;'>ПРОЦЕНТНАЯ СТАВКА</div>
              <div style='font-size: 44px; font-weight: 800; color: #0F172A; line-height: 1.1; margin-top: 6px;'>{rate*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    pricing_params = ui_sidebar_pricing()

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

    # Верхний акцентированный блок под заголовком для отображения процентной ставки
    top_placeholder = st.empty()
    top_placeholder.markdown(
        """
        <div style='padding: 18px; background: #F8FAFC; border: 1px dashed #E2E8F0; border-radius: 12px; text-align: center; margin-bottom: 12px;'>
          <div style='font-size: 14px; color: #64748B; font-weight: 600; letter-spacing: .3px;'>ПРОЦЕНТНАЯ СТАВКА</div>
          <div style='font-size: 32px; font-weight: 700; color: #334155; line-height: 1.1; margin-top: 6px;'>—</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_single, tab_info = st.tabs(["Одиночный расчёт", "О модели"])

    with tab_single:
        ui_single_prediction(model, feature_names, cat_features, pricing_params, top_placeholder)

    with tab_info:
        st.markdown("""
        - Модель: CatBoostClassifier (как в ноутбуке Attempt2.ipynb)
        - Признаки: все после очистки и удаления `client_loan_amount`, `client_loan_duration`
        - Формула ставки: `Rate = r_funds + (PD*LGD)/(1 - RR) + pricing_spread`
        """)


if __name__ == "__main__":
    main()


