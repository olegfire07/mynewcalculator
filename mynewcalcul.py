import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging
from scipy.optimize import bisect
import plotly.express as px
from io import BytesIO
from sklearn.linear_model import LinearRegression
import itertools
import numpy_financial as npf

# =============================================================================
# Настройка логирования
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Данные и Маппинги
# =============================================================================
storage_type_mapping = {
    'storage_share': 'Простое хранение',
    'loan_share': 'Хранение с займами',
    'vip_share': 'VIP-хранение',
    'short_term_share': 'Краткосрочное хранение'
}

# =============================================================================
# Декларация Датакласса для Параметров Склада
# =============================================================================
@dataclass
class WarehouseParams:
    """
    Класс для хранения параметров склада.
    """
    total_area: float  # Общая площадь склада (м²)
    rental_cost_per_m2: float  # Арендная плата за 1 м² (руб./мес.)
    useful_area_ratio: float  # Доля полезной площади (%)
    storage_share: float  # Доля площади под простое хранение
    loan_share: float  # Доля площади под хранение с займами
    vip_share: float  # Доля площади под VIP-хранение
    short_term_share: float  # Доля площади под краткосрочное хранение
    storage_fee: float  # Тариф простого хранения (руб./м²/мес.)
    shelves_per_m2: int  # Количество полок на 1 м²
    short_term_daily_rate: float  # Тариф краткосрочного хранения (руб./день/м²)
    item_evaluation: float  # Коэффициент оценки товара (доля)
    item_realization_markup: float  # Наценка при реализации (%) 
    average_item_value: float  # Средняя оценка товара (руб./м²)
    loan_interest_rate: float  # Ставка по займам в день (%)
    realization_share_storage: float  # Доля реализации из простого хранения
    realization_share_loan: float  # Доля реализации из хранения с займами
    realization_share_vip: float  # Доля реализации из VIP-хранения
    realization_share_short_term: float  # Доля реализации из краткосрочного хранения
    salary_expense: float  # Ежемесячные расходы на зарплату (руб.)
    miscellaneous_expenses: float  # Ежемесячные прочие расходы (руб.)
    depreciation_expense: float  # Ежемесячные расходы на амортизацию (руб.)
    marketing_expenses: float  # Ежемесячные расходы на маркетинг (руб.)
    insurance_expenses: float  # Ежемесячные страховые расходы (руб.)
    taxes: float  # Ежемесячные налоговые обязательства (руб.)
    time_horizon: int  # Горизонт прогноза (мес.)
    monthly_rent_growth: float  # Месячный рост аренды (%)
    default_probability: float  # Вероятность невозврата (%) 
    liquidity_factor: float  # Коэффициент ликвидности
    safety_factor: float  # Коэффициент запаса для расчета минимальной суммы займа
    storage_items_density: float  # Плотность товаров в простом хранении (вещей/м²)
    loan_items_density: float  # Плотность товаров в хранении с займами (вещей/м²)
    vip_items_density: float  # Плотность товаров в VIP-хранении (вещей/м²)
    short_term_items_density: float  # Плотность товаров в краткосрочном хранении (вещей/м²)
    one_time_setup_cost: float  # Единовременные расходы на настройку (руб.)
    one_time_equipment_cost: float  # Единовременные расходы на оборудование (руб.)
    one_time_other_costs: float  # Другие единовременные расходы (руб.)
    storage_area: Optional[float] = 0.0  # Выделенная площадь под простое хранение (м²)
    loan_area: Optional[float] = 0.0  # Выделенная площадь под хранение с займами (м²)
    vip_area: Optional[float] = 0.0  # Выделенная площадь под VIP-хранение (м²)
    short_term_area: Optional[float] = 0.0  # Выделенная площадь под краткосрочное хранение (м²)
    one_time_expenses: Optional[float] = 0.0  # Общие единовременные расходы (руб.)
    vip_extra_fee: Optional[float] = 0.0  # Дополнительная наценка VIP (руб./м²/мес.)

# =============================================================================
# Инициализация Состояния Приложения
# =============================================================================
if 'shares' not in st.session_state:
    # Инициализация долей хранения при первом запуске приложения
    st.session_state.shares = {
        'storage_share': 0.5,       # 50% под простое хранение
        'loan_share': 0.3,          # 30% под хранение с займами
        'vip_share': 0.1,           # 10% под VIP-хранение
        'short_term_share': 0.1     # 10% под краткосрочное хранение
    }

# =============================================================================
# Функции
# =============================================================================

def normalize_shares(share_key: str, new_value: float) -> None:
    """
    Нормализует доли хранения, чтобы сумма всех долей равнялась 1.0 (100%).

    :param share_key: Ключ доли, которую необходимо обновить.
    :param new_value: Новое значение доли (в долях, например, 0.3 для 30%).
    """
    total_shares = 1.0
    st.session_state.shares[share_key] = new_value
    remaining = total_shares - new_value
    other_keys = [k for k in st.session_state.shares.keys() if k != share_key and st.session_state.shares[k] > 0]

    if not other_keys:
        return

    current_other_sum = sum([st.session_state.shares[k] for k in other_keys])

    if current_other_sum == 0 and other_keys:
        # Если все остальные доли равны 0, распределяем оставшуюся долю поровну
        equal_share = remaining / len(other_keys)
        for k in other_keys:
            st.session_state.shares[k] = equal_share
    elif current_other_sum > 0 and other_keys:
        # Пропорционально распределяем оставшуюся долю между остальными долями
        for k in other_keys:
            st.session_state.shares[k] = (st.session_state.shares[k] / current_other_sum) * remaining

def validate_inputs(params: WarehouseParams) -> bool:
    """
    Проверяет корректность входных данных. Если данные некорректны, выводит ошибки.

    :param params: Объект с параметрами склада.
    :return: True, если все входные данные корректны, иначе False.
    """
    errors = []
    if params.total_area <= 0:
        errors.append("Общая площадь склада должна быть больше нуля.")
    if params.rental_cost_per_m2 <= 0:
        errors.append("Аренда за 1 м² должна быть больше нуля.")
    if params.loan_interest_rate < 0:
        errors.append("Процентная ставка по займам не может быть отрицательной.")
    if params.storage_fee < 0:
        errors.append("Тариф простого хранения не может быть отрицательным.")
    if not (0 <= params.useful_area_ratio <= 1):
        errors.append("Доля полезной площади должна быть между 0% и 100%.")
    for share_key, share_value in [
        ("storage_share", params.storage_share),
        ("loan_share", params.loan_share),
        ("vip_share", params.vip_share),
        ("short_term_share", params.short_term_share)
    ]:
        if not (0 <= share_value <= 1):
            errors.append(f"Доля {storage_type_mapping.get(share_key, share_key.replace('_', ' ').capitalize())} должна быть между 0 и 1.")
    if params.average_item_value < 0:
        errors.append("Средняя оценка товара не может быть отрицательной.")
    if params.salary_expense < 0:
        errors.append("Зарплата не может быть отрицательной.")
    if params.miscellaneous_expenses < 0:
        errors.append("Прочие расходы не могут быть отрицательными.")
    if params.depreciation_expense < 0:
        errors.append("Амортизация не может быть отрицательной.")
    if params.one_time_setup_cost < 0:
        errors.append("Расходы на настройку не могут быть отрицательными.")
    if params.one_time_equipment_cost < 0:
        errors.append("Расходы на оборудование не могут быть отрицательными.")
    if params.one_time_other_costs < 0:
        errors.append("Другие единовременные расходы не могут быть отрицательными.")
    if not (0 <= params.default_probability <= 1):
        errors.append("Вероятность невозврата должна быть между 0% и 100%.")
    if params.marketing_expenses < 0:
        errors.append("Маркетинговые расходы не могут быть отрицательными.")
    if params.insurance_expenses < 0:
        errors.append("Страховые расходы не могут быть отрицательными.")
    if params.taxes < 0:
        errors.append("Налоги не могут быть отрицательными.")

    # Проверка суммы долей хранения в ручном режиме
    total_shares = params.storage_share + params.loan_share + params.vip_share + params.short_term_share
    if not np.isclose(total_shares, 1.0):
        errors.append("Сумма долей хранения должна равняться 100%.")

    for error in errors:
        st.error(error)
    return len(errors) == 0

@st.cache_data(ttl=600)
def calculate_additional_metrics(total_income: float, total_expenses: float, profit: float) -> tuple:
    """
    Рассчитывает маржу прибыли и рентабельность.

    :param total_income: Общий доход (руб.).
    :param total_expenses: Общие расходы (руб.).
    :param profit: Прибыль (руб.).
    :return: Кортеж из маржи прибыли (%) и рентабельности (%).
    """
    profit_margin = (profit / total_income * 100) if total_income > 0 else 0
    profitability = (profit / total_expenses * 100) if total_expenses > 0 else 0
    return profit_margin, profitability

def generate_download_link(df: pd.DataFrame, filename: str = "results.csv") -> None:
    """
    Генерирует кнопку для скачивания результатов в формате CSV.

    :param df: DataFrame с результатами.
    :param filename: Имя файла для скачивания.
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Скачать результаты в CSV",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )

def generate_excel_download(df: pd.DataFrame, filename: str = "results.xlsx") -> None:
    """
    Генерирует кнопку для скачивания результатов в формате Excel.

    :param df: DataFrame с результатами.
    :param filename: Имя файла для скачивания.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Результаты')
    processed_data = output.getvalue()
    st.download_button(
        label="📥 Скачать результаты в Excel",
        data=processed_data,
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@st.cache_data(ttl=600)
def calculate_areas(total_area: float, useful_area_ratio: float, shelves_per_m2: int,
                    storage_share: float, loan_share: float, vip_share: float, short_term_share: float) -> dict:
    """
    Рассчитывает площади для разных типов хранения исходя из общих параметров.

    :param total_area: Общая площадь склада (м²).
    :param useful_area_ratio: Доля полезной площади.
    :param shelves_per_m2: Количество полок на 1 м².
    :param storage_share: Доля под простое хранение.
    :param loan_share: Доля под хранение с займами.
    :param vip_share: Доля под VIP-хранение.
    :param short_term_share: Доля под краткосрочное хранение.
    :return: Словарь с площадями для каждого типа хранения.
    """
    # Рассчитываем полезную площадь склада
    useful_area = total_area * useful_area_ratio
    # Умножаем на 2 для учета всех полок
    double_shelf_area = useful_area * 2 * shelves_per_m2
    # Рассчитываем площадь для каждого типа хранения
    storage_area = double_shelf_area * storage_share
    loan_area = double_shelf_area * loan_share
    vip_area = double_shelf_area * vip_share
    short_term_area = double_shelf_area * short_term_share
    return {
        "storage_area": storage_area,
        "loan_area": loan_area,
        "vip_area": vip_area,
        "short_term_area": short_term_area
    }

@st.cache_data(ttl=600)
def calculate_items(storage_area: float, loan_area: float, vip_area: float, short_term_area: float,
                    storage_items_density: float, loan_items_density: float,
                    vip_items_density: float, short_term_items_density: float) -> dict:
    """
    Рассчитывает количество вещей для каждого типа хранения.

    :param storage_area: Площадь под простое хранение (м²).
    :param loan_area: Площадь под хранение с займами (м²).
    :param vip_area: Площадь под VIP-хранение (м²).
    :param short_term_area: Площадь под краткосрочное хранение (м²).
    :param storage_items_density: Плотность товаров в простом хранении (вещей/м²).
    :param loan_items_density: Плотность товаров в хранении с займами (вещей/м²).
    :param vip_items_density: Плотность товаров в VIP-хранении (вещей/м²).
    :param short_term_items_density: Плотность товаров в краткосрочном хранении (вещей/м²).
    :return: Словарь с количеством вещей для каждого типа хранения.
    """
    stored_items = storage_area * storage_items_density
    total_items_loan = loan_area * loan_items_density
    vip_stored_items = vip_area * vip_items_density
    short_term_stored_items = short_term_area * short_term_items_density
    return {
        "stored_items": stored_items,
        "total_items_loan": total_items_loan,
        "vip_stored_items": vip_stored_items,
        "short_term_stored_items": short_term_stored_items
    }

@st.cache_data(ttl=600)
def calculate_financials(params: WarehouseParams) -> dict:
    """
    Рассчитывает основные финансовые показатели склада.

    :param params: Объект с параметрами склада.
    :return: Словарь с финансовыми показателями.
    """
    try:
        # Количество вещей
        stored_items = params.storage_area * params.storage_items_density
        total_items_loan = params.loan_area * params.loan_items_density
        vip_stored_items = params.vip_area * params.vip_items_density
        short_term_stored_items = params.short_term_area * params.short_term_items_density

        # Доходы от простого хранения
        storage_income = params.storage_area * params.storage_fee

        # Доходы от займов
        loan_interest_rate = max(params.loan_interest_rate, 0)  # Обеспечиваем неотрицательность ставки
        loan_amount = params.loan_area * params.average_item_value * params.item_evaluation
        loan_income_month = loan_amount * (loan_interest_rate / 100) * 30  # Доход от процентов за месяц

        # Реализация невостребованных товаров
        realization_items_storage = stored_items * params.realization_share_storage
        realization_items_loan = total_items_loan * params.realization_share_loan
        realization_items_vip = vip_stored_items * params.realization_share_vip
        realization_items_short_term = short_term_stored_items * params.realization_share_short_term

        realization_income_storage = realization_items_storage * params.average_item_value * (params.item_realization_markup / 100)
        realization_income_loan = realization_items_loan * params.average_item_value * (params.item_realization_markup / 100)
        realization_income_vip = realization_items_vip * params.average_item_value * (params.item_realization_markup / 100)
        realization_income_short_term = realization_items_short_term * params.average_item_value * (params.item_realization_markup / 100)

        realization_income = (realization_income_storage + realization_income_loan +
                              realization_income_vip + realization_income_short_term)

        # Применение вероятности дефолта к займам
        loan_income_after_realization = loan_income_month * (1 - params.realization_share_loan) * (1 - params.default_probability)

        # VIP доход
        vip_income = params.vip_area * (params.storage_fee + params.vip_extra_fee)

        # Краткосрочное хранение
        short_term_income = params.short_term_area * params.short_term_daily_rate * 30  # Доход за месяц

        # Маркетинговые доходы (предполагаем, что маркетинг увеличивает доходы)
        marketing_income = params.marketing_expenses * 1.5  # Пример коэффициента

        # Общий доход
        total_income = (storage_income + loan_income_after_realization +
                        realization_income + vip_income + short_term_income + marketing_income)

        # Расходы
        rental_expense = params.total_area * params.rental_cost_per_m2  # Ежемесячные расходы на аренду
        monthly_expenses = (
            rental_expense + 
            params.salary_expense + 
            params.miscellaneous_expenses + 
            params.depreciation_expense +
            params.marketing_expenses +    # Маркетинговые расходы
            params.insurance_expenses +    # Страховые расходы
            params.taxes                   # Налоги
        )
        # Общие расходы включают ежемесячные и единовременные расходы
        total_expenses = monthly_expenses + params.one_time_expenses

        # Прибыль
        profit = total_income - total_expenses
        daily_storage_fee = params.storage_fee / 30  # Расчет дневного тарифа

        return {
            "total_income": total_income,
            "total_expenses": total_expenses,
            "profit": profit,
            "realization_income": realization_income,
            "storage_income": storage_income,
            "loan_income_after_realization": loan_income_after_realization,
            "vip_income": vip_income,
            "short_term_income": short_term_income,
            "marketing_income": marketing_income,
            "rental_expense": rental_expense,
            "monthly_expenses": monthly_expenses,
            "one_time_expenses": params.one_time_expenses,
            "salary_expense": params.salary_expense,
            "miscellaneous_expenses": params.miscellaneous_expenses,
            "depreciation_expense": params.depreciation_expense,
            "marketing_expenses": params.marketing_expenses,
            "insurance_expenses": params.insurance_expenses,
            "taxes": params.taxes,
            "loan_interest_rate": loan_interest_rate,
            "daily_storage_fee": daily_storage_fee
        }
    except Exception as e:
        logger.error(f"Ошибка при расчёте финансовых показателей: {e}")
        # Возвращаем нулевые значения в случае ошибки
        return {
            "total_income": 0.0,
            "total_expenses": 0.0,
            "profit": 0.0,
            "realization_income": 0.0,
            "storage_income": 0.0,
            "loan_income_after_realization": 0.0,
            "vip_income": 0.0,
            "short_term_income": 0.0,
            "marketing_income": 0.0,
            "rental_expense": 0.0,
            "monthly_expenses": 0.0,
            "one_time_expenses": 0.0,
            "salary_expense": 0.0,
            "miscellaneous_expenses": 0.0,
            "depreciation_expense": 0.0,
            "marketing_expenses": 0.0,
            "insurance_expenses": 0.0,
            "taxes": 0.0,
            "loan_interest_rate": 0.0,
            "daily_storage_fee": 0.0
        }

def calculate_roi(total_income: float, total_expenses: float) -> float:
    """
    Рассчитывает Return on Investment (ROI).

    :param total_income: Общий доход (руб.).
    :param total_expenses: Общие расходы (руб.).
    :return: ROI (%).
    """
    if total_expenses == 0:
        return 0.0
    return (total_income - total_expenses) / total_expenses * 100

def calculate_irr(cash_flows: list, initial_investment: float) -> float:
    """
    Рассчитывает Internal Rate of Return (IRR).

    :param cash_flows: Список денежных потоков (руб.).
    :param initial_investment: Первоначальные инвестиции (руб.).
    :return: IRR (%).
    """
    try:
        irr = npf.irr([-initial_investment] + cash_flows) * 100
        return irr
    except Exception as e:
        logger.error(f"Ошибка при расчёте IRR: {e}")
        return 0.0

def display_metrics_card(metrics: dict, col):
    """
    Отображает метрики в виде карточек.

    :param metrics: Словарь с названиями метрик и их значениями.
    :param col: Колонка Streamlit для отображения.
    """
    for key, value in metrics.items():
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 15px;
                ">
                    <h4 style="margin:0;">{key}</h4>
                    <p style="font-size: 1.4em; margin:0;">{value}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

def perform_sensitivity_analysis(params: WarehouseParams, param_key: str, param_range: np.ndarray) -> pd.DataFrame:
    """
    Выполняет анализ чувствительности для одного параметра.

    :param params: Объект с параметрами склада.
    :param param_key: Ключ параметра, для которого выполняется анализ.
    :param param_range: Массив значений параметра для анализа.
    :return: DataFrame с результатами анализа.
    """
    results = []
    for val in param_range:
        updated_params = WarehouseParams(**vars(params))
        setattr(updated_params, param_key, val)
        financials = calculate_financials(updated_params)
        profit = financials.get("profit", 0)
        results.append({"Параметр": val, "Прибыль (руб.)": profit})
    return pd.DataFrame(results)

def train_ml_model(df: pd.DataFrame, target: str) -> LinearRegression:
    """
    Обучает модель машинного обучения на исторических данных.

    :param df: DataFrame с историческими данными.
    :param target: Название целевой переменной для прогнозирования.
    :return: Обученная модель LinearRegression.
    """
    model = LinearRegression()
    X = df[['Месяц']]
    y = df[target]
    model.fit(X, y)
    return model

def predict_with_model(model: LinearRegression, future_months: np.ndarray) -> list:
    """
    Делает прогноз с использованием обученной модели.

    :param model: Обученная модель LinearRegression.
    :param future_months: Массив будущих месяцев для прогнозирования.
    :return: Список прогнозируемых значений.
    """
    X_future = future_months.reshape(-1, 1)
    predictions = model.predict(X_future)
    return predictions.tolist()

def calculate_bep(param_key, base_value, financials_func, params: WarehouseParams) -> Optional[float]:
    """
    Рассчитывает точку безубыточности (BEP) для заданного параметра с использованием scipy.bisect.
    Учитывает единовременные расходы.

    :param param_key: Ключ параметра, для которого рассчитывается BEP.
    :param base_value: Базовое значение параметра.
    :param financials_func: Функция для расчета финансовых показателей.
    :param params: Объект с параметрами склада.
    :return: Значение параметра при котором прибыль равна 0, или None если не найдено.
    """
    def profit_at_param(value):
        # Создаем копию параметров и обновляем нужный параметр
        updated_params = WarehouseParams(**vars(params))
        setattr(updated_params, param_key, value)
        # Расчет финансов с обновленным параметром
        recalc_financials = financials_func(updated_params)
        # Прибыль уже включает единовременные расходы через `total_expenses`
        return recalc_financials.get("profit", 0)
    
    # Определение начального диапазона поиска
    lower = base_value * 0.5
    upper = base_value * 1.5

    try:
        profit_low = profit_at_param(lower)
        profit_high = profit_at_param(upper)
        if profit_low * profit_high > 0:
            logger.warning("Начальный диапазон не содержит точки безубыточности.")
            return None
        # Использование метода бисекции для поиска корня (прибыли = 0)
        bep = bisect(profit_at_param, lower, upper, xtol=0.01, maxiter=100)
        logger.info(f"BEP для {param_key}: {bep}")
        return bep
    except Exception as e:
        logger.error(f"Ошибка при расчёте BEP для {param_key}: {e}")
        st.error(f"Ошибка при расчёте BEP для {param_key}: {e}")
        return None

# =============================================================================
# Основная структура интерфейса
# =============================================================================
st.set_page_config(page_title="Экономическая модель склада 📦", layout="wide")
st.markdown("# Экономическая модель склада 📦")

st.markdown("""
**Инструкция:**
1. **Настройте параметры** в боковой панели слева.
2. В разделе **"📦 Параметры хранения"** выберите режим распределения долей: автоматический или ручной.
   - В **автоматическом режиме** используйте слайдеры для настройки долей, которые автоматически нормализуются до 100%.
   - В **ручном режиме** введите доли хранения вручную. Сумма долей должна равняться 100%.
3. В разделе **"📦 Параметры хранения"** также выберите или введите дневной тариф для краткосрочного хранения.
4. Месячный доход краткосрочного хранения = дневной тариф × площадь × 30.
5. Перейдите на вкладку **"📊 Общие результаты"**, чтобы **просмотреть итоговые метрики**.
6. Используйте вкладку **"🔍 Точка безубыточности"** для **анализа точки безубыточности** выбранного параметра.
7. В разделе **"📈 Прогнозирование"** наблюдайте за **динамикой финансовых показателей** на выбранный горизонт.
   - При необходимости включите **прогнозирование с использованием машинного обучения**.
8. **Скачивайте результаты** в формате CSV или Excel при необходимости.
9. **Сохраняйте и загружайте сценарии** для быстрого доступа к часто используемым настройкам.
    
**Точка безубыточности (BEP)** — это значение параметра (тариф, ставка или наценка), при котором прибыль становится 0. Выше этой точки — прибыль, ниже — убыток.
""")

# =============================================================================
# Боковая панель с вводом параметров
# =============================================================================
with st.sidebar:
    st.markdown("## Ввод параметров")
    
    # Параметры склада
    with st.sidebar.expander("🏢 Параметры склада", expanded=True):
        total_area = st.number_input(
            "📏 Общая площадь склада (м²)",
            value=250,
            step=10,
            min_value=1,
            help="Введите общую площадь вашего склада в квадратных метрах. Это значение должно быть больше нуля."
        )
        rental_cost_per_m2 = st.number_input(
            "💰 Аренда за 1 м² (руб./мес.)",
            value=1000,
            step=50,
            min_value=1,
            help="Ежемесячная арендная плата за один квадратный метр."
        )
        useful_area_ratio = st.slider(
            "📐 Доля полезной площади (%)",
            40,
            80,
            50,
            5,
            help="Процент полезной площади склада."
        ) / 100.0  # Преобразуем в доли (0-1)
    
    # Параметры хранения
    with st.sidebar.expander("📦 Параметры хранения"):
        # Переключатель между автоматическим и ручным режимами
        mode = st.radio(
            "🔄 Режим распределения долей хранения",
            options=["Автоматический", "Ручной"],
            index=0,
            help="Выберите режим распределения долей хранения."
        )

        # Инициализация 'short_term_daily_rate' с дефолтным значением
        short_term_daily_rate = 60.0  # Значение по умолчанию

        if mode == "Автоматический":
            # Автоматический режим: используем слайдеры с нормализацией
            storage_fee = st.number_input(
                "💳 Тариф простого хранения (руб./м²/мес.)",
                value=1500,
                step=100,
                min_value=0,
                help="Стоимость хранения одного квадратного метра в месяц."
            )
            shelves_per_m2 = st.number_input(
                "📚 Количество полок на 1 м²",
                value=3,
                step=1,
                min_value=1,
                max_value=100,
                help="Количество полок на один квадратный метр полезной площади."
            )

            # Опции для отключения определенных типов хранения
            no_storage_for_storage = st.checkbox(
                "🚫 Нет простого хранения",
                value=False,
                help="Отключить простой склад."
            )
            no_storage_for_loan = st.checkbox(
                "🚫 Нет хранения с займами",
                value=False,
                help="Отключить склад с займами."
            )
            no_storage_for_vip = st.checkbox(
                "🚫 Нет VIP-хранения",
                value=False,
                help="Отключить VIP-хранение."
            )
            no_storage_for_short_term = st.checkbox(
                "🚫 Нет краткосрочного хранения",
                value=False,
                help="Отключить краткосрочное хранение."
            )

            # Установка долей в 0 при отключении типа хранения
            if no_storage_for_storage:
                st.session_state.shares['storage_share'] = 0.0
            if no_storage_for_loan:
                st.session_state.shares['loan_share'] = 0.0
            if no_storage_for_vip:
                st.session_state.shares['vip_share'] = 0.0
            if no_storage_for_short_term:
                st.session_state.shares['short_term_share'] = 0.0

            st.markdown("### 📊 Распределение площади (%)")
            # Определяем, какие типы хранения активны
            storage_options = []
            if not no_storage_for_storage:
                storage_options.append("storage_share")
            if not no_storage_for_loan:
                storage_options.append("loan_share")
            if not no_storage_for_vip:
                storage_options.append("vip_share")
            if not no_storage_for_short_term:
                storage_options.append("short_term_share")

            total_storages = len(storage_options)
            if total_storages == 0:
                st.warning("🚫 Все типы хранения отключены.")
                remaining_share = 1.0
            else:
                # Создаем слайдеры для каждой активной доли хранения
                for share_key in storage_options:
                    storage_type = storage_type_mapping.get(share_key, share_key.replace('_', ' ').capitalize())
                    current_share = st.session_state.shares.get(share_key, 0.0) * 100  # В процентах
                    new_share = st.slider(
                        f"{storage_type} (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=current_share,
                        step=1.0,
                        key=share_key,
                        help=f"Доля площади, выделенная под {storage_type.lower()}."
                    )
                    # Нормализуем доли хранения
                    normalize_shares(share_key, new_share / 100.0)
                    # Расчет выделенной площади для отображения
                    allocated_area = total_area * useful_area_ratio * 2 * shelves_per_m2 * st.session_state.shares[share_key]
                    st.markdown(f"**{storage_type}:** {st.session_state.shares[share_key] * 100:.1f}% ({allocated_area:.2f} м²)")

                # Вычисление оставшейся доли площади
                remaining_share = 1.0 - sum(st.session_state.shares.values())
                # Убедимся, что remaining_share не отрицателен
                remaining_share = max(min(remaining_share, 1.0), 0.0)

            # Вычисление оставшейся площади в м²
            remaining_area = total_area * useful_area_ratio * 2 * shelves_per_m2 * remaining_share

            # Отображение прогресс-бара с текстовой меткой
            progress_bar = st.progress(remaining_share)
            st.markdown(f"**Оставшаяся площадь:** {remaining_share * 100:.2f}% ({remaining_area:.2f} м²)")

            # Тариф для краткосрочного хранения
            st.markdown("### 🕒 Тариф для краткосрочного хранения (руб./день/м²)")
            short_term_rate_choice = st.selectbox(
                "Выберите дневной тариф краткосрочного хранения",
                ["50 руб./день/м²", "60 руб./день/м²", "100 руб./день/м²", "Другое (ввести вручную)"],
                help="Выберите один из предустановленных тарифов или введите свой."
            )
            if short_term_rate_choice == "Другое (ввести вручную)":
                short_term_daily_rate = st.number_input(
                    "Введите дневной тариф (руб./день/м²)",
                    value=60.0,
                    step=5.0,
                    min_value=0.0,
                    help="Вручную введите дневной тариф для краткосрочного хранения."
                )
            else:
                short_term_daily_rate = float(short_term_rate_choice.split()[0])

        else:
            # Ручной режим: пользователь вводит доли вручную
            st.markdown("### 🖊️ Ручной ввод долей хранения")
            storage_fee = st.number_input(
                "💳 Тариф простого хранения (руб./м²/мес.)",
                value=1500,
                step=100,
                min_value=0,
                help="Стоимость хранения одного квадратного метра в месяц."
            )
            shelves_per_m2 = st.number_input(
                "📚 Количество полок на 1 м²",
                value=3,
                step=1,
                min_value=1,
                max_value=100,
                help="Количество полок на один квадратный метр полезной площади."
            )

            # Опции для отключения определенных типов хранения
            no_storage_for_storage = st.checkbox(
                "🚫 Нет простого хранения",
                value=False,
                help="Отключить простой склад."
            )
            no_storage_for_loan = st.checkbox(
                "🚫 Нет хранения с займами",
                value=False,
                help="Отключить склад с займами."
            )
            no_storage_for_vip = st.checkbox(
                "🚫 Нет VIP-хранения",
                value=False,
                help="Отключить VIP-хранение."
            )
            no_storage_for_short_term = st.checkbox(
                "🚫 Нет краткосрочного хранения",
                value=False,
                help="Отключить краткосрочное хранение."
            )

            # Установка долей в 0 при отключении типа хранения
            if no_storage_for_storage:
                st.session_state.shares['storage_share'] = 0.0
            if no_storage_for_loan:
                st.session_state.shares['loan_share'] = 0.0
            if no_storage_for_vip:
                st.session_state.shares['vip_share'] = 0.0
            if no_storage_for_short_term:
                st.session_state.shares['short_term_share'] = 0.0

            st.markdown("### 📊 Распределение площади (%)")
            # Определяем, какие типы хранения активны
            storage_options = []
            if not no_storage_for_storage:
                storage_options.append("storage_share")
            if not no_storage_for_loan:
                storage_options.append("loan_share")
            if not no_storage_for_vip:
                storage_options.append("vip_share")
            if not no_storage_for_short_term:
                storage_options.append("short_term_share")

            if len(storage_options) == 0:
                st.warning("🚫 Все типы хранения отключены.")
            else:
                # Ручной ввод долей хранения
                manual_shares = {}
                for share_key in storage_options:
                    storage_type = storage_type_mapping.get(share_key, share_key.replace('_', ' ').capitalize())
                    current_share = st.session_state.shares.get(share_key, 0.0) * 100  # В процентах
                    manual_share = st.number_input(
                        f"{storage_type} (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=current_share,
                        step=1.0,
                        key=f"manual_{share_key}",
                        help=f"Введите долю площади для {storage_type.lower()}."
                    )
                    manual_shares[share_key] = manual_share / 100.0  # Преобразуем в доли (0-1)

                # Обновляем доли хранения
                for share_key, share_value in manual_shares.items():
                    st.session_state.shares[share_key] = share_value

                # Расчет выделенной площади для отображения
                for share_key in storage_options:
                    storage_type = storage_type_mapping.get(share_key, share_key.replace('_', ' ').capitalize())
                    allocated_area = total_area * useful_area_ratio * 2 * shelves_per_m2 * st.session_state.shares[share_key]
                    st.markdown(f"**{storage_type}:** {st.session_state.shares[share_key] * 100:.1f}% ({allocated_area:.2f} м²)")

                # Вычисление оставшейся доли площади
                remaining_share = 1.0 - sum(st.session_state.shares.values())
                # Вычисление оставшейся площади в м²
                remaining_area = total_area * useful_area_ratio * 2 * shelves_per_m2 * remaining_share

                # Отображение прогресс-бара с текстовой меткой
                progress_bar = st.progress(remaining_share)
                st.markdown(f"**Оставшаяся площадь:** {remaining_share * 100:.2f}% ({remaining_area:.2f} м²)")

            # Тариф для краткосрочного хранения
            st.markdown("### 🕒 Тариф для краткосрочного хранения (руб./день/м²)")
            short_term_rate_choice = st.selectbox(
                "Выберите дневной тариф краткосрочного хранения",
                ["50 руб./день/м²", "60 руб./день/м²", "100 руб./день/м²", "Другое (ввести вручную)"],
                help="Выберите один из предустановленных тарифов или введите свой."
            )
            if short_term_rate_choice == "Другое (ввести вручную)":
                short_term_daily_rate = st.number_input(
                    "Введите дневной тариф (руб./день/м²)",
                    value=60.0,
                    step=5.0,
                    min_value=0.0,
                    help="Вручную введите дневной тариф для краткосрочного хранения."
                )
            else:
                short_term_daily_rate = float(short_term_rate_choice.split()[0])

    # Параметры vip_extra_fee
    with st.sidebar.expander("👑 Дополнительные параметры VIP-хранения"):
        vip_extra_fee = st.number_input(
            "👑 Дополнительная наценка VIP (руб./м²/мес.)",
            value=0.0,
            step=50.0,
            min_value=0.0,
            help="Дополнительная наценка за VIP-хранение."
        )
        # Сохраняем vip_extra_fee в session_state.params, если params уже существует
        if 'params' in st.session_state and hasattr(st.session_state.params, 'vip_extra_fee'):
            st.session_state.params.vip_extra_fee = vip_extra_fee

    # Параметры оценок и займов
    with st.sidebar.expander("🔍 Параметры оценок и займов"):
        item_evaluation = st.slider(
            "🔍 Оценка вещи (%)",
            0,
            100,
            80,
            5,
            help="Процент оценки вещи."
        ) / 100.0  # Преобразуем в доли (0-1)
        item_realization_markup = st.number_input(
            "📈 Наценка реализации (%)",
            value=20.0,
            step=5.0,
            min_value=0.0,
            max_value=100.0,
            help="Процент наценки при реализации товаров."
        )
        average_item_value = st.number_input(
            "💲 Средняя оценка (руб./м²)",
            value=10000,
            step=500,
            min_value=0,
            help="Средняя оценка товара в рублях за квадратный метр."
        )
        loan_interest_rate = st.number_input(
            "💳 Ставка займов в день (%)",
            value=0.317,
            step=0.01,
            min_value=0.0,
            help="Процентная ставка по займам в день."
        )

    # Параметры плотности
    with st.sidebar.expander("📦 Параметры плотности"):
        storage_items_density = st.number_input(
            "📦 Простое (вещей/м²)",
            value=5,
            step=1,
            min_value=1,
            max_value=100,
            help="Количество вещей на один квадратный метр простого склада."
        )
        loan_items_density = st.number_input(
            "💳 Займы (вещей/м²)",
            value=5,
            step=1,
            min_value=1,
            max_value=100,
            help="Количество вещей на один квадратный метр склада с займами."
        )
        vip_items_density = st.number_input(
            "👑 VIP (вещей/м²)",
            value=2,
            step=1,
            min_value=1,
            max_value=100,
            help="Количество вещей на один квадратный метр VIP-хранения."
        )
        short_term_items_density = st.number_input(
            "⏳ Краткосрочное (вещей/м²)",
            value=4,
            step=1,
            min_value=1,
            max_value=100,
            help="Количество вещей на один квадратный метр краткосрочного хранения."
        )

    # Параметры реализации
    with st.sidebar.expander("📈 Параметры реализации"):
        realization_share_storage = st.slider(
            "📦 Простое (%)",
            0,
            100,
            50,
            5,
            help="Процент товаров для реализации из простого хранения."
        ) / 100.0  # Преобразуем в доли (0-1)
        realization_share_loan = st.slider(
            "💳 Займы (%)",
            0,
            100,
            50,
            5,
            help="Процент товаров для реализации из хранения с займами."
        ) / 100.0
        realization_share_vip = st.slider(
            "👑 VIP (%)",
            0,
            100,
            50,
            5,
            help="Процент товаров для реализации из VIP-хранения."
        ) / 100.0
        realization_share_short_term = st.slider(
            "⏳ Краткосрочное (%)",
            0,
            100,
            50,
            5,
            help="Процент товаров для реализации из краткосрочного хранения."
        ) / 100.0

    # Финансовые параметры
    with st.sidebar.expander("💼 Финансовые параметры"):
        # Ежемесячные расходы
        st.markdown("### 🗓️ Ежемесячные расходы")
        salary_expense = st.number_input(
            "💼 Зарплата (руб./мес.)",
            value=240000,
            step=10000,
            min_value=0,
            help="Ежемесячные расходы на зарплату."
        )
        miscellaneous_expenses = st.number_input(
            "🧾 Прочие расходы (руб./мес.)",
            value=50000,
            step=5000,
            min_value=0,
            help="Ежемесячные прочие расходы."
        )
        depreciation_expense = st.number_input(
            "📉 Амортизация (руб./мес.)",
            value=20000,
            step=5000,
            min_value=0,
            help="Ежемесячные расходы на амортизацию."
        )
        marketing_expenses = st.number_input(
            "📢 Маркетинговые расходы (руб./мес.)",
            value=30000,
            step=5000,
            min_value=0,
            help="Ежемесячные расходы на маркетинг."
        )
        insurance_expenses = st.number_input(
            "🛡️ Страховые расходы (руб./мес.)",
            value=10000,
            step=1000,
            min_value=0,
            help="Ежемесячные страховые расходы."
        )
        taxes = st.number_input(
            "💰 Налоги (руб./мес.)",
            value=50000,
            step=5000,
            min_value=0,
            help="Ежемесячные налоговые обязательства."
        )

        # Единовременные расходы
        st.markdown("### 💸 Единовременные расходы")
        one_time_setup_cost = st.number_input(
            "🔧 Расходы на настройку (руб.)",
            value=100000,
            step=5000,
            min_value=0,
            help="Единовременные расходы на настройку склада."
        )
        one_time_equipment_cost = st.number_input(
            "🛠️ Расходы на оборудование (руб.)",
            value=200000,
            step=5000,
            min_value=0,
            help="Единовременные расходы на оборудование склада."
        )
        one_time_other_costs = st.number_input(
            "📦 Другие единовременные расходы (руб.)",
            value=50000,
            step=5000,
            min_value=0,
            help="Другие единовременные расходы."
        )

    # Расширенные параметры
    with st.sidebar.expander("📈 Расширенные параметры (Временная динамика и риск)", expanded=True):
        disable_extended = st.checkbox(
            "🚫 Отключить расширенные параметры",
            value=False,
            help="Отключить дополнительные параметры прогноза и риска."
        )

        if not disable_extended:
            time_horizon = st.slider(
                "🕒 Горизонт прогноза (мес.)",
                1,
                24,
                6,
                help="Количество месяцев для прогноза финансовых показателей."
            )
            monthly_rent_growth = st.number_input(
                "📈 Месячный рост аренды (%)",
                value=1.0,
                step=0.5,
                min_value=0.0,
                help="Процентный рост аренды в месяц."
            ) / 100.0  # Преобразуем в доли (0-1)
            default_probability = st.number_input(
                "❌ Вероятность невозврата (%)",
                value=5.0,
                step=1.0,
                min_value=0.0,
                max_value=100.0,
                help="Процентная вероятность невозврата займов."
            ) / 100.0  # Преобразуем в доли (0-1)
            liquidity_factor = st.number_input(
                "💧 Ликвидность",
                value=1.0,
                step=0.1,
                min_value=0.1,
                help="Коэффициент ликвидности."
            )
            safety_factor = st.number_input(
                "🛡️ Коэффициент запаса",
                value=1.2,
                step=0.1,
                min_value=0.1,
                help="Коэффициент запаса для расчёта минимальной суммы займа."
            )
        else:
            # Устанавливаем базовые значения, если расширенные параметры отключены
            time_horizon = 1
            monthly_rent_growth = 0.0
            default_probability = 0.0
            liquidity_factor = 1.0
            safety_factor = 1.0

    # =============================================================================
    # Обработка данных и расчёты
    # =============================================================================

    # Создание экземпляра параметров
    params = WarehouseParams(
        total_area=total_area,
        rental_cost_per_m2=rental_cost_per_m2,
        useful_area_ratio=useful_area_ratio,
        storage_share=st.session_state.shares.get('storage_share', 0.0),
        loan_share=st.session_state.shares.get('loan_share', 0.0),
        vip_share=st.session_state.shares.get('vip_share', 0.0),
        short_term_share=st.session_state.shares.get('short_term_share', 0.0),
        storage_fee=storage_fee if mode == "Автоматический" else st.session_state.shares.get('storage_share', 0.0),
        shelves_per_m2=shelves_per_m2,
        short_term_daily_rate=short_term_daily_rate,  # Переменная теперь определена всегда
        item_evaluation=item_evaluation,
        item_realization_markup=item_realization_markup,
        average_item_value=average_item_value,
        loan_interest_rate=loan_interest_rate,
        realization_share_storage=realization_share_storage,
        realization_share_loan=realization_share_loan,
        realization_share_vip=realization_share_vip,
        realization_share_short_term=realization_share_short_term,
        salary_expense=salary_expense,
        miscellaneous_expenses=miscellaneous_expenses,
        depreciation_expense=depreciation_expense,
        marketing_expenses=marketing_expenses,  # Новое поле
        insurance_expenses=insurance_expenses,    # Новое поле
        taxes=taxes,                              # Новое поле
        time_horizon=time_horizon,
        monthly_rent_growth=monthly_rent_growth,
        default_probability=default_probability,
        liquidity_factor=liquidity_factor,
        safety_factor=safety_factor,
        storage_items_density=storage_items_density,
        loan_items_density=loan_items_density,
        vip_items_density=vip_items_density,
        short_term_items_density=short_term_items_density,
        one_time_setup_cost=one_time_setup_cost,
        one_time_equipment_cost=one_time_equipment_cost,
        one_time_other_costs=one_time_other_costs,
        vip_extra_fee=vip_extra_fee  # Используем значение из виджета
    )

    # Расчет общей суммы единовременных расходов
    params.one_time_expenses = (
        params.one_time_setup_cost +
        params.one_time_equipment_cost +
        params.one_time_other_costs
    )

    # Расчет площадей и добавление их в params
    areas = calculate_areas(
        total_area=params.total_area,
        useful_area_ratio=params.useful_area_ratio,
        shelves_per_m2=params.shelves_per_m2,
        storage_share=params.storage_share,
        loan_share=params.loan_share,
        vip_share=params.vip_share,
        short_term_share=params.short_term_share
    )
    params = WarehouseParams(**{**vars(params), **areas})  # Обновляем params с добавленными площадями

    # Сохранение params в session_state
    st.session_state.params = params

    # Валидация входных данных
    inputs_valid = validate_inputs(params)

# =============================================================================
# Расчеты и отображение результатов в основной части приложения
# =============================================================================
if 'params' in st.session_state:
    if inputs_valid:
        with st.spinner("Выполняются расчёты..."):
            # Расчет количества вещей
            items = calculate_items(
                storage_area=st.session_state.params.storage_area,
                loan_area=st.session_state.params.loan_area,
                vip_area=st.session_state.params.vip_area,
                short_term_area=st.session_state.params.short_term_area,
                storage_items_density=st.session_state.params.storage_items_density,
                loan_items_density=st.session_state.params.loan_items_density,
                vip_items_density=st.session_state.params.vip_items_density,
                short_term_items_density=st.session_state.params.short_term_items_density
            )
            # Расчет финансовых показателей
            base_financials = calculate_financials(st.session_state.params)

        # Расчет дополнительных метрик
        profit_margin, profitability = calculate_additional_metrics(
            total_income=base_financials["total_income"],
            total_expenses=base_financials["total_expenses"],
            profit=base_financials["profit"]
        )

        # Расчет ROI и IRR
        roi = calculate_roi(base_financials["total_income"], base_financials["total_expenses"])
        cash_flows = [base_financials["profit"]] * st.session_state.params.time_horizon
        initial_investment = st.session_state.params.one_time_expenses
        irr = calculate_irr(cash_flows, initial_investment)

        # Расчет минимальной суммы займа
        if not disable_extended and st.session_state.params.loan_interest_rate > 0:
            average_growth_factor = 1 + st.session_state.params.monthly_rent_growth * (st.session_state.params.time_horizon / 2)
            adjusted_daily_storage_fee = base_financials["daily_storage_fee"] * average_growth_factor
            min_loan_amount = st.session_state.params.safety_factor * (
                adjusted_daily_storage_fee / (
                    (st.session_state.params.loan_interest_rate / 100) *
                    (1 - st.session_state.params.default_probability) *
                    st.session_state.params.liquidity_factor
                )
            )
            loan_label = "Мин. сумма займа (учёт рисков и динамики) (руб.)"
        else:
            if st.session_state.params.loan_interest_rate > 0:
                min_loan_amount = base_financials["daily_storage_fee"] / (st.session_state.params.loan_interest_rate / 100)
            else:
                min_loan_amount = 0.0
            loan_label = "Мин. сумма займа (базовый расчёт) (руб.)"

        # Создание вкладок в основной части приложения
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Общие результаты", "📈 Прогнозирование", "🔍 Точка безубыточности", "📋 Детализация"])
        
        # Вкладка 1: Общие результаты
        with tab1:
            st.header("📊 Общие результаты")
            
            # Создаём 3 колонки для размещения метрик
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.subheader("🔑 Ключевые метрики")
                metrics = {
                    "📈 Общий доход (руб./мес.)": f"{base_financials['total_income']:,.2f}",
                    "💸 Общие расходы (руб./мес.)": f"{base_financials['total_expenses']:,.2f}",
                    "💰 Прибыль (руб./мес.)": f"{base_financials['profit']:,.2f}"
                }
                display_metrics_card(metrics, col1)
            
            with col2:
                st.subheader("📊 Финансовые показатели")
                metrics = {
                    "📈 Маржа прибыли (%)": f"{profit_margin:.2f}%",
                    "🔍 Рентабельность (%)": f"{profitability:.2f}%",
                    "💳 " + loan_label: f"{min_loan_amount:,.2f}",
                    "📈 ROI (%)": f"{roi:.2f}%",
                    "📉 IRR (%)": f"{irr:.2f}%"
                }
                display_metrics_card(metrics, col2)
            
            with col3:
                st.subheader("🛍️ Дополнительные метрики")
                metrics = {
                    "🛍️ Доход от реализации (руб.)": f"{base_financials['realization_income']:,.2f}",
                    "📢 Доход от маркетинга (руб.)": f"{base_financials['marketing_income']:,.2f}",
                    "💸 Единовременные расходы (руб.)": f"{st.session_state.params.one_time_expenses:,.2f}"
                    # Добавьте дополнительные метрики при необходимости
                }
                display_metrics_card(metrics, col3)
            
            # Уведомления на основе метрик
            if profit_margin < 10:
                st.warning("⚠️ Маржа прибыли ниже 10%. Рассмотрите возможность оптимизации расходов или увеличения тарифов.")
            elif profit_margin > 50:
                st.success("✅ Отличная маржа прибыли!")
        
            if roi < 0:
                st.warning("⚠️ ROI отрицательный. Ваши инвестиции не окупаются.")
            elif roi > 20:
                st.success("✅ Высокий ROI. Инвестиции окупаются эффективно.")
        
            if irr < 0:
                st.warning("⚠️ IRR отрицательный. Проект не приносит дохода.")
            elif irr > 15:
                st.success("✅ Высокий IRR. Проект является прибыльным.")
            
            # Разделение метрик и графиков с использованием горизонтальной линии
            st.markdown("---")

            # Структура доходов (улучшенная круговая диаграмма с более современной цветовой схемой)
            st.subheader("📈 Структура доходов (круговая диаграмма)")
            labels = [
                "Простое хранение",
                "Займы (после реализации)",
                "Реализация невостребованных",
                "VIP-хранение",
                "Краткосрочное хранение",
                "Маркетинг"
            ]
            values = [
                base_financials["storage_income"],
                base_financials["loan_income_after_realization"],
                base_financials["realization_income"],
                base_financials["vip_income"],
                base_financials["short_term_income"],
                base_financials["marketing_income"]
            ]
            values = [max(v, 0) for v in values]  # Убираем отрицательные значения
            if sum(values) <= 0:
                labels = ["Нет данных"]
                values = [0]

            # Используем Plotly для интерактивной круговой диаграммы с улучшенной визуализацией
            fig = px.pie(
                names=labels,
                values=values,
                title="Структура доходов",
                hole=0.3,
                color_discrete_sequence=px.colors.sequential.RdBu,
                template="plotly_dark"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Минимальная выручка для BEP с улучшенными визуальными элементами
            st.markdown("---")
            st.subheader("📈 Безубыточность (BEP) в денежном выражении")
            st.metric(
                label="💸 Выручка для BEP (руб./мес.)",
                value=f"{base_financials['total_expenses']:,.2f}",
                delta=None
            )
            st.markdown("""
                При этой ежемесячной выручке доходы полностью покрывают расходы.  
                Выше этого порога — вы в прибыли, ниже — в убытке.
            """)

        # Вкладка 2: Прогнозирование
        with tab2:
            st.header("📈 Прогнозирование")
            forecast_enabled = st.checkbox("🔍 Включить прогнозирование с машинным обучением", value=False)
            if st.session_state.params.time_horizon > 1:
                # Создание простого прогноза на основе текущих данных
                df_projections = pd.DataFrame({
                    "Месяц": np.arange(1, st.session_state.params.time_horizon + 1),
                    "Доходы (руб.)": np.linspace(base_financials["total_income"], base_financials["total_income"] * 1.2, st.session_state.params.time_horizon),
                    "Расходы (руб.)": np.linspace(base_financials["total_expenses"], base_financials["total_expenses"] * 1.1, st.session_state.params.time_horizon),
                })
                df_projections["Прибыль (руб.)"] = df_projections["Доходы (руб.)"] - df_projections["Расходы (руб.)"]

                st.subheader("📊 Финансовые показатели по месяцам")
                st.dataframe(df_projections.style.format({"Доходы (руб.)": "{:,.2f}", "Расходы (руб.)": "{:,.2f}", "Прибыль (руб.)": "{:,.2f}"}))
                
                st.markdown("---")
                st.subheader("📈 Динамика финансовых показателей")
                # Преобразование данных в длинный формат для Plotly
                df_long = df_projections.melt(
                    id_vars="Месяц",
                    value_vars=["Доходы (руб.)", "Расходы (руб.)", "Прибыль (руб.)"],
                    var_name="Показатель",
                    value_name="Значение"
                )

                # Создание линейного графика с улучшенными визуальными элементами
                fig = px.line(
                    df_long,
                    x="Месяц",
                    y="Значение",
                    color="Показатель",
                    title="Динамика финансовых показателей",
                    labels={"Значение": "Сумма (руб.)", "Месяц": "Месяц"},
                    template="plotly_dark",
                    hover_data={"Значение": ':.2f'}
                )
                fig.update_traces(mode='lines+markers', hovertemplate='%{y:,.2f}')
                fig.update_layout(
                    hovermode='x unified',
                    legend_title_text='Показатели'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Добавляем столбчатую диаграмму с улучшенной визуализацией
                st.markdown("---")
                st.subheader("📊 Сравнение Доходов, Расходов и Прибыли по Месяцам")
                fig_bar = px.bar(
                    df_projections,
                    x="Месяц",
                    y=["Доходы (руб.)", "Расходы (руб.)", "Прибыль (руб.)"],
                    title="Сравнение Доходов, Расходов и Прибыли по Месяцам",
                    labels={"value": "Сумма (руб.)", "Месяц": "Месяц"},
                    barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    template="plotly_dark"
                )
                fig_bar.update_traces(hovertemplate='%{y:,.2f}')
                st.plotly_chart(fig_bar, use_container_width=True)

                # Включение прогнозирования с машинным обучением
                if forecast_enabled:
                    st.markdown("---")
                    st.subheader("📊 Прогноз финансовых показателей (ML)")
                    
                    # Создание искусственных исторических данных (пример)
                    historical_months = np.arange(1, st.session_state.params.time_horizon + 1)
                    historical_income = base_financials["total_income"] * (1 + 0.05 * (historical_months - 1))  # Пример роста 5% в месяц
                    historical_expenses = base_financials["total_expenses"] * (1 + 0.03 * (historical_months - 1))  # Пример роста 3% в месяц
                    df_historical = pd.DataFrame({
                        "Месяц": historical_months,
                        "Доходы (руб.)": historical_income,
                        "Расходы (руб.)": historical_expenses
                    })
                    
                    # Обучение модели на исторических данных
                    model_income = train_ml_model(df_historical, "Доходы (руб.)")
                    model_expenses = train_ml_model(df_historical, "Расходы (руб.)")
                    
                    # Прогнозирование на будущие месяцы
                    future_months = np.arange(st.session_state.params.time_horizon + 1, st.session_state.params.time_horizon + 7)  # Прогноз на 6 месяцев
                    predicted_income = predict_with_model(model_income, future_months)
                    predicted_expenses = predict_with_model(model_expenses, future_months)
                    
                    # Отображение прогнозов
                    df_forecast = pd.DataFrame({
                        "Месяц": future_months,
                        "Прогноз Доходов (руб.)": predicted_income,
                        "Прогноз Расходов (руб.)": predicted_expenses
                    })
                    st.subheader("📈 Прогноз финансовых показателей (ML)")
                    st.dataframe(df_forecast.style.format({"Прогноз Доходов (руб.)": "{:,.2f}", "Прогноз Расходов (руб.)": "{:,.2f}"}))
                    
                    # Визуализация прогнозов
                    df_forecast_long = df_forecast.melt(
                        id_vars="Месяц",
                        value_vars=["Прогноз Доходов (руб.)", "Прогноз Расходов (руб.)"],
                        var_name="Показатель",
                        value_name="Значение"
                    )
                    
                    fig_ml = px.line(
                        df_forecast_long,
                        x="Месяц",
                        y="Значение",
                        color="Показатель",
                        title="Прогноз финансовых показателей с использованием ML",
                        labels={"Значение": "Сумма (руб.)", "Месяц": "Месяц"},
                        template="plotly_dark",
                        hover_data={"Значение": ':.2f'}
                    )
                    fig_ml.update_traces(mode='lines+markers', hovertemplate='%{y:,.2f}')
                    st.plotly_chart(fig_ml, use_container_width=True)
                    
                    # Добавление столбчатой диаграммы
                    st.markdown("---")
                    st.subheader("📊 Сравнение Доходов, Расходов и Прибыли по Месяцам")
                    fig_bar_ml = px.bar(
                        df_projections,
                        x="Месяц",
                        y=["Доходы (руб.)", "Расходы (руб.)", "Прибыль (руб.)"],
                        title="Сравнение Доходов, Расходов и Прибыли по Месяцам",
                        labels={"value": "Сумма (руб.)", "Месяц": "Месяц"},
                        barmode='group',
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        template="plotly_dark"
                    )
                    fig_bar_ml.update_traces(hovertemplate='%{y:,.2f}')
                    st.plotly_chart(fig_bar_ml, use_container_width=True)
            else:
                st.info("Для прогнозирования установите горизонт прогноза более 1 месяца.")

        # Вкладка 3: Точка безубыточности
        with tab3:
            st.header("🔍 Точка безубыточности (BEP)")
            st.subheader("Определение BEP для выбранного параметра")
            
            storage_type = st.selectbox(
                "🔍 Вид хранения",
                ["Простое хранение", "Хранение с займами", "VIP-хранение", "Краткосрочное хранение"],
                help="Выберите вид хранения для расчёта BEP."
            )

            parameter_options = {}
            base_param_value = 0.0

            if storage_type == "Простое хранение":
                parameter_options = {"Тариф простого хранения (руб./м²/мес.)": "storage_fee"}
                base_param_value = st.session_state.params.storage_fee
            elif storage_type == "Хранение с займами":
                parameter_options = {"Ставка по займам (%)": "loan_interest_rate"}
                base_param_value = st.session_state.params.loan_interest_rate
            elif storage_type == "VIP-хранение":
                parameter_options = {"Дополнительная наценка VIP (руб.)": "vip_extra_fee"}
                base_param_value = st.session_state.params.vip_extra_fee
            else:
                parameter_options = {"Тариф краткосрочного хранения (руб./день/м²)": "short_term_daily_rate"}
                base_param_value = st.session_state.params.short_term_daily_rate

            parameter_choice = st.selectbox(
                "📊 Параметр для BEP",
                list(parameter_options.keys()),
                help="Выберите параметр, для которого нужно рассчитать BEP."
            )
            param_key = parameter_options[parameter_choice]

            # Автоматический расчёт BEP
            bep_result = calculate_bep(param_key, base_param_value, calculate_financials, st.session_state.params)

            # Генерация диапазона параметров для графика
            if param_key == "storage_fee":
                param_values = np.linspace(st.session_state.params.storage_fee * 0.5, st.session_state.params.storage_fee * 1.5, 100)
            elif param_key == "loan_interest_rate":
                param_values = np.linspace(st.session_state.params.loan_interest_rate * 0.5, st.session_state.params.loan_interest_rate * 1.5, 100)
            elif param_key == "vip_extra_fee":
                param_values = np.linspace(500, 1500, 100)
            else:  # short_term_daily_rate
                param_values = np.linspace(st.session_state.params.short_term_daily_rate * 0.5, st.session_state.params.short_term_daily_rate * 1.5, 100)

            # Выполнение анализа чувствительности
            df_sensitivity = perform_sensitivity_analysis(st.session_state.params, param_key, param_values)

            # Поиск точки безубыточности (BEP) - где прибыль пересекает ноль
            bep_row = df_sensitivity[(df_sensitivity["Прибыль (руб.)"] >= 0) & (df_sensitivity["Прибыль (руб.)"].shift(1) < 0)]
            if not bep_row.empty:
                bep_value = bep_row.iloc[0]["Параметр"]
                st.success(f"Точка безубыточности для {parameter_choice}: **{bep_value:.2f}**")
                # Обновляем vip_extra_fee, если это необходимо
                if param_key == "vip_extra_fee":
                    st.session_state.params.vip_extra_fee = bep_value
                # Визуализация
                fig = px.line(df_sensitivity, x="Параметр", y="Прибыль (руб.)",
                              title=f"Влияние {parameter_choice} на прибыль",
                              labels={"Параметр": parameter_choice, "Прибыль (руб.)": "Прибыль (руб./мес.)"},
                              template="plotly_dark")
                fig.add_vline(x=bep_value, line_dash="dash", line_color="red",
                             annotation=dict(text="BEP", x=bep_value, y=0, showarrow=True, arrowhead=1))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Не удалось найти точку безубыточности в данном диапазоне.")

        # Вкладка 4: Детализация
        with tab4:
            st.header("📋 Детализация")
            st.subheader("📦 Общее количество вещей")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Простое хранение", f"{int(items['stored_items']):,}")
            col2.metric("Хранение с займами", f"{int(items['total_items_loan']):,}")
            col3.metric("VIP-хранение", f"{int(items['vip_stored_items']):,}")
            col4.metric("Краткосрочное хранение", f"{int(items['short_term_stored_items']):,}")

            st.markdown("---")
            st.subheader("📐 Площади для разных типов хранения (м²)")
            storage_data = {
                "Тип хранения": ["Простое хранение", "Хранение с займами", "VIP-хранение", "Краткосрочное хранение"],
                "Площадь (м²)": [
                    st.session_state.params.storage_area,
                    st.session_state.params.loan_area,
                    st.session_state.params.vip_area,
                    st.session_state.params.short_term_area
                ],
                "Количество вещей": [
                    items["stored_items"],
                    items["total_items_loan"],
                    items["vip_stored_items"],
                    items["short_term_stored_items"]
                ],
            }
            df_storage = pd.DataFrame(storage_data)
            st.dataframe(
                df_storage.style.format({"Площадь (м²)": "{:,.2f}", "Количество вещей": "{:,.0f}"}).set_properties(
                    **{'background-color': 'rgba(0, 0, 0, 0)', 'color': '#333333'}
                ).set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#2e7d32'), ('color', 'white'), ('font-size', '14px')]}
                ])
            )

            st.markdown("---")
            st.subheader("📊 Распределение Прибыли по Типам Хранения")
            profit_data = {
                "Тип хранения": ["Простое хранение", "Хранение с займами", "VIP-хранение", "Краткосрочное хранение"],
                "Прибыль (руб.)": [
                    base_financials["storage_income"] - (st.session_state.params.storage_area * st.session_state.params.rental_cost_per_m2),
                    base_financials["loan_income_after_realization"] - (st.session_state.params.loan_area * st.session_state.params.rental_cost_per_m2),
                    base_financials["vip_income"] - (st.session_state.params.vip_area * st.session_state.params.rental_cost_per_m2),
                    base_financials["short_term_income"] - (st.session_state.params.short_term_area * st.session_state.params.rental_cost_per_m2)
                ],
                "Ежемесячные расходы (руб.)": [
                    st.session_state.params.storage_area * st.session_state.params.rental_cost_per_m2,
                    st.session_state.params.loan_area * st.session_state.params.rental_cost_per_m2,
                    st.session_state.params.vip_area * st.session_state.params.rental_cost_per_m2,
                    st.session_state.params.short_term_area * st.session_state.params.rental_cost_per_m2
                ],
                "Единовременные расходы (руб.)": [
                    st.session_state.params.one_time_setup_cost,
                    st.session_state.params.one_time_equipment_cost,
                    st.session_state.params.one_time_other_costs,
                    0.0  # Единовременные расходы уже учтены в первом месяце
                ]
            }
            df_profit = pd.DataFrame(profit_data)
            # Применение условного форматирования для выделения убытков
            def highlight_negative(s):
                return ['background-color: #ffcccc' if v < 0 else '' for v in s]
            st.dataframe(
                df_profit.style.apply(highlight_negative, subset=["Прибыль (руб.)"]).format({
                    "Прибыль (руб.)": "{:,.2f}",
                    "Ежемесячные расходы (руб.)": "{:,.2f}",
                    "Единовременные расходы (руб.)": "{:,.2f}"
                }),
                use_container_width=True
            )

            st.markdown("---")
            st.subheader("📥 Скачать результаты")
            df_results = pd.DataFrame({
                "Показатель": [
                    "Общий доход",
                    "Общие расходы",
                    "Прибыль",
                    "Маржа прибыли (%)",
                    "Рентабельность (%)",
                    "ROI (%)",
                    "IRR (%)",
                    "Доход от реализации",
                    "Доход от маркетинга",
                    ("Мин. сумма займа (учёт рисков и динамики) (руб.)" if not disable_extended and st.session_state.params.loan_interest_rate > 0 else "Мин. сумма займа (базовый расчёт) (руб.)"),
                    "Единовременные расходы (руб.)",
                    "Налоги (руб./мес.)",
                    "Страхование (руб./мес.)"
                ],
                "Значение": [
                    base_financials["total_income"],
                    base_financials["total_expenses"],
                    base_financials["profit"],
                    profit_margin,
                    profitability,
                    roi,
                    irr,
                    base_financials["realization_income"],
                    base_financials["marketing_income"],
                    min_loan_amount,
                    st.session_state.params.one_time_expenses,
                    st.session_state.params.taxes,
                    st.session_state.params.insurance_expenses
                ]
            })
            generate_download_link(df_results)
            generate_excel_download(df_results)

            # Поскольку функции генерации PDF отключены, удаляем или заменяем соответствующие элементы
            st.markdown("---")
            st.subheader("📝 Генерация PDF отчёта")
            st.info("Функция генерации PDF отчётов отключена.")

            if st.session_state.params.loan_interest_rate == 0:
                st.warning("⚠️ Внимание: ставка по займам равна 0. Доход от займов будет отсутствовать.")

# =============================================================================
# Информационное сообщение внизу страницы с улучшенной стилизацией
# =============================================================================
st.info("""
### 📌 Как использовать приложение:
1. **Настройте параметры** в боковой панели слева.
2. В разделе **"📦 Параметры хранения"** выберите режим распределения долей: автоматический или ручной.
   - В **автоматическом режиме** используйте слайдеры для настройки долей, которые автоматически нормализуются до 100%.
   - В **ручном режиме** введите доли хранения вручную. Сумма долей должна равняться 100%.
3. В разделе **"📦 Параметры хранения"** также выберите или введите дневной тариф для краткосрочного хранения.
4. Месячный доход краткосрочного хранения = дневной тариф × площадь × 30.
5. Перейдите на вкладку **"📊 Общие результаты"**, чтобы **просмотреть итоговые метрики**.
6. Используйте вкладку **"🔍 Точка безубыточности"** для **анализа точки безубыточности** выбранного параметра.
7. В разделе **"📈 Прогнозирование"** наблюдайте за **динамикой финансовых показателей** на выбранный горизонт.
   - При необходимости включите **прогнозирование с использованием машинного обучения**.
8. **Скачивайте результаты** в формате CSV или Excel при необходимости.
9. **Сохраняйте и загружайте сценарии** для быстрого доступа к часто используемым настройкам.
    
**Точка безубыточности (BEP)** — это значение параметра (тариф, ставка или наценка), при котором прибыль становится 0. Выше этой точки — прибыль, ниже — убыток.
""")