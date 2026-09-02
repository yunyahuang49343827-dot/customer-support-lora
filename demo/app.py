"""Streamlit entry point for the polished Stage C9 comparison demo."""

from __future__ import annotations

from html import escape

import streamlit as st

from src.demo.comparison import (
    ADAPTER_PATH,
    CURATED_EXAMPLES,
    benchmark_rows,
    expected_for_message,
    load_model_bundle,
    load_project_evidence,
    repo_root,
    run_inference,
    verify_frozen_integrity,
)


st.set_page_config(page_title="客服分類 LoRA 微調效果比較", page_icon="⚖️", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #f7f9fc; }
    [data-testid="stMainBlockContainer"] { max-width: 1380px; padding-top: 2.2rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: #182230; letter-spacing: -0.02em; }
    .hero-subtitle { color: #536273; font-size: 1.08rem; margin-top: -0.5rem; margin-bottom: 1.8rem; }
    .kpi-card, .model-card, .promotion-card, .expected-card, .response-preview {
        background: #ffffff; border: 1px solid #e3e8ef; border-radius: 16px;
        box-shadow: 0 5px 18px rgba(27, 39, 51, 0.045); padding: 1.25rem 1.35rem;
    }
    .kpi-card { min-height: 148px; border-top: 4px solid var(--accent); }
    .kpi-label { color: #526173; font-size: 0.94rem; font-weight: 650; }
    .kpi-value { color: #142133; font-size: 1.9rem; font-weight: 760; margin: 0.45rem 0 0.15rem; }
    .kpi-delta { color: var(--accent); font-size: 0.92rem; font-weight: 700; }
    .section-title { color: #233044; font-size: 1.22rem; font-weight: 750; margin: 1.8rem 0 0.8rem; }
    .expected-card { min-height: 188px; border-left: 4px solid #6f63d9; }
    .expected-label { color: #5c6677; font-size: 0.82rem; margin-bottom: 0.2rem; }
    .expected-value { color: #172033; font-size: 1rem; font-weight: 700; }
    .model-card { min-height: 326px; border-top: 4px solid var(--accent); }
    .model-title { color: #1e293b; font-size: 1.16rem; font-weight: 760; margin-bottom: 0.9rem; }
    .result-table { width: 100%; border-collapse: collapse; font-size: 0.93rem; }
    .result-table th { color: #667085; font-weight: 650; text-align: left; border-bottom: 1px solid #e5e9f0; padding: 0.55rem 0.4rem; }
    .result-table td { color: #243044; border-bottom: 1px solid #eef1f5; padding: 0.62rem 0.4rem; vertical-align: middle; }
    .result-table td:nth-child(2) { font-weight: 680; word-break: break-word; }
    .latency-note { color: #758196; font-size: 0.82rem; margin-top: 0.85rem; }
    .response-preview { min-height: 138px; border-left: 4px solid var(--accent); }
    .response-title { color: #253247; font-size: 0.92rem; font-weight: 720; margin-bottom: 0.55rem; }
    .response-text { color: #455468; font-size: 0.9rem; line-height: 1.58; display: -webkit-box;
        -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; }
    .promotion-card { border-left: 5px solid #2f9e72; background: #fbfefc; }
    .promotion-title { color: #167454; font-size: 1.12rem; font-weight: 760; }
    .promotion-scope { color: #465568; font-size: 0.92rem; line-height: 1.65; margin-top: 0.55rem; }
    .governance-warning { background: #fff9eb; border: 1px solid #f2d99b; border-radius: 12px;
        color: #765619; padding: 0.85rem 1rem; margin-top: 0.7rem; font-size: 0.9rem; }
    div.stButton > button[kind="primary"] { background: #2563a9; border-color: #2563a9; border-radius: 10px; font-weight: 700; }
    div.stButton > button[kind="primary"]:hover { background: #1d518e; border-color: #1d518e; }
    [data-testid="stExpander"] { background: #ffffff; border-color: #e1e7ee; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_base_model():
    return load_model_bundle(repo_root(), adapter_path=None)


@st.cache_resource(show_spinner=False)
def load_candidate_model():
    return load_model_bundle(repo_root(), adapter_path=ADAPTER_PATH)


@st.cache_data(show_spinner=False)
def load_demo_evidence():
    return load_project_evidence(repo_root())


def safe_text(value) -> str:
    if value is None:
        return "無法取得"
    if isinstance(value, bool):
        return str(value)
    return str(value)


def field_status(result, expected, field: str) -> str:
    if expected is None:
        return "—"
    expected_key = {
        "intent": "expected_intent",
        "category": "expected_category",
        "needs_human": "expected_needs_human",
    }[field]
    return "✅ 符合" if result.get(field) == expected[expected_key] else "❌ 不符合"


def result_rows(result, expected=None):
    return (
        ("意圖", safe_text(result.get("intent")), field_status(result, expected, "intent")),
        ("類別", safe_text(result.get("category")), field_status(result, expected, "category")),
        ("需要人工處理", safe_text(result.get("needs_human")), field_status(result, expected, "needs_human")),
        ("JSON 有效", "有效" if result.get("json_valid") else "無效", "✅ 有效" if result.get("json_valid") else "❌ 無效"),
        ("Schema 合規", "合規" if result.get("schema_compliant") else "不合規", "✅ 合規" if result.get("schema_compliant") else "❌ 不合規"),
    )


def render_kpi(title: str, base: float, lora: float, delta: float, accent: str, base_digits: int, lora_digits: int):
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{accent}">
          <div class="kpi-label">{escape(title)}</div>
          <div class="kpi-value">{base:.{base_digits}f}% → {lora:.{lora_digits}f}%</div>
          <div class="kpi-delta">+{delta:.1f} 個百分點</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_expected(expected):
    st.markdown(
        f"""
        <div class="expected-card">
          <div class="model-title">預期結果（Expected）</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem">
            <div><div class="expected-label">意圖</div><div class="expected-value">{escape(expected['expected_intent'])}</div></div>
            <div><div class="expected-label">類別</div><div class="expected-value">{escape(expected['expected_category'])}</div></div>
            <div><div class="expected-label">需要人工處理</div><div class="expected-value">{escape(str(expected['expected_needs_human']))}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_card(title: str, result, expected, accent: str):
    if result.get("error"):
        st.error(result["error"])
        return
    table_rows = "".join(
        f"<tr><td>{escape(item)}</td><td>{escape(value)}</td><td>{escape(status)}</td></tr>"
        for item, value, status in result_rows(result, expected)
    )
    st.markdown(
        f"""
        <div class="model-card" style="--accent:{accent}">
          <div class="model-title">{escape(title)}</div>
          <table class="result-table">
            <thead><tr><th>項目</th><th>結果</th><th>狀態</th></tr></thead>
            <tbody>{table_rows}</tbody>
          </table>
          <div class="latency-note">推論時間：{result['latency_ms'] / 1000:.3f} 秒</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_response(title: str, result, accent: str):
    if result.get("error"):
        return
    response = result.get("response") or "模型未提供可解析的 response。"
    st.markdown(
        f"""
        <div class="response-preview" style="--accent:{accent}">
          <div class="response-title">{escape(title)}回覆（節錄）</div>
          <div class="response-text">{escape(response)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("▶ 查看完整回覆與 JSON"):
        st.markdown("**完整 Response**")
        st.write(response)
        st.markdown("**原始模型輸出（Raw Model Output）**")
        if result["json_valid"]:
            st.code(result["raw_output"], language="json")
        else:
            st.error("JSON 無效")
            st.code(result["raw_output"] or "（空白輸出）", language="text")
        st.write(f"JSON 狀態：{'有效' if result['json_valid'] else '無效'}")
        st.write(f"Schema 狀態：{'合規' if result['schema_compliant'] else '不合規'}")
        if result["generation_truncated"]:
            st.warning("生成內容已達 frozen 512-token 上限，可能遭到截斷。")


def safe_inference(bundle, message):
    try:
        return run_inference(bundle, message, repo_root())
    except Exception:
        return {"error": "推論失敗。請確認 frozen 模型 artifacts 可用後再試一次。"}


def formatted_benchmark_rows(benchmark):
    return tuple({
        "指標": row["Metric"],
        "Base": f"{row['Base']:.1f}%",
        "LoRA": f"{row['QLoRA']:.1f}%",
        "改善": f"{row['Delta']:+.1f} pp",
    } for row in benchmark_rows(benchmark))


def main():
    st.title("客服分類 LoRA 微調效果比較")
    st.markdown(
        '<div class="hero-subtitle">在固定的 Locked Test 上，微調顯著提升結構化分類與轉介決策能力。</div>',
        unsafe_allow_html=True,
    )

    integrity = verify_frozen_integrity(repo_root(), check_model_availability=True)
    if integrity["status"] != "PASS":
        st.error("Frozen artifact 完整性檢查失敗。")
        st.info("請確認指定的 Base revision、Candidate 01 adapter 與 frozen system prompt 均可在本機取得。")
        st.stop()

    try:
        evidence = load_demo_evidence()
    except Exception:
        st.error("無法載入 Frozen benchmark 或 promotion evidence。")
        st.stop()

    metrics = evidence["benchmark"]["metrics"]
    kpi_columns = st.columns(3, gap="medium")
    with kpi_columns[0]:
        row = metrics["intent_accuracy"]
        render_kpi("意圖分類準確率", row["base"], row["lora"], row["absolute_delta"], "#3578c6", 0, 0)
    with kpi_columns[1]:
        row = metrics["schema_compliance"]
        render_kpi("Schema 合規率", row["base"], row["lora"], row["absolute_delta"], "#2f9e72", 1, 1)
    with kpi_columns[2]:
        row = metrics["escalation_accuracy"]
        render_kpi("轉介判斷準確率", row["base"], row["lora"], row["absolute_delta"], "#7568c9", 0, 1)

    st.markdown('<div class="section-title">測試訊息</div>', unsafe_allow_html=True)
    example_by_label = {item["label"]: item for item in CURATED_EXAMPLES}
    example_labels = ["自訂問題", *example_by_label]
    if "customer_message" not in st.session_state:
        st.session_state.customer_message = "I need help getting my money back."

    def update_example():
        selected = example_by_label.get(st.session_state.example_selector)
        if selected:
            st.session_state.customer_message = selected["message"]

    input_column, expected_column = st.columns((3, 2), gap="large")
    with input_column:
        st.selectbox("範例問題", example_labels, key="example_selector", on_change=update_example)
        message = st.text_area("客服訊息", key="customer_message", height=132)
        compare_clicked = st.button("▶ 比較 Base 與 LoRA", type="primary", width="stretch")
    expected = expected_for_message(message)
    with expected_column:
        if expected:
            render_expected(expected)

    if compare_clicked:
        if not message.strip():
            st.warning("請輸入客服訊息。")
        else:
            with st.spinner("正在載入 Frozen Base 與 Candidate 01 模型…"):
                try:
                    base_bundle = load_base_model()
                    lora_bundle = load_candidate_model()
                except Exception:
                    st.error("Frozen 模型或 adapter 無法使用，因此未執行推論。")
                    base_bundle = lora_bundle = None
            if base_bundle is not None and lora_bundle is not None:
                with st.spinner("正在以相同 prompt 與 deterministic decoding 執行比較…"):
                    base_result = safe_inference(base_bundle, message)
                    lora_result = safe_inference(lora_bundle, message)
                left, right = st.columns(2, gap="large")
                with left:
                    render_model_card("Base Model", base_result, expected, "#d68191")
                with right:
                    render_model_card("LoRA Candidate 01", lora_result, expected, "#4c8dcc")

                st.markdown('<div class="section-title">模型回覆（Generated Response）</div>', unsafe_allow_html=True)
                left_response, right_response = st.columns(2, gap="large")
                with left_response:
                    render_response("Base Model ", base_result, "#d68191")
                with right_response:
                    render_response("LoRA Candidate 01 ", lora_result, "#4c8dcc")

    st.write("")
    with st.expander("▶ 查看 Locked Test 完整指標結果"):
        st.dataframe(formatted_benchmark_rows(evidence["benchmark"]), width="stretch", hide_index=True)

    st.write("")
    st.markdown(
        """
        <div class="promotion-card">
          <div class="promotion-title">✅ Candidate 01 已通過 Promotion Gate</div>
          <div class="promotion-scope"><strong>適用範圍</strong><br>
          • 結構化分類<br>• Schema-constrained routing<br>• 人工轉介判斷</div>
        </div>
        <div class="governance-warning"><strong>PROMOTE ≠ unrestricted production approval</strong><br>
        Promotion 不代表可不受限制地直接投入正式客服環境。</div>
        """,
        unsafe_allow_html=True,
    )

    constraints = evidence["constraints"]
    with st.expander("▶ 查看模型限制與部署邊界"):
        st.markdown("### 回覆限制")
        st.markdown("- 仍可能產生 unsupported policy / capability claims\n- 回覆有時較冗長\n- 曾觀察到 1 次 generation degeneration / truncation")
        st.markdown("### 外部資訊需求")
        st.write("以下資訊需要 external grounding：")
        for item in constraints["requires_external_grounding_for"]:
            st.markdown(f"- {item}")
        st.markdown("### Backend actions")
        st.write("以下操作需要真實 tools / APIs：")
        for item in constraints["requires_backend_tools_for"]:
            st.markdown(f"- {item}")
        st.markdown("### 模型限制")
        st.markdown("- semantically similar intents 仍可能混淆\n- inference latency 較高")
        st.info("QLoRA 顯著改善 structured classification behavior，但生成式 response 仍可能產生 unsupported policy/capability claims，因此模型不應直接被視為企業 factual authority。")


if __name__ == "__main__":
    main()
