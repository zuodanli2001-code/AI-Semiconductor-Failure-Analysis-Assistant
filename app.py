
import csv
import io
import json
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from sem_analyzer import analyze_sem_image
from rag_retriever import search_knowledge
from agent import SupervisorAgent


# ============================================================
# 1. 页面设置
# ============================================================

st.set_page_config(
    page_title="AI Semiconductor Failure Analysis Assistant",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI 半导体失效分析助手")
st.caption("AI Semiconductor Failure Analysis Assistant | MVP Prototype")


# ============================================================
# 2. Session State
# ============================================================

DEFAULT_STATE = {
    "last_report": None,
    "last_agent_result": None,
    "last_sem_result": None,
    "last_rag_result": None,
    "last_csv_result": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 3. 通用工具函数
# ============================================================

def display_result(data):
    """显示字典、列表或普通文本。"""
    if isinstance(data, (dict, list)):
        st.json(data)
    else:
        st.write(data)


def get_image_bytes(uploaded_file):
    """读取上传图像的二进制内容。"""
    if uploaded_file is None:
        return None
    return uploaded_file.getvalue()


def run_local_sem_analysis(image_bytes):
    """调用本地 SEM 图像分析，不调用大模型 API。"""
    try:
        return analyze_sem_image(image_bytes)
    except Exception as exc:
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
        }


def run_local_rag(query, top_k=3):
    """调用本地知识检索模块，不调用大模型 API。"""
    try:
        results = search_knowledge(query, top_k=top_k)

        return {
            "status": "success",
            "query": query,
            "count": len(results),
            "results": results,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# 4. CSV 编码与分隔符自动识别
# ============================================================

def read_uploaded_csv(uploaded_file):
    """
    自动读取 CSV：
    1. 检查是否误传 Excel 文件；
    2. 尝试多种常见编码；
    3. 自动识别逗号、分号、制表符等分隔符；
    4. 返回 DataFrame 和实际使用的编码。
    """

    raw_bytes = uploaded_file.getvalue()

    if not raw_bytes:
        raise ValueError("上传的文件为空，请重新选择文件。")

    # 检查是否将 Excel 文件误当作 CSV 上传
    # .xlsx 通常以 ZIP 文件头 PK 开头
    if raw_bytes.startswith(b"PK\x03\x04"):
        raise ValueError(
            "检测到你上传的可能是 Excel（.xlsx）文件，"
            "而不是 CSV。请在 Excel 中打开文件，"
            "选择“另存为”，文件类型选择“CSV UTF-8（逗号分隔）(*.csv)”，"
            "保存后重新上传。"
        )

    # 老版本 Excel .xls 的常见文件头
    if raw_bytes.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        raise ValueError(
            "检测到你上传的可能是旧版 Excel（.xls）文件。"
            "请在 Excel 中另存为 CSV UTF-8 文件后重新上传。"
        )

    # 优先尝试常见中文及 UTF-8 编码
    encodings = [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
        "big5",
    ]

    decoded_text = None
    encoding_used = None
    decode_errors = []

    for encoding in encodings:
        try:
            decoded_text = raw_bytes.decode(encoding)
            encoding_used = encoding
            break
        except UnicodeDecodeError as exc:
            decode_errors.append(f"{encoding}: {exc}")

    # 如果常见编码均失败，尝试自动识别编码
    if decoded_text is None:
        try:
            from charset_normalizer import from_bytes

            detected = from_bytes(raw_bytes).best()

            if detected is not None:
                decoded_text = str(detected)
                encoding_used = (
                    detected.encoding or "自动识别编码"
                )
        except ImportError:
            pass
        except Exception:
            pass

    # 最后尝试宽容解码，避免仅因个别异常字节直接中断
    if decoded_text is None:
        try:
            decoded_text = raw_bytes.decode(
                "gb18030",
                errors="replace",
            )
            encoding_used = "gb18030（替换异常字符）"
        except Exception as exc:
            raise ValueError(
                "文件内容无法解码。请确认上传的是有效的 CSV 文件，"
                "而不是图片、PDF 或 Excel 文件。"
            ) from exc

    # 删除可能存在的 BOM
    decoded_text = decoded_text.lstrip("\ufeff")

    if not decoded_text.strip():
        raise ValueError("文件没有可读取的文本内容。")

    # 自动判断分隔符
    try:
        sample = decoded_text[:10000]
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|",
        )
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    # 优先使用识别出的分隔符读取
    try:
        df = pd.read_csv(
            io.StringIO(decoded_text),
            sep=delimiter,
            engine="python",
        )
    except Exception as first_error:
        # 再尝试 pandas 自动识别分隔符
        try:
            df = pd.read_csv(
                io.StringIO(decoded_text),
                sep=None,
                engine="python",
            )
        except Exception as second_error:
            raise ValueError(
                "文件解码成功，但 CSV 内容解析失败。\n"
                f"首次解析错误：{first_error}\n"
                f"再次解析错误：{second_error}"
            ) from second_error

    # 清理列名两端的空格
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # 避免将明显没有有效数据的文件当作正常结果
    if len(df.columns) == 0:
        raise ValueError("没有读取到有效的数据列。")

    if len(df.columns) == 1 and len(df) == 0:
        raise ValueError(
            "只读取到一列且没有数据行，请检查 CSV 分隔符和文件内容。"
        )

    return df, encoding_used


# ============================================================
# 5. 知识检索结果展示
# ============================================================

def render_rag_results(result):
    """展示本地知识检索结果及来源。"""

    if not isinstance(result, dict):
        st.write(result)
        return

    if result.get("status") != "success":
        st.error(
            result.get("message", "知识库检索失败。")
        )
        return

    results = result.get("results", [])

    if not results:
        st.warning(
            "本地知识库没有检索到相关内容。"
            "这不代表该问题不存在相关研究资料。"
        )
        return

    st.success(
        f"检索完成，共找到 {len(results)} 条参考内容。"
    )

    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            with st.expander(f"参考资料 {index}"):
                st.write(item)
            continue

        source = item.get("source", "未知来源")
        chunk_id = item.get("chunk_id", "未知")
        score = item.get("score")
        content = item.get("content", "")

        with st.expander(
            f"参考资料 {index}｜{source}｜片段 {chunk_id}",
            expanded=(index == 1),
        ):
            st.write(content)
            st.caption(f"来源文件：{source}")

            if score is not None:
                st.caption(f"检索匹配分数：{score}")


# ============================================================
# 6. 分析报告生成
# ============================================================

def build_markdown_report(report):
    """将分析结果整理成 Markdown 报告。"""

    if not isinstance(report, dict):
        report = {"message": str(report)}

    question = report.get(
        "question",
        report.get("query", "未记录"),
    )

    status = report.get("status", "unknown")
    message = report.get(
        "message",
        report.get("answer", "暂无文字结论。"),
    )

    selected_tools = report.get(
        "selected_tools",
        report.get("selected_tool", []),
    )

    if isinstance(selected_tools, str):
        selected_tools = [selected_tools]

    if not isinstance(selected_tools, list):
        selected_tools = []

    tool_results = report.get("tool_results", [])
    validation = report.get("validation")

    lines = [
        "# AI 半导体失效分析报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 报告性质：AI 辅助分析记录",
        "",
        "> 本报告不能替代实验验证、工程师审核或正式失效分析报告。",
        "",
        "## 一、分析任务",
        "",
        f"- 分析问题：{question}",
        f"- 执行状态：{status}",
        f"- 关联图像：{report.get('image_name', '未提供')}",
        "",
        "## 二、AI 分析结果",
        "",
        str(message),
        "",
        "## 三、调用工具",
        "",
    ]

    if selected_tools:
        for tool in selected_tools:
            lines.append(f"- {tool}")
    else:
        lines.append("未记录工具调用信息。")

    lines.extend([
        "",
        "## 四、工具执行结果与参考证据",
        "",
    ])

    if isinstance(tool_results, list) and tool_results:
        for index, record in enumerate(tool_results, start=1):
            if not isinstance(record, dict):
                lines.extend([
                    f"### 工具结果 {index}",
                    "",
                    str(record),
                    "",
                ])
                continue

            tool_name = record.get(
                "tool",
                record.get("name", "未知工具"),
            )

            tool_result = record.get(
                "result",
                record.get("output", {}),
            )

            lines.extend([
                f"### {index}. {tool_name}",
                "",
            ])

            if (
                isinstance(tool_result, dict)
                and tool_name == "rag_search"
            ):
                rag_items = tool_result.get("results", [])

                if rag_items:
                    for hit_index, hit in enumerate(
                        rag_items,
                        start=1,
                    ):
                        if not isinstance(hit, dict):
                            lines.append(str(hit))
                            lines.append("")
                            continue

                        lines.extend([
                            f"**参考内容 {hit_index}**",
                            "",
                            f"- 来源：{hit.get('source', '未知来源')}",
                            f"- 片段：{hit.get('chunk_id', '未知')}",
                            "",
                            str(hit.get("content", "")),
                            "",
                        ])
                else:
                    lines.extend([
                        "本次检索没有返回匹配内容。",
                        "",
                    ])
            else:
                lines.extend([
                    "```json",
                    json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    "```",
                    "",
                ])
    else:
        lines.extend([
            "没有记录工具执行明细。",
            "",
        ])

    lines.extend([
        "## 五、结果校验",
        "",
    ])

    if validation is None:
        lines.append("没有提供单独的结果校验信息。")
    elif isinstance(validation, (dict, list)):
        lines.extend([
            "```json",
            json.dumps(
                validation,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "```",
        ])
    else:
        lines.append(str(validation))

    lines.extend([
        "",
        "## 六、使用限制与后续建议",
        "",
        "- 灰度统计、边缘比例等基础图像特征不能单独确认具体缺陷。",
        "- 知识库内容仅作为参考，不代表当前样品已经发生相同失效。",
        "- 候选失效机制需要结合样品结构、制程信息、对照样品和补充表征验证。",
        "- 可靠性数据的相关性不等同于失效原因的因果证明。",
        "- 正式技术结论应由专业人员结合原始证据审核。",
        "",
    ])

    return "\n".join(lines)


def render_report_preview(report):
    """展示和下载报告。"""

    if not report:
        st.info(
            "当前还没有报告。请先运行 SEM 分析、"
            "可靠性数据分析、知识检索或综合失效分析。"
        )
        return

    markdown_text = build_markdown_report(report)

    st.markdown(markdown_text)

    st.download_button(
        label="下载 Markdown 分析报告",
        data=markdown_text.encode("utf-8"),
        file_name="failure_analysis_report.md",
        mime="text/markdown",
        key="download_markdown_report",
    )

    json_text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    st.download_button(
        label="下载 JSON 原始记录",
        data=json_text.encode("utf-8"),
        file_name="failure_analysis_report.json",
        mime="application/json",
        key="download_json_report",
    )


# ============================================================
# 7. 侧边栏导航
# ============================================================

st.sidebar.title("🔬 功能导航")

page = st.sidebar.radio(
    "选择功能",
    [
        "🏠 项目首页",
        "🔬 SEM 图像分析",
        "📈 可靠性数据分析",
        "📚 专业知识检索",
        "🧠 综合失效分析",
        "📄 分析报告",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "本地图像分析和知识检索默认不调用大模型 API。"
)
st.sidebar.caption(
    "只有确认并点击综合分析按钮后才会调用大模型。"
)


# ============================================================
# 页面一：项目首页
# ============================================================

if page == "🏠 项目首页":

    st.header("项目首页")

    st.write(
        "面向半导体及微纳器件研发场景的 AI 辅助失效分析原型。"
        "系统整合 SEM 图像基础分析、可靠性数据处理、"
        "专业知识检索和大模型辅助推理。"
    )

    st.info(
        "当前版本属于 MVP 原型。基础图像特征和知识检索结果"
        "不能单独作为缺陷定性或失效根因的依据。"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔬 SEM 图像分析")
        st.write("上传图像并提取基础图像特征。")
        st.caption("本地运行，不调用大模型。")

    with col2:
        st.subheader("📈 可靠性数据分析")
        st.write("读取 CSV，查看统计指标和变化趋势。")
        st.caption("支持常见中文 CSV 编码。")

    with col3:
        st.subheader("📚 专业知识检索")
        st.write("检索本地技术资料和知识片段。")
        st.caption("通过已有 RAG 模块执行。")

    st.subheader("系统工作流")

    st.code(
        "用户输入问题 / 上传 SEM 图像 / 上传可靠性 CSV\n"
        "                  ↓\n"
        "       本地图像分析与数据处理\n"
        "                  ↓\n"
        "          本地知识库检索\n"
        "                  ↓\n"
        "      Supervisor Agent 按需调用工具\n"
        "                  ↓\n"
        "        大模型整合结果与证据\n"
        "                  ↓\n"
        "         生成辅助分析报告",
        language="text",
    )


# ============================================================
# 页面二：SEM 图像分析
# ============================================================

elif page == "🔬 SEM 图像分析":

    st.header("SEM 图像分析")

    st.write(
        "本模块调用现有的本地 OpenCV/NumPy 分析程序，"
        "提取图像尺寸、灰度统计和边缘等基础特征。"
    )

    uploaded_image = st.file_uploader(
        "上传 SEM 图像",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        key="sem_upload",
    )

    if uploaded_image is not None:

        image_bytes = get_image_bytes(uploaded_image)

        try:
            image = Image.open(io.BytesIO(image_bytes))

            st.image(
                image,
                caption=uploaded_image.name,
                use_container_width=True,
            )

            st.caption(
                f"图像尺寸：{image.width} × {image.height} px"
            )

        except Exception as exc:
            st.error(f"图像预览失败：{exc}")
            image_bytes = None

        if image_bytes is not None:

            if st.button(
                "运行本地图像分析",
                type="primary",
                key="run_sem_analysis",
            ):
                with st.spinner("正在提取图像基础特征……"):
                    result = run_local_sem_analysis(image_bytes)

                st.session_state.last_sem_result = result

                st.session_state.last_report = {
                    "question": (
                        f"对 SEM 图像 {uploaded_image.name} "
                        "进行基础图像特征分析"
                    ),
                    "status": (
                        result.get("status", "unknown")
                        if isinstance(result, dict)
                        else "unknown"
                    ),
                    "message": "本地 SEM 图像分析结果。",
                    "selected_tools": ["analyze_sem_image"],
                    "tool_results": [
                        {
                            "tool": "analyze_sem_image",
                            "result": result,
                        }
                    ],
                    "image_name": uploaded_image.name,
                }

            if st.session_state.last_sem_result is not None:

                st.subheader("最近一次分析结果")
                display_result(st.session_state.last_sem_result)

                st.warning(
                    "灰度、边缘比例等基础特征不能直接证明存在裂纹、"
                    "孔洞、分层或污染，需要结合局部形貌及其他表征证据判断。"
                )

    else:
        st.info("请上传一张 SEM 图像。")


# ============================================================
# 页面三：可靠性数据分析
# ============================================================

elif page == "📈 可靠性数据分析":

    st.header("可靠性数据分析")

    st.write(
        "上传可靠性测试 CSV 文件，查看数据预览、"
        "缺失值、数值字段和指标变化趋势。"
    )

    st.caption(
        "支持 UTF-8、UTF-8 BOM、GB18030、GBK 等常见编码，"
        "并尝试自动识别分隔符。"
    )

    uploaded_csv = st.file_uploader(
        "上传可靠性测试 CSV",
        type=["csv"],
        key="reliability_upload",
    )

    if uploaded_csv is not None:

        try:
            df, encoding_used = read_uploaded_csv(uploaded_csv)

            st.success(
                f"CSV 读取成功。识别编码：{encoding_used}"
            )

            st.subheader("数据预览")
            st.dataframe(
                df.head(100),
                use_container_width=True,
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("数据行数", len(df))
            col2.metric("数据列数", len(df.columns))
            col3.metric(
                "缺失单元格",
                int(df.isna().sum().sum()),
            )

            st.subheader("字段类型与缺失值")

            dtype_df = pd.DataFrame({
                "字段": [str(column) for column in df.columns],
                "数据类型": [
                    str(dtype) for dtype in df.dtypes
                ],
                "缺失数量": [
                    int(df[column].isna().sum())
                    for column in df.columns
                ],
            })

            st.dataframe(
                dtype_df,
                use_container_width=True,
            )

            # 尝试将本来是数字、但被识别为文本的列转换为数值
            analysis_df = df.copy()

            for column in analysis_df.columns:
                if analysis_df[column].dtype == "object":
                    converted = pd.to_numeric(
                        analysis_df[column],
                        errors="coerce",
                    )

                    non_empty_count = (
                        analysis_df[column].notna().sum()
                    )

                    if (
                        non_empty_count > 0
                        and converted.notna().sum()
                        >= non_empty_count * 0.8
                    ):
                        analysis_df[column] = converted

            numeric_columns = analysis_df.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_columns:

                st.subheader("数值指标分析")

                selected_columns = st.multiselect(
                    "选择要分析的数值字段",
                    numeric_columns,
                    default=numeric_columns[
                        :min(3, len(numeric_columns))
                    ],
                    key="reliability_numeric_columns",
                )

                if selected_columns:

                    chart_df = analysis_df[
                        selected_columns
                    ].copy()

                    st.line_chart(chart_df)

                    st.subheader("统计摘要")

                    st.dataframe(
                        analysis_df[
                            selected_columns
                        ].describe().T,
                        use_container_width=True,
                    )

                else:
                    st.info("请至少选择一个数值字段。")

            else:
                st.warning(
                    "没有识别到数值型字段。"
                    "请检查 CSV 是否包含数值数据，"
                    "以及分隔符是否正确。"
                )

            if st.button(
                "保存当前 CSV 数据概况到报告",
                type="primary",
                key="save_csv_report",
            ):

                summary = {
                    "filename": uploaded_csv.name,
                    "encoding": encoding_used,
                    "rows": int(len(df)),
                    "columns": [
                        str(column) for column in df.columns
                    ],
                    "missing_cells": int(
                        df.isna().sum().sum()
                    ),
                    "dtypes": {
                        str(column): str(dtype)
                        for column, dtype in df.dtypes.items()
                    },
                    "numeric_summary": (
                        analysis_df
                        .select_dtypes(include="number")
                        .describe()
                        .to_dict()
                    ),
                }

                st.session_state.last_csv_result = summary

                st.session_state.last_report = {
                    "question": "可靠性 CSV 数据概况检查",
                    "status": "success",
                    "message": (
                        "已完成 CSV 数据读取、字段类型、缺失值"
                        "和数值字段统计检查。"
                        "该结果不等于可靠性失效机理结论。"
                    ),
                    "selected_tools": ["pandas_csv_analysis"],
                    "tool_results": [
                        {
                            "tool": "pandas_csv_analysis",
                            "result": summary,
                        }
                    ],
                }

                st.success(
                    "数据概况已保存，可进入“分析报告”页面查看和下载。"
                )

        except Exception as exc:

            st.error(
                f"CSV 读取或分析失败：{type(exc).__name__}: {exc}"
            )

            st.info(
                "如果提示上传的是 Excel 文件，请用 Excel 打开，"
                "选择“另存为”，文件类型选“CSV UTF-8（逗号分隔）(*.csv)”，"
                "然后重新上传。"
            )

    else:
        st.info(
            "请上传 CSV 文件。可以使用可靠性测试数据，"
            "也可以先用模拟数据验证页面功能。"
        )


# ============================================================
# 页面四：专业知识检索
# ============================================================

elif page == "📚 专业知识检索":

    st.header("专业知识检索")

    st.write(
        "调用现有 rag_retriever.py，从本地知识库检索相关内容。"
    )

    query = st.text_input(
        "输入检索问题",
        placeholder="例如：SEM 图像中的裂纹应如何判断？",
        key="rag_query",
    )

    top_k = st.slider(
        "返回参考内容数量",
        min_value=1,
        max_value=5,
        value=3,
        key="rag_top_k",
    )

    if st.button(
        "检索本地知识库",
        type="primary",
        key="run_rag_search",
    ):

        if not query.strip():
            st.warning("请先输入检索问题。")

        else:
            with st.spinner("正在检索本地知识库……"):
                result = run_local_rag(
                    query.strip(),
                    top_k=top_k,
                )

            st.session_state.last_rag_result = result

            st.session_state.last_report = {
                "question": query.strip(),
                "status": result.get("status", "unknown"),
                "message": (
                    "以下内容来自本地知识检索，"
                    "不是大模型生成的最终分析结论。"
                ),
                "selected_tools": ["rag_search"],
                "tool_results": [
                    {
                        "tool": "rag_search",
                        "result": result,
                    }
                ],
            }

    if st.session_state.last_rag_result is not None:
        st.subheader("检索结果")
        render_rag_results(
            st.session_state.last_rag_result
        )

    st.caption(
        "检索结果受本地知识库内容和检索算法影响，"
        "请核对原始文献和资料来源。"
    )


# ============================================================
# 页面五：综合失效分析
# ============================================================

elif page == "🧠 综合失效分析":

    st.header("综合失效分析")

    st.write(
        "输入分析问题，可选上传 SEM 图像。"
        "系统调用现有 SupervisorAgent，由 Agent 按任务需要选择工具。"
    )

    uploaded_image = st.file_uploader(
        "可选：上传 SEM 图像",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        key="agent_sem_upload",
    )

    image_bytes = get_image_bytes(uploaded_image)

    if uploaded_image is not None:
        try:
            image = Image.open(io.BytesIO(image_bytes))

            st.image(
                image,
                caption=uploaded_image.name,
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"图像预览失败：{exc}")
            image_bytes = None

    question = st.text_area(
        "描述你的分析问题",
        placeholder=(
            "例如：请分析这张 SEM 图像的基础形貌特征，"
            "结合知识库说明判断裂纹需要哪些证据。"
        ),
        height=120,
        key="agent_question",
    )

    st.warning(
        "点击分析按钮会调用配置好的大模型 API，可能产生费用。"
        "不要上传个人敏感信息、公司机密或未经授权的数据。"
    )

    api_confirmed = st.checkbox(
        "我确认现在调用大模型 API 进行分析。",
        value=False,
        key="confirm_agent_api",
    )

    if st.button(
        "开始 AI 综合分析",
        type="primary",
        disabled=not api_confirmed,
        key="run_agent_analysis",
    ):

        if not question.strip():
            st.warning("请先填写分析问题。")

        else:
            try:
                with st.spinner(
                    "Agent 正在分析任务并调用所需工具……"
                ):
                    agent = SupervisorAgent()

                    result = agent.run(
                        question.strip(),
                        image_bytes=image_bytes,
                    )

                if not isinstance(result, dict):
                    result = {
                        "status": "success",
                        "message": str(result),
                    }

                st.session_state.last_agent_result = result

                st.session_state.last_report = {
                    "question": question.strip(),
                    "status": result.get("status", "unknown"),
                    "message": result.get(
                        "message",
                        result.get("answer", ""),
                    ),
                    "selected_tools": result.get(
                        "selected_tools",
                        result.get("selected_tool", []),
                    ),
                    "tool_results": result.get(
                        "tool_results",
                        [],
                    ),
                    "validation": result.get("validation"),
                    "image_name": (
                        uploaded_image.name
                        if uploaded_image is not None
                        else None
                    ),
                    "raw_agent_result": result,
                }

                st.success("Agent 已完成本次运行。")

            except Exception as exc:

                error_result = {
                    "status": "error",
                    "message": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }

                st.session_state.last_agent_result = error_result

                st.session_state.last_report = {
                    "question": question.strip(),
                    "status": "error",
                    "message": error_result["message"],
                    "selected_tools": [],
                    "tool_results": [],
                }

                st.error(
                    f"Agent 运行失败：{type(exc).__name__}: {exc}"
                )

    if st.session_state.last_agent_result is not None:

        st.subheader("Agent 分析结果")

        result = st.session_state.last_agent_result

        if isinstance(result, dict):

            status = result.get("status")

            if status == "success":
                st.success("Agent 返回成功状态。")
            elif status == "partial":
                st.warning(
                    "Agent 返回了部分结果，请检查工具执行情况。"
                )
            elif status == "error":
                st.error(
                    "Agent 执行过程中出现错误，请查看错误信息。"
                )

            st.markdown("### AI 分析文本")

            st.markdown(
                result.get("message")
                or result.get("answer")
                or "没有返回文字分析结果。"
            )

            with st.expander(
                "查看原始 Agent JSON",
                expanded=False,
            ):
                display_result(result)

        else:
            display_result(result)

        st.caption(
            "模型输出属于辅助分析意见。候选失效机制仍需结合"
            "实验数据、对照样品和专业人员判断进行验证。"
        )

        st.info(
            "本次分析已保存到“分析报告”页面，可以在那里预览和下载。"
        )


# ============================================================
# 页面六：分析报告
# ============================================================

elif page == "📄 分析报告":

    st.header("分析报告")

    st.write(
        "查看最近一次 SEM 分析、可靠性数据分析、"
        "知识检索或 Agent 分析结果。"
    )

    report = st.session_state.last_report

    if report is None:
        st.info(
            "当前会话还没有分析报告。"
            "请先运行一个分析模块。"
        )

    else:
        st.subheader("报告预览")

        render_report_preview(report)

        with st.expander(
            "查看原始报告 JSON",
            expanded=False,
        ):
            display_result(report)

    st.caption(
        "报告是 AI 辅助分析记录，不是经过工程审核的正式失效分析报告。"
    )
