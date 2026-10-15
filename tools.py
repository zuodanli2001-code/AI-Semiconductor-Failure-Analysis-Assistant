from sem_analyzer import analyze_sem_image
from rag_retriever import search_knowledge


def sem_analysis_tool(image_bytes):
    """
    SEM 图像分析工具。

    参数：
        image_bytes: SEM 图像的二进制数据

    返回：
        OpenCV 提取的图像基础特征
    """
    try:
        result = analyze_sem_image(image_bytes)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"SEM 图像分析失败：{type(e).__name__}: {e}",
        }


def rag_search_tool(query, top_k=3):
    """
    本地 RAG 知识检索工具。

    参数：
        query: 用户提出的问题
        top_k: 返回的知识片段数量

    返回：
        匹配的知识片段、来源和关键词匹配分数

    注意：
        本工具只进行本地检索，不调用大模型 API。
    """
    try:
        results = search_knowledge(query, top_k=top_k)

        return {
            "status": "success",
            "query": query,
            "count": len(results),
            "results": results,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"知识库检索失败：{type(e).__name__}: {e}",
        }
