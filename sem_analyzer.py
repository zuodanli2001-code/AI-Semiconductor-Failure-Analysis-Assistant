import cv2
import numpy as np


def analyze_sem_image(image_bytes):
    """
    SEM 图像基础分析工具。

    输入：
        image_bytes: 上传图片的二进制数据

    输出：
        一个结构化字典，供后续 Agent 使用。
    """

    # =========================
    # 1. 读取图片
    # =========================

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_GRAYSCALE
    )

    # 图片读取失败
    if image is None:
        return {
            "status": "error",
            "message": "无法读取 SEM 图像"
        }

    # =========================
    # 2. 获取图像基本信息
    # =========================

    height, width = image.shape

    # =========================
    # 3. 计算灰度统计特征
    # =========================

    mean_intensity = float(
        np.mean(image)
    )

    std_intensity = float(
        np.std(image)
    )

    min_intensity = int(
        np.min(image)
    )

    max_intensity = int(
        np.max(image)
    )

    # =========================
    # 4. 边缘检测
    # =========================

    edges = cv2.Canny(
        image,
        50,
        150
    )

    edge_pixels = int(
        np.sum(edges > 0)
    )

    total_pixels = width * height

    edge_ratio = edge_pixels / total_pixels

    # =========================
    # 5. 简单异常特征判断
    # =========================

    observations = []

    if edge_ratio > 0.08:
        observations.append(
            "图像中存在较丰富的边缘结构"
        )

    if std_intensity > 50:
        observations.append(
            "图像灰度分布较宽，存在明显的亮暗区域差异"
        )

    if edge_ratio <= 0.08:
        observations.append(
            "图像整体边缘结构相对较少"
        )

    if not observations:
        observations.append(
            "暂未发现明显的异常图像特征"
        )

    # =========================
    # 6. 返回结构化结果
    # =========================

    result = {
        "status": "success",

        "image_info": {
            "width": width,
            "height": height,
            "mean_intensity": round(
                mean_intensity,
                2
            ),
            "std_intensity": round(
                std_intensity,
                2
            ),
            "min_intensity": min_intensity,
            "max_intensity": max_intensity
        },

        "image_features": {
            "edge_ratio": round(
                edge_ratio,
                4
            )
        },

        "observations": observations,

        "ai_analysis": {
            "defect_type": "Not determined",
            "confidence": 0.0,
            "severity": "Not determined"
        }
    }

    return result