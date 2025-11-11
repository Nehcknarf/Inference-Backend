from typing import List, Dict

from pydantic import BaseModel


class QuadrantStats(BaseModel):
    """四象限统计结果"""

    q1: Dict[int, int]
    q2: Dict[int, int]
    q3: Dict[int, int]
    q4: Dict[int, int]


class StreamResult(BaseModel):
    """单路流的检测结果"""

    # detections: List[dict]
    quadrant_stats: QuadrantStats


class DualStreamInferenceResponse(BaseModel):
    """定义API的响应体结构"""

    stream_1: StreamResult
    stream_2: StreamResult
    annotated_image: str  # Base64 编码的图片字符串
