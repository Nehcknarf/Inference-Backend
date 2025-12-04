from typing import List, Dict, Optional

from pydantic import BaseModel


class QuadrantStats(BaseModel):
    """四象限统计结果"""

    q1: Dict[str, int]
    q2: Dict[str, int]
    q3: Dict[str, int]
    q4: Dict[str, int]


class StreamResult(BaseModel):
    """单路流的检测结果"""

    # 如果客户端不需要详细检测框信息则返回 None
    detections: Optional[List[dict]] = None
    quadrant_stats: QuadrantStats


class DualStreamInferenceResponse(BaseModel):
    """定义API的响应体结构"""

    stream_1: StreamResult
    stream_2: StreamResult
    # 如果客户端不需要图片则返回 None
    annotated_image: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    active_streams: str
    available_infer_devices: List[str]
