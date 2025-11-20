import tomllib
from pathlib import Path


root_dir = Path(__file__).resolve().parent.parent
model_dir = root_dir / "model"
config_path = root_dir / "conf" / "config.toml"


with open(config_path, "rb") as f:
    cfg = tomllib.load(f)
    stream_cfg = cfg["stream"]
    infer_cfg = cfg["infer"]
    num_streams = len(stream_cfg.keys())
    stream_url = [i.get("url") for i in stream_cfg.values()]
    roi = []
    for stream_config in stream_cfg.values():
        roi.append(
            [
                stream_config.get("roi_x"),
                stream_config.get("roi_y"),
                stream_config.get("roi_width"),
                stream_config.get("roi_height"),
            ]
        )
    confidence_threshold = infer_cfg.get("confidence_threshold")
    nms_threshold = infer_cfg.get("nms_threshold")
    class_names = infer_cfg.get("class_names")
