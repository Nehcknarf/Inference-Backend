import tomllib
from pathlib import Path


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
config_path = project_root / "conf" / "config.toml"

with open(config_path, "rb") as f:
    cfg = tomllib.load(f)
    stream_cfg = cfg["stream"]
    infer_cfg = cfg["infer"]
    num_streams = len(stream_cfg.keys())
    stream_url = [i.get("url") for i in stream_cfg.values()]
    roi = [list(i.values())[1:] for i in stream_cfg.values()]
    confidence_threshold = infer_cfg.get("confidence_threshold")
    nms_threshold = infer_cfg.get("nms_threshold")
    class_names = infer_cfg.get("class_names")
