from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import pickle

import yaml


@dataclass(frozen=True)
class ImagePairConfig:
    """用途: 管理单个图像对的数据配置, 包含两张输入图像及对应的 ground truth 文件.

    Attributes:
        src_path (str | Path): 第一张源图像的文件名, 例如 "1a_notredame.jpg".
        dst_path (str | Path): 第二张目标图像的文件名, 例如 "1b_notredame.jpg".
        gt_path (str | Path): 该图像对对应的 ground truth 文件名, 例如 "notredame.pkl".

            **必须使用 `pickle.load(f, encoding="latin1")`**, 否则 MATLAB 生成的 uint8 数据无法正确解码.

    Notes:
        反序列化后得到一个 dict, 包含以下键:
        - x1 (np.ndarray): 图像 A 中特征点的 x 坐标
        - y1 (np.ndarray): 图像 A 中特征点的 y 坐标
        - x2 (np.ndarray): 图像 B 中对应匹配点的 x 坐标
        - y2 (np.ndarray): 图像 B 中对应匹配点的 y 坐标
    """

    src_path: str | Path
    dst_path: str | Path
    gt_path: str | Path

    def __post_init__(self) -> None:
        if isinstance(self.src_path, str):
            object.__setattr__(self, "src_path", Path(self.src_path))
        if isinstance(self.dst_path, str):
            object.__setattr__(self, "dst_path", Path(self.dst_path))
        if isinstance(self.gt_path, str):
            object.__setattr__(self, "gt_path", Path(self.gt_path))


@dataclass(frozen=True)
class PathConfig:
    """
    用途: 管理项目输入输出路径及图像对配置.

    Attributes:
        data_dir (str | Path): 图像数据目录, 默认 ./data, 支持字符串或 Path 对象输入.
        ground_truth_dir (str | Path): ground truth 数据目录, 默认 ./ground_truth, 支持字符串或 Path 对象输入.
        output_dir (str | Path): 输出结果目录, 默认 ./outputs, 支持字符串或 Path 对象输入.
        image_pairs (list[ImagePairConfig]): 图像对列表, 默认包含 notredame/rushmore/gaudi 三组.
    """

    data_dir: str | Path = Path("./data")
    ground_truth_dir: str | Path = Path("./ground_truth")
    output_dir: str | Path = Path("./outputs")
    image_pairs: list[ImagePairConfig] = field(
        default_factory=lambda: [
            ImagePairConfig(
                src_path="1a_notredame.jpg",
                dst_path="1b_notredame.jpg",
                gt_path="notredame.pkl",
            ),
            ImagePairConfig(
                src_path="2a_rushmore.jpg",
                dst_path="2b_rushmore.jpg",
                gt_path="rushmore.pkl",
            ),
            ImagePairConfig(
                src_path="3a_gaudi.jpg", dst_path="3b_gaudi.jpg", gt_path="gaudi.pkl"
            ),
        ]
    )

    def __post_init__(self) -> None:
        # 转换父目录为 Path 对象
        if isinstance(self.data_dir, str):
            object.__setattr__(self, "data_dir", Path(self.data_dir))
        if isinstance(self.ground_truth_dir, str):
            object.__setattr__(self, "ground_truth_dir", Path(self.ground_truth_dir))
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        # 将 data_dir / ground_truth_dir 拼接到每个图像对的文件名上, 生成完整路径
        data_dir = Path(self.data_dir)
        gt_dir = Path(self.ground_truth_dir)
        concat_pairs = [
            ImagePairConfig(
                src_path=data_dir.joinpath(pair.src_path),
                dst_path=data_dir.joinpath(pair.dst_path),
                gt_path=gt_dir.joinpath(pair.gt_path),
            )
            for pair in self.image_pairs
        ]
        object.__setattr__(self, "image_pairs", concat_pairs)


@dataclass(frozen=True)
class LoggingConfig:
    """
    用途: 管理日志行为配置.

    Attributes:
        log_dir (str | Path): 日志输出目录, 默认 ./outputs/logs, 支持字符串或 Path 对象输入.
        level (str): 日志级别, 默认 "DEBUG", 可选 "DEBUG"/"INFO"/"WARNING"/"ERROR"/"CRITICAL".
        retention (str): 日志文件保留时间, 默认 "7 days", 支持自然语言格式.
        compression (str): 日志文件压缩格式, 默认 "zip", 可选 "zip"/"gz"/"tar".
        file_pattern (str): 日志文件名模式, 默认 "{time:YYYY-MM-DD}.log", 使用 loguru 时间格式化.
    """

    log_dir: str | Path = Path("./outputs/logs")
    level: str = "DEBUG"
    retention: str = "7 days"
    compression: str = "zip"
    file_pattern: str = "{time:YYYY-MM-DD}.log"

    def __post_init__(self) -> None:
        if isinstance(self.log_dir, str):
            object.__setattr__(self, "log_dir", Path(self.log_dir))


@dataclass(frozen=True)
class AppConfig:
    """
    用途: 聚合全局配置入口.

    Attributes:
        paths (PathConfig): 路径及图像对配置.
        logging (LoggingConfig): 日志相关配置.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path) -> "AppConfig":
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(
                f"\033[1;91mInvalid YAML format in {yaml_path}: {e}\033[0m"
            ) from e
        except FileNotFoundError as e:
            raise ValueError(
                f"\033[1;91mConfiguration file not found: {yaml_path}\033[0m"
            ) from e

        paths_dict = config_dict.get("paths", {})
        logging_dict = config_dict.get("logging", {})

        # 处理 image_pairs 列表: 将 dict 列表转为 ImagePairConfig 对象列表
        image_pairs_list = paths_dict.get("image_pairs", [])
        paths_dict["image_pairs"] = [
            ImagePairConfig(**pair) for pair in image_pairs_list
        ]

        return cls(
            paths=PathConfig(**paths_dict),
            logging=LoggingConfig(**logging_dict),
        )


# 全局配置实例 (模块导入时加载一次, 全局唯一)
# 其他模块通过 `from configs.app_config import APP_CONFIG` 导入使用
# _temp_dir = Path(__file__).parent
# _yaml_path = _temp_dir.joinpath("app_config.yml")
_yaml_path = Path("./configs/app_config.yml")
try:
    APP_CONFIG = AppConfig.load_from_yaml(_yaml_path)
except ValueError as e:
    print(f"\033[1;93m[WARNING] 加载 YAML 失败, 使用默认配置: {e}\033[0m")
    APP_CONFIG = AppConfig()

if __name__ == "__main__":
    # 直接运行该模块时, 打印加载的配置内容
    print("Loaded Configuration:")
    print(APP_CONFIG)
    with open(APP_CONFIG.paths.image_pairs[0].gt_path, "rb") as f:
        gt_data = pickle.load(f, encoding="latin1")
        print(gt_data)
