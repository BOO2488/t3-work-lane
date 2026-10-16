# -*- coding: utf-8 -*-
"""
senna_qa_utils.py
纯工具库（与具体多模态后端解耦）。供 Qwen2.5-VL 版数据转换与 QA 生成调用。

导出：
- load_image(path_or_url) -> PIL.Image.Image
- load_images(list_of_path_or_url) -> List[PIL.Image.Image]
- get_obj_acc_or_dec(traj, stop_speed=0.5, acc_thre=0.7) -> {'stop','accelerate','decelerate','const'}
- get_obj_turn_or_lane_change(traj, lat_thresh=3.5, turn_angle_thre=25.0)
    -> {'right lane change','left lane change','right turn','left turn','straight'}
- get_obj_rel_position(loc_xy) -> {'front','front left','back left','front right','back right','back'}

说明：
- 不再引入/依赖任意 LLaVA / llava_next 符号或处理流程。
- 仅提供图像加载与几何/运动学判定的工具函数。
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import Iterable, List

import numpy as np
import requests
from PIL import Image


__all__ = [
    "load_image",
    "load_images",
    "get_obj_acc_or_dec",
    "get_obj_turn_or_lane_change",
    "get_obj_rel_position",
]


# -------------------------------
# Image loading helpers
# -------------------------------
def load_image(image_file: str) -> Image.Image:
    """
    支持本地路径与 http(s) URL；统一转为 RGB。
  
    加载图片，支持本地路径和网络URL，统一转换为RGB格式
    输入示例：
    load_image("test.jpg")          # 本地图片
    load_image("https://xxx.jpg")   # 网络图片
    输出：PIL Image对象（RGB格式）
    """
    if image_file.startswith(("http://", "https://")):
        resp = requests.get(image_file, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(image_file).convert("RGB")
    return img


def load_images(image_files: Iterable[str]) -> List[Image.Image]:
    """
    批量加载多张图片，调用load_image函数
    输入示例：
    load_images(["img1.jpg", "img2.jpg"])
    输出：PIL Image对象列表
    """
    return [load_image(p) for p in image_files]


# -------------------------------
# Kinematics / semantics helpers
# -------------------------------
def get_obj_acc_or_dec(
    traj: np.ndarray | list,
    stop_speed: float = 0.5,   # 末速 <0.5 m/s 判 STOP
    acc_thre: float = 0.7      # 平均加速度阈值 ≈ 0.7 m/s²
) -> str:
    """
    根据目标轨迹计算运动状态：停车/加速/减速/匀速
    输入：轨迹点序列（单位：米，采样周期 0.1s），形状 [T,2] 或 list。
    输入示例：[[0,0],[1,0],[2,0]]
    输出：'stop' / 'accelerate' / 'decelerate' / 'const'
    """
    traj = np.asarray(traj)
    # 最少需要两帧,轨迹格式不合法则默认匀速
    if traj.ndim != 2 or traj.shape[0] < 2:
        return "const"

    v = np.linalg.norm(traj[1:] - traj[:-1], axis=-1) / 0.1  # 计算瞬时速度 m/s
    vend, v0 = float(v[-1]), float(v[0])
    T = (len(traj) - 1) * 0.1  # 秒
    if T <= 0:
        return "const"

    # 末速度低于阈值判定停车
    if vend < stop_speed:
        return "stop"

    # 计算平均加速度判断运动状态
    a_mean = (vend - v0) / T  # m/s²
    if a_mean > acc_thre:
        return "accelerate"
    if a_mean < -acc_thre:
        return "decelerate"
    return "const"


def get_obj_turn_or_lane_change(
    traj: np.ndarray | list,
    lat_thresh: float = 3.5,          # 约等一条车道宽
    turn_angle_thre: float = 25.0     # 拐弯阈值（度）
) -> str:
    """
    根据轨迹判断行驶行为：左/右变道、左/右转弯、直行
    输入：轨迹点序列（单位：米，采样周期 0.1s），形状 [T,2] 或 list。
    输入示例：[[0,0],[0,5],[3.5,10]]
    输出：'right lane change' / 'left lane change' / 'right turn' / 'left turn' / 'straight'
    约定：y 为前进方向（longitudinal），x 为横向（lateral）。
    """
    traj = np.asarray(traj)
    # 轨迹不合法默认直行
    if traj.ndim != 2 or traj.shape[0] < 2:
        return "straight"

    dx, dy = traj[-1] - traj[0]  # x: lateral, y: longitudinal  总横向/纵向位移

    # 几乎没有横向位移 → 直行
    if abs(dx) < 0.5:
        return "straight"

    # 与前向夹角（注意使用 atan2(dx, dy) 以 y 作为前进轴）
    theta = math.degrees(math.atan2(dx, dy))

    # 变道：横移达到一条道宽，且偏角不大
    if abs(dx) >= lat_thresh and abs(theta) < turn_angle_thre:
        return "right lane change" if dx > 0 else "left lane change"

    # 拐弯：偏角显著
    if abs(theta) >= turn_angle_thre:
        return "right turn" if dx > 0 else "left turn"

    return "straight"


def get_obj_rel_position(loc: np.ndarray | list) -> str:
    """
    根据坐标判定目标相对方位：前/前左/后左/前右/后右/后，
    即用角度划分前后左右 6 个扇形区域，判断目标点落在哪个区域，从而返回方位
    输入示例：[0,5]、[3,2]、[-2,-1]
    输出："front"/"front left"/"back left"/"front right"/"back right"/"back"
                front (55-125°)
                    ↑
                    |
    front left      |    front right
    (110-180°)      |     (0-70°)
    ----------------⊙（自车）-----------------> x
                    |
    back left       |     back right
    (165-235°)      |     (305-360°)
                    |
                back (-145~-35°)

    基于简单 FOV 划分判定相对方位（front/front left/.../back）。
    约定：loc = [x(lateral), y(longitudinal)]，角度 0° 指向 +x 轴，顺时针为正。
    注意：此函数仅粗略划分视场，供描述使用；不是严格可见性判定。
    """
    x, y = float(loc[0]), float(loc[1])

    # 将数学角度转换到 [0, 360)
    angle = math.degrees(math.atan2(y, x))
    angle1 = angle if angle >= 0 else angle + 360.0

    # nuScenes 摄像头视场（粗略）
    cf_fov = 70.0
    cf_start = 90.0 - cf_fov / 2  # 55°
    cam_offset = 55.0
    cb_fov = 110.0

    cf_range  = (cf_start, cf_start + cf_fov)                 # [55, 125]
    cfl_range = (cf_start + cam_offset, cf_start + cf_fov + cam_offset)     # [110, 180]
    cbl_range = (cf_start + 2 * cam_offset, cf_start + cf_fov + 2 * cam_offset)  # [165, 235]
    cfr_range = (cf_start - cam_offset, cf_start + cf_fov - cam_offset)     # [0, 70]
    cbr_range = (cf_start - 2 * cam_offset, cf_start + cf_fov - 2 * cam_offset)  # [-55, 15]
    cb_range  = ((cb_fov - 180) / 2, (cb_fov - 180) / 2 - cb_fov)           # [-35, -145]

    # 前向扇区采用 [0,360) 角度
    if cf_range[0]  <= angle1 < cf_range[1]:
        return "front"
    if cfl_range[0] <= angle1 < cfl_range[1]:
        return "front left"
    if cbl_range[0] <= angle1 < cbl_range[1]:
        return "back left"
    if cfr_range[0] <= angle1 < cfr_range[1]:
        return "front right"

    # 右后使用同一角度系（修正：原实现误用 angle，这里统一用 angle1）
    if cbr_range[0] <= angle1 < cbr_range[1]:
        return "back right"

    # 后向扇区使用原始 [-180,180) 角度定义（保持原逻辑避免回归风险）
    if cb_range[1] <= angle < cb_range[0]:
        return "back"

    # 落在侧向重叠或边界之外
    raise ValueError("Point not in any camera nominal FOV sector.")
