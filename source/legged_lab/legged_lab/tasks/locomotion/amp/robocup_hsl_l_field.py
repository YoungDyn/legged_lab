"""RoboCup HSL Large L-Field scene primitives for Isaac Lab."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils


FIELD_LENGTH = 22.0
FIELD_WIDTH = 14.0
BORDER_STRIP_WIDTH = 1.0
LINE_WIDTH = 0.12
LINE_HEIGHT = 0.012
TURF_HEIGHT = 0.024

GOAL_AREA_DEPTH = 1.0
GOAL_AREA_WIDTH = 5.0
PENALTY_AREA_DEPTH = 3.5
PENALTY_AREA_WIDTH = 7.0
PENALTY_MARK_DISTANCE = 2.5
PENALTY_MARK_RADIUS = 0.075
CENTER_CIRCLE_RADIUS = 2.0
CORNER_ARC_RADIUS = 1.0

GOAL_WIDTH = 2.8
GOAL_DEPTH = 1.5
GOAL_HEIGHT = 1.9
GOAL_POST_RADIUS = 0.055

ROOT_PRIM_PATH = "/World/RoboCup_HSL_L_Field"


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def _material(color: tuple[float, float, float], opacity: float = 1.0) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.85, opacity=opacity)


def _spawn_box(
    prim_path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    yaw: float = 0.0,
    opacity: float = 1.0,
    collision: bool = False,
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        if collision
        else None,
        visual_material=_material(color, opacity),
    )
    cfg.func(prim_path, cfg, translation=translation, orientation=_yaw_quat(yaw))


def _spawn_cylinder(
    prim_path: str,
    radius: float,
    height: float,
    axis: str,
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
) -> None:
    cfg = sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        axis=axis,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        if collision
        else None,
        visual_material=_material(color),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _spawn_line(prim_path: str, start: tuple[float, float], end: tuple[float, float], z: float) -> None:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length <= 0.0:
        return

    center = ((sx + ex) * 0.5, (sy + ey) * 0.5, z)
    _spawn_box(prim_path, (length, LINE_WIDTH, LINE_HEIGHT), center, (1.0, 1.0, 1.0), yaw=math.atan2(dy, dx))


def _spawn_arc(
    prim_path: str,
    center: tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    z: float,
    segments: int = 24,
) -> None:
    previous = None
    for i in range(segments + 1):
        ratio = i / segments
        angle = start_angle + (end_angle - start_angle) * ratio
        point = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
        if previous is not None:
            _spawn_line(f"{prim_path}/seg_{i:02d}", previous, point, z)
        previous = point


def _spawn_goal(prefix: str, side: int) -> None:
    white = (0.95, 0.95, 0.92)
    net = (0.78, 0.78, 0.78)
    mouth_x = side * FIELD_LENGTH * 0.5
    back_x = mouth_x + side * GOAL_DEPTH
    post_y = GOAL_WIDTH * 0.5
    post_z = GOAL_HEIGHT * 0.5
    crossbar_z = GOAL_HEIGHT

    for name, x in (("front", mouth_x), ("rear", back_x)):
        for y_sign in (-1, 1):
            _spawn_cylinder(
                f"{prefix}/{name}_post_{'upper' if y_sign > 0 else 'lower'}",
                GOAL_POST_RADIUS,
                GOAL_HEIGHT,
                "Z",
                (x, y_sign * post_y, post_z),
                white,
                collision=True,
            )
        _spawn_cylinder(
            f"{prefix}/{name}_crossbar",
            GOAL_POST_RADIUS,
            GOAL_WIDTH,
            "Y",
            (x, 0.0, crossbar_z),
            white,
            collision=True,
        )

    for y_sign in (-1, 1):
        _spawn_cylinder(
            f"{prefix}/top_depth_bar_{'upper' if y_sign > 0 else 'lower'}",
            GOAL_POST_RADIUS,
            GOAL_DEPTH,
            "X",
            ((mouth_x + back_x) * 0.5, y_sign * post_y, crossbar_z),
            white,
            collision=True,
        )
        _spawn_cylinder(
            f"{prefix}/ground_depth_bar_{'upper' if y_sign > 0 else 'lower'}",
            GOAL_POST_RADIUS,
            GOAL_DEPTH,
            "X",
            ((mouth_x + back_x) * 0.5, y_sign * post_y, GOAL_POST_RADIUS),
            white,
            collision=True,
        )

    _spawn_box(f"{prefix}/back_net", (0.02, GOAL_WIDTH, GOAL_HEIGHT), (back_x, 0.0, post_z), net, opacity=0.35)
    for y_sign in (-1, 1):
        _spawn_box(
            f"{prefix}/side_net_{'upper' if y_sign > 0 else 'lower'}",
            (GOAL_DEPTH, 0.02, GOAL_HEIGHT),
            ((mouth_x + back_x) * 0.5, y_sign * post_y, post_z),
            net,
            opacity=0.35,
        )


def spawn_robocup_hsl_l_field(env, env_ids, prim_path: str = ROOT_PRIM_PATH) -> None:  # noqa: ARG001
    """Spawn a RoboCup HSL Large L-Field centered at world origin."""

    import isaacsim.core.utils.prims as prim_utils

    if prim_utils.is_prim_path_valid(prim_path):
        return

    prim_utils.create_prim(prim_path, "Xform")
    prim_utils.create_prim(f"{prim_path}/lines", "Xform")
    prim_utils.create_prim(f"{prim_path}/goals", "Xform")

    x_min = -FIELD_LENGTH * 0.5
    x_max = FIELD_LENGTH * 0.5
    y_min = -FIELD_WIDTH * 0.5
    y_max = FIELD_WIDTH * 0.5
    z_line = TURF_HEIGHT + LINE_HEIGHT * 0.5 + 0.001

    _spawn_box(
        f"{prim_path}/turf_with_border",
        (FIELD_LENGTH + 2.0 * BORDER_STRIP_WIDTH, FIELD_WIDTH + 2.0 * BORDER_STRIP_WIDTH, TURF_HEIGHT),
        (0.0, 0.0, TURF_HEIGHT * 0.5),
        (0.10, 0.45, 0.12),
    )

    _spawn_line(f"{prim_path}/lines/goal_line_left", (x_min, y_min), (x_min, y_max), z_line)
    _spawn_line(f"{prim_path}/lines/goal_line_right", (x_max, y_min), (x_max, y_max), z_line)
    _spawn_line(f"{prim_path}/lines/touchline_top", (x_min, y_max), (x_max, y_max), z_line)
    _spawn_line(f"{prim_path}/lines/touchline_bottom", (x_min, y_min), (x_max, y_min), z_line)
    _spawn_line(f"{prim_path}/lines/halfway", (0.0, y_min), (0.0, y_max), z_line)
    _spawn_arc(f"{prim_path}/lines/center_circle", (0.0, 0.0), CENTER_CIRCLE_RADIUS, 0.0, 2.0 * math.pi, z_line, 96)
    _spawn_cylinder(
        f"{prim_path}/lines/center_mark", PENALTY_MARK_RADIUS, LINE_HEIGHT, "Z", (0.0, 0.0, z_line), (1.0, 1.0, 1.0)
    )

    for side_name, side in (("left", -1), ("right", 1)):
        goal_x = side * FIELD_LENGTH * 0.5
        goal_area_inner_x = goal_x - side * GOAL_AREA_DEPTH
        penalty_inner_x = goal_x - side * PENALTY_AREA_DEPTH
        goal_y = GOAL_AREA_WIDTH * 0.5
        penalty_y = PENALTY_AREA_WIDTH * 0.5
        penalty_mark_x = goal_x - side * PENALTY_MARK_DISTANCE

        _spawn_line(f"{prim_path}/lines/{side_name}_goal_area_front", (goal_area_inner_x, -goal_y), (goal_area_inner_x, goal_y), z_line)
        _spawn_line(f"{prim_path}/lines/{side_name}_goal_area_top", (goal_x, goal_y), (goal_area_inner_x, goal_y), z_line)
        _spawn_line(f"{prim_path}/lines/{side_name}_goal_area_bottom", (goal_x, -goal_y), (goal_area_inner_x, -goal_y), z_line)

        _spawn_line(f"{prim_path}/lines/{side_name}_penalty_area_front", (penalty_inner_x, -penalty_y), (penalty_inner_x, penalty_y), z_line)
        _spawn_line(f"{prim_path}/lines/{side_name}_penalty_area_top", (goal_x, penalty_y), (penalty_inner_x, penalty_y), z_line)
        _spawn_line(f"{prim_path}/lines/{side_name}_penalty_area_bottom", (goal_x, -penalty_y), (penalty_inner_x, -penalty_y), z_line)
        _spawn_cylinder(
            f"{prim_path}/lines/{side_name}_penalty_mark",
            PENALTY_MARK_RADIUS,
            LINE_HEIGHT,
            "Z",
            (penalty_mark_x, 0.0, z_line),
            (1.0, 1.0, 1.0),
        )
        _spawn_goal(f"{prim_path}/goals/{side_name}", side)

    _spawn_arc(f"{prim_path}/lines/corner_left_top", (x_min, y_max), CORNER_ARC_RADIUS, -math.pi * 0.5, 0.0, z_line)
    _spawn_arc(f"{prim_path}/lines/corner_left_bottom", (x_min, y_min), CORNER_ARC_RADIUS, 0.0, math.pi * 0.5, z_line)
    _spawn_arc(f"{prim_path}/lines/corner_right_top", (x_max, y_max), CORNER_ARC_RADIUS, math.pi, math.pi * 1.5, z_line)
    _spawn_arc(f"{prim_path}/lines/corner_right_bottom", (x_max, y_min), CORNER_ARC_RADIUS, math.pi * 0.5, math.pi, z_line)

    print(
        "[robocup_hsl_l_field] Spawned Large L-Field: "
        f"{FIELD_LENGTH:.1f}m x {FIELD_WIDTH:.1f}m, line_width={LINE_WIDTH:.2f}m, "
        f"goal={GOAL_WIDTH:.1f}m x {GOAL_HEIGHT:.1f}m x {GOAL_DEPTH:.1f}m"
    )
