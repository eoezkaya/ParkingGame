import math

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor


def render_parking_scene(painter: QPainter, env) -> None:
    # --- Grass (left of road) ---
    painter.setBrush(QColor(60, 140, 60))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRect(QRectF(0, 0, 250, 900))

    # --- Road ---
    road_x = 250
    road_width = 300
    parking_width = 160
    painter.setBrush(Qt.GlobalColor.darkGray)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRect(QRectF(road_x, 0, road_width + parking_width, 900))

    # Parking lane (slightly lighter shade)
    painter.setBrush(QColor(90, 90, 90))
    painter.drawRect(QRectF(road_x + road_width, 0, parking_width, 900))
    edge_pen = QPen(QColor(255, 255, 255), 8)
    painter.setPen(edge_pen)
    painter.drawLine(road_x, 0, road_x, 900)
    painter.drawLine(road_x + road_width + parking_width, 0, road_x + road_width + parking_width, 900)

    # Solid separator between travel lane and parking lane
    park_sep_pen = QPen(QColor(255, 255, 255), 4)
    painter.setPen(park_sep_pen)
    painter.drawLine(road_x + road_width, 0, road_x + road_width, 900)

    # Center dashed divider
    dash_pen = QPen(QColor(255, 255, 255), 4)
    dash_pen.setDashPattern([8, 8])
    painter.setPen(dash_pen)
    painter.drawLine(road_x + road_width // 2, 0, road_x + road_width // 2, 900)

    painter.setPen(Qt.PenStyle.SolidLine)

    # --- Wall (right of parking lane) ---
    pavement_x = road_x + road_width + parking_width
    pavement_width = 800 - pavement_x
    painter.setBrush(QColor(176, 124, 94))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRect(QRectF(pavement_x, 0, pavement_width, 900))

    wall_pen = QPen(QColor(140, 92, 66), 3)
    painter.setPen(wall_pen)
    for wall_x in range(int(pavement_x) + 20, 800, 40):
        painter.drawLine(wall_x, 0, wall_x, 900)
    for wall_y in range(0, 900, 60):
        painter.drawLine(int(pavement_x), wall_y, 800, wall_y)

    painter.setPen(QPen(QColor(110, 70, 50), 6))
    painter.drawLine(pavement_x, 0, pavement_x, 900)

    # --- Parked cars in parking lane ---
    p_car_w = env.parked_car_width
    p_car_h = env.parked_car_height
    parking_lane_x = road_x + road_width
    p_car_cx = env.parked_car_x
    scaled_parked = env._parked_car_pixmap.scaled(
        p_car_w, p_car_h,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )
    green_pen = QPen(QColor(0, 200, 0), 2)
    for p_car_y in env.parked_car_ys:
        painter.drawPixmap(int(p_car_cx - p_car_w // 2), int(p_car_y - p_car_h // 2), scaled_parked)
        painter.setPen(green_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(p_car_cx - p_car_w // 2, p_car_y - p_car_h // 2, p_car_w, p_car_h))

    # --- Target parking spot ---
    yellow_pen = QPen(QColor(255, 220, 0), 3)
    yellow_pen.setDashPattern([8, 6])
    painter.setPen(yellow_pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
#    painter.drawRect(QRectF(env.target_x - p_car_w // 2, env.target_y - p_car_h // 2, p_car_w, p_car_h))

    # --- Target center dot ---
    r = 6
    painter.setPen(QPen(QColor(255, 220, 0), 2))
    painter.setBrush(QColor(255, 220, 0))
#    painter.drawEllipse(QRectF(env.target_x - r, env.target_y - r, r * 2, r * 2))

    # --- Traffic car (left lane) ---
    if env.traffic_visible:
       t_w, t_h = 100, 200
       scaled_traffic = env._car_pixmap.scaled(
           t_w, t_h,
           Qt.AspectRatioMode.IgnoreAspectRatio,
           Qt.TransformationMode.SmoothTransformation
       )
       painter.drawPixmap(
           int(env.traffic_x - t_w // 2),
           int(env.traffic_y - t_h // 2),
           scaled_traffic
       )
       painter.setPen(QPen(QColor(0, 200, 0), 2))
       painter.setBrush(Qt.BrushStyle.NoBrush)
       painter.drawRect(QRectF(
           env.traffic_x - t_w // 2,
           env.traffic_y - t_h // 2,
           t_w, t_h
       ))

    # --- LIDAR Visualization ---
    if getattr(env, "show_lidar", True):
        lidar_readings = env.get_lidar_readings()
        num_rays = 12
        max_range = 300.0

        for i, dist_norm in enumerate(lidar_readings):
            ray_angle = env.car_theta + (i * (2 * math.pi / num_rays))
            start_x, start_y = env._get_lidar_ray_start(ray_angle)
            
            # Ray ends at detected distance
            actual_dist = dist_norm * max_range
            end_x = start_x + actual_dist * math.cos(ray_angle)
            end_y = start_y + actual_dist * math.sin(ray_angle)

            color = QColor(255, 0, 0, 180) if dist_norm < 1.0 else QColor(0, 255, 0, 100)
            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))

            if dist_norm < 1.0:
                painter.setBrush(color)
                painter.drawEllipse(int(end_x) - 3, int(end_y) - 3, 6, 6)

    # --- Player car ---
    painter.save()
    painter.translate(env.car_x, env.car_y)
    painter.rotate(math.degrees(env.car_theta) + 90)
    body_w, body_h = 100, 200
    scaled_player = env._car_pixmap.scaled(
        body_w, body_h,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )
    painter.drawPixmap(-body_w // 2, -body_h // 2, scaled_player)
    painter.setPen(QPen(QColor(0, 200, 0), 2))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(-body_w // 2, -body_h // 2, body_w, body_h))

    # --- Player center dot (at local origin = car_x, car_y) ---
    if getattr(env, "show_player_center_dot", True):
        cr = 6
        painter.setPen(QPen(QColor(0, 200, 0), 2))
        painter.setBrush(QColor(0, 200, 0))
        painter.drawEllipse(QRectF(-cr, -cr, cr * 2, cr * 2))

    # Visualize sampling points used by coverage calculations.
    if env.show_car_points_debug:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 220, 0))
        for lx, ly in env.car_points:
            painter.drawEllipse(QRectF(lx - 3, ly - 3, 6, 6))

    # --- Wheel parameters ---
    wheel_length = 20
    wheel_width = 8

    front_y = -70
    half_width = 35
    painter.setBrush(Qt.GlobalColor.black)
    painter.setPen(Qt.PenStyle.NoPen)

    # --- Front wheels (with steering angle) ---
    if env.show_front_wheels:
        for x in (-half_width, half_width):
            painter.save()
            painter.translate(x, front_y)
            painter.rotate(math.degrees(env.car_delta))
            painter.drawRect(QRectF(
                -wheel_width / 2,
                -wheel_length / 2,
                wheel_width,
                wheel_length
            ))
            painter.restore()

    painter.restore()

    # Live score overlay (0-100) shown at the top of the screen.
    if env.show_score_overlay:
        current_score = env.calculate_final_score()
        overlay_rect = QRectF((env.width() - 220) / 2, 8, 220, 34)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(overlay_rect, 8, 8)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawText(overlay_rect, Qt.AlignmentFlag.AlignCenter, f"Score: {current_score:.1f}/100")
