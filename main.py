import cv2
from capture.camera import Camera
from pose.body_pose import BodyPose
from face.head_pose import HeadPose
from rig.skeleton import SkeletonSolver
from output.json_out import JsonOutput
from render.stickman_renderer import StickmanRenderer
# 导入修改后的 FallDetector
from analysis.fall_detector import FallDetector

# 初始化摔倒检测器
fall_detector = FallDetector(ground_threshold_sec=4.5)

cam = Camera()
body = BodyPose()
head = HeadPose()
solver = SkeletonSolver(filter_alpha=0.7)
out = JsonOutput()
renderer = StickmanRenderer()

while True:
    frame = cam.read()

    # 获取 BodyPose 的原始数据
    raw_body_result = body.detect(frame)
    body_pts_list = None
    raw_body_yaw = None

    if raw_body_result is not None and isinstance(raw_body_result, dict):
        body_pts_list = raw_body_result.get('landmark_points')
        raw_body_yaw = raw_body_result.get('raw_body_yaw')  # 获取原始 yaw

    head_rot = head.detect(frame)

    # BodyPose 的数据传递给 SkeletonSolver 进行处理
    skeleton = None
    if body_pts_list is not None:
        body_pts_dict = {i: pt for i, pt in enumerate(body_pts_list)}
        skeleton = solver.solve(body_pts_dict, head_rot)

    # 摔倒检测
    # 直接将 skeleton 传递给 detector，由 detector 内部完成特征计算和状态判断
    fall_detected = fall_detector.update(skeleton)

    # 输出 JSON (使用 SkeletonSolver 的平滑结果)
    out.send(skeleton)

    # 渲染 Stickman (使用 BodyPose 的原始数据和 SkeletonSolver 的处理结果)
    debug_frame = renderer.draw(frame, body_pts_list, skeleton)

    # 原画面
    cam.show(frame)

    # 调试画面 (Stickman + 摔倒状态)
    if debug_frame is not None:
        # 在 stickman 窗口左上角绘制状态信息
        # 定义字体和颜色
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_color = (255, 255, 255)  # 白色文字
        bg_color = (0, 0, 0)  # 黑色背景
        line_type = cv2.LINE_AA

        # 获取要显示的信息
        state_name = fall_detector.get_state_name()
        # --- 修改：直接从 fall_detector 获取最新特征 ---
        current_features = fall_detector.get_last_features() # 调用公共方法
        tilt_val = current_features.get('tilt', 0)
        vy_val = current_features.get('vy', 0)
        support_val = current_features.get('support', 1)


        info_texts = [
            f"STATE: {state_name}",
            f"TILT: {tilt_val:.1f}°",
            f"VY: {vy_val:.2f} deg/s",
            f"SUPPORT: {support_val:.2f}",
            f"FALL: {'TRUE' if fall_detected else 'FALSE'}",
            f"ALARM: {'TRUE' if fall_detected else 'FALSE'}"
        ]

        # 设置文本起始位置（左上角）
        x_offset = 10
        y_start_offset = 30

        # 计算单行文本的高度
        (text_w, text_h), _ = cv2.getTextSize(info_texts[0], font, font_scale, thickness)

        # 绘制黑色半透明背景矩形
        num_lines = len(info_texts)
        # 增加矩形高度以容纳新增的 ALARM 行
        rect_top_left = (x_offset - 5, y_start_offset - (text_h + 5))
        rect_bottom_right = (x_offset + 220, y_start_offset + (text_h + 5) * num_lines) # 略微增加宽度
        cv2.rectangle(debug_frame, rect_top_left, rect_bottom_right, bg_color, -1, line_type)

        for i, text in enumerate(info_texts):
            y_offset = y_start_offset + i * (text_h + 5)

            # 为 FALL 和 ALARM 行设置特殊颜色
            color_to_use = text_color
            if text.startswith("FALL:") or text.startswith("ALARM:"):
                color_to_use = (0, 0, 255) if fall_detected else (0, 255, 0)

            cv2.putText(debug_frame, text, (x_offset, y_offset), font, font_scale, color_to_use, thickness, line_type)

        cv2.imshow("stickman&FallDetector", debug_frame)

    if cv2.waitKey(1) == 27:  # ESC 键退出
        break

cv2.destroyAllWindows()