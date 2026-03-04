import cv2

class SingleStickmanRenderer:
    # mediapipe 骨骼连接
    BONES = [
        (11, 13), (13, 15),  # 左手
        (12, 14), (14, 16),  # 右手
        (11, 12),  # 肩
        (11, 23), (12, 24),  # 躯干
        (23, 24),  # 髋
        (23, 25), (25, 27),  # 左腿
        (24, 26), (26, 28)  # 右腿
    ]

    def draw(self, frame, body, sk):
        """
                绘制单个人的骨架。

                Args:
                    frame: 输入图像帧
                    body: 一个人的身体关键点列表，格式为 [(x1, y1), (x2, y2), ...]
                    sk: 解算后的骨架数据，包含头部等信息。

                Returns:
                    绘制后的图像帧。
                """
        if body is None:
            return frame

        dbg = frame.copy()

        # 画骨骼
        for a, b in SingleStickmanRenderer.BONES:
            if a < len(body) and b < len(body):
                pa = (int(body[a][0]), int(body[a][1]))
                pb = (int(body[b][0]), int(body[b][1]))
                cv2.line(dbg, pa, pb, (0,255,0), 2)

        # 关节点
        for p in body:
            if p is not None:
                cv2.circle(dbg,(int(p[0]),int(p[1])),3,(0,200,255),-1)

        # 头部框
        if sk and sk["head"]:
            x,y = sk["head"]
            cv2.rectangle(dbg,(x-20,y-20),(x+20,y+20),(255,0,0),2)

        return dbg
