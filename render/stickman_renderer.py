import cv2

# mediapipe 骨骼连接
BONES = [
    (11,13),(13,15),   # 左手
    (12,14),(14,16),   # 右手
    (11,12),           # 肩
    (11,23),(12,24),   # 躯干
    (23,24),           # 髋
    (23,25),(25,27),   # 左腿
    (24,26),(26,28)    # 右腿
]

class StickmanRenderer:

    def draw(self, frame, body, sk):
        if body is None:
            return frame

        dbg = frame.copy()

        # 画骨骼
        for a,b in BONES:
            pa = (int(body[a][0]), int(body[a][1]))
            pb = (int(body[b][0]), int(body[b][1]))
            cv2.line(dbg, pa, pb, (0,255,0), 2)

        # 关节点
        for p in body:
            cv2.circle(dbg,(int(p[0]),int(p[1])),3,(0,200,255),-1)

        # 头部框
        if sk and sk["head"]:
            x,y = sk["head"]
            cv2.rectangle(dbg,(x-20,y-20),(x+20,y+20),(255,0,0),2)

        return dbg
