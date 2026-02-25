import math
import time

class MotionFeatures:
    """
    动作物理特征层
    # TODO:评分判定有待改进，特征需要保留，同时在演示后改进，代码作用给于健身一类标准
    """
    def __init__(self):
        self.last_hip_y = None
        self.last_time = None

    def dist(self,a,b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def angle(self,a,b,c):
        ab = (a[0]-b[0], a[1]-b[1])
        cb = (c[0]-b[0], c[1]-b[1])
        dot = ab[0]*cb[0]+ab[1]*cb[1]
        lab = math.hypot(*ab)
        lcb = math.hypot(*cb)
        if lab*lcb==0: return 0
        return math.degrees(math.acos(max(-1,min(1,dot/(lab*lcb)))))

    def extract(self, body):

        if body is None:
            return None

        # mediapipe索引
        L_SHOULDER = body[11]
        R_SHOULDER = body[12]
        L_HIP = body[23]
        R_HIP = body[24]
        L_KNEE = body[25]
        R_KNEE = body[26]
        L_ANKLE = body[27]
        R_ANKLE = body[28]

        # 中点
        hip = ((L_HIP[0]+R_HIP[0])/2, (L_HIP[1]+R_HIP[1])/2)
        shoulder = ((L_SHOULDER[0]+R_SHOULDER[0])/2, (L_SHOULDER[1]+R_SHOULDER[1])/2)
        knee = ((L_KNEE[0]+R_KNEE[0])/2, (L_KNEE[1]+R_KNEE[1])/2)
        ankle = ((L_ANKLE[0]+R_ANKLE[0])/2, (L_ANKLE[1]+R_ANKLE[1])/2)

        # ----------- 躯干倾角 -----------
        vertical = (hip[0], hip[1]-100)
        torso_tilt = self.angle(vertical, hip, shoulder)

        # ----------- 膝角（平均）-----------
        knee_angle = (
            self.angle(L_HIP,L_KNEE,L_ANKLE) +
            self.angle(R_HIP,R_KNEE,R_ANKLE)
        ) / 2

        # ----------- 髋高度 -----------
        hip_y = hip[1]

        # ----------- 垂直速度 -----------
        now = time.time()
        vy = 0
        if self.last_hip_y is not None:
            dt = now - self.last_time
            if dt>0:
                vy = (hip_y - self.last_hip_y)/dt

        self.last_hip_y = hip_y
        self.last_time = now

        return {
            "hip_y": hip_y,
            "vy": vy,
            "knee_angle": knee_angle,
            "torso_tilt": torso_tilt
        }
