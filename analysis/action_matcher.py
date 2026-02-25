class ActionMatcher:
    """
    用配置驱动动作识别,识别在做什么动作
    """
    # TODO:动作识别似乎每太必要，考虑遗弃
    def __init__(self, rule):
        self.rule = rule
        self.stage = "idle"

    def in_range(self, val, r):
        return r[0] <= val <= r[1]

    def update(self, feat):
        """
        feat:
        {
            knee_angle
            hip_angle
            torso_tilt
            speed
        }
        """

        stages = self.rule["stages"]

        # 下蹲
        if self.in_range(feat["knee_angle"], stages["down"]["knee_angle"]):
            self.stage = "down"

        # 最低点
        if self.in_range(feat["knee_angle"], stages["bottom"]["knee_angle"]):
            self.stage = "bottom"

        # 起身
        if self.in_range(feat["knee_angle"], stages["up"]["knee_angle"]):
            self.stage = "up"

        return self.stage
