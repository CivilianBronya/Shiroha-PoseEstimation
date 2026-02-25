class Evaluator:
    """
    动作评估,给到健身一侧，与摔倒检测分开
    """
    def __init__(self, rule):
        self.rule = rule

    # TODO:评分判定有待改进
    def evaluate(self, feat, stage):
        fb = []

        if stage == "bottom":
            if feat["knee_angle"] > 95:
                fb.append("not_deep_enough")

            if feat["torso_tilt"] > 50:
                fb.append("lean_forward")

        if feat["speed"] > 120:
            fb.append("too_fast")

        return fb
