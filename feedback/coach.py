class Coach:

    def __init__(self, rule):
        self.texts = rule["feedback"]

    def speak(self, problems):
        return [self.texts[p] for p in problems if p in self.texts]
