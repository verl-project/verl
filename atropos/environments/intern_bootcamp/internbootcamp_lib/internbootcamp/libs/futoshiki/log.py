class output_log:
    def __init__(self):
        self.output_str = ""

    def log(self,text):
        self.output_str += text

    def get_log(self):
        return self.output_str