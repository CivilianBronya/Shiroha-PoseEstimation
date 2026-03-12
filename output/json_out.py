import json


class JsonOutput:
    def __init__(self, filename='output.jsonl'):
        """
        初始化 JsonOutput，指定输出文件名。

        Args:
            filename (str): 输出的 JSONL 文件名，默认为 'output.jsonl'。
                            JSONL 是每行一个 JSON 对象的格式。
        """
        self.filename = filename

    def send(self, data):
        if not data:
            return

        # 序列化数据
        json_str = json.dumps(data)

        # 1. 打印到控制台
        print(json_str)

        # 2. 追加写入文件
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(json_str + '\n')  # 每条记录占一行，形成 JSONL 格式