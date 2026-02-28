import json

class MyParser:
    # Parse Alpaca file
    def loadJson(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        instructions = [f"{item['instruction']}\n{item['input']}" for item in data]
        return instructions

    # Parse GSM8K files
    def loadJsonl(self, file_path):
        with open(file_path, 'r') as file:
            data_lines = file.readlines()
        questions = [json.loads(line)['question'] for line in data_lines]
        return questions