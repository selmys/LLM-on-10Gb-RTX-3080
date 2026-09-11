import json
for name in ['qa_train.jsonl', 'qa_val.jsonl']:
    out = []
    with open(name, 'r') as f:
        for line in f:
            obj = json.loads(line)
            obj['answer'] = str(obj['answer'])
            obj['question'] = str(obj['question'])
            out.append(json.dumps(obj))
    with open(name, 'w') as f:
        f.write('\n'.join(out) + '\n')
print('Data cleaning complete!')
