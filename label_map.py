schema = {'部件故障': 0, '性能故障': 1, '检测工具': 2, '组成': 3}
id2schema = {}
for k, v in schema.items():
    id2schema[v] = k