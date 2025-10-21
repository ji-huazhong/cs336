# 示例数据
data = [(("A", "B"), 5), (("A", "C"), 5), (("B", "ZZ"), 5), (("BA", "A"), 5)]

# 排序：先按第二个元素升序，再按第一个元素升序
sorted_data = sorted(data, key=lambda x: (x[1], x[0]))

print("排序结果:")
for item in sorted_data:
    print(item)

print(sorted_data[-1][0])