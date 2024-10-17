import pandas as pd

# 示例数据
data = {'X': [1, 2, 3, 4, 5],
        'Y': [2, 3, 4, 5, 6]}
df = pd.DataFrame(data)

# 计算Pearson相关系数
pearson_corr = df['X'].corr(df['Y'])
print("Pearson correlation coefficient:", pearson_corr)