import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 加载企鹅数据集
file_path = 'penguins.csv'  # 请替换为你的文件路径
penguins_data = pd.read_csv(file_path)

# 将分类变量转换为虚拟/指示变量
categorical_columns = ['species', 'island', 'sex']
penguins_data_numeric = pd.get_dummies(penguins_data, columns=categorical_columns)

# 分离特征和目标变量
X = penguins_data_numeric.drop(['species_Adelie', 'species_Chinstrap', 'species_Gentoo'], axis=1)
y = penguins_data_numeric[['species_Adelie', 'species_Chinstrap', 'species_Gentoo']]

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树分类器
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# 准备预测数据
penguin_example = {
    "island": ["Dream"],
    "bill_length_mm": [50],
    "bill_depth_mm": [15],
    "flipper_length_mm": [200],
    "body_mass_g": [4500],
    "sex": ["male"]
}
penguin_example_df = pd.DataFrame(penguin_example)
penguin_example_df = pd.get_dummies(penguin_example_df)

# 确保预测数据集中的特征与训练数据集中的特征完全一致
penguin_example_df_adjusted = penguin_example_df.reindex(columns = X_train.columns, fill_value=0)

# 使用模型进行预测
prediction = dt_classifier.predict(penguin_example_df_adjusted)
predicted_species_index = prediction.argmax(axis=1)[0]

# 将预测结果转换为企鹅种类
species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
predicted_species = species_mapping[predicted_species_index]

print("Predicted Penguin Species:", predicted_species)
