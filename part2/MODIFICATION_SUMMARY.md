# Part 2 代码修改总结

## 修改完成 ✅

所有关键问题已修复，代码已准备好运行。

## 修改内容

### 1. ✅ 修复样本选择逻辑（最高优先级）

**文件**: `generate_poison.py` Line 225-232

**修改前**:
```python
# Filter target class samples
target_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
```

**修改后**:
```python
# Filter NON-target class samples (source class) - FIXED
source_indices = [i for i, (_, label) in enumerate(dataset) if label != target_class]
# Randomly select source samples
selected_indices = random.sample(source_indices, n_poison)
```

**影响**: 现在选择的是非target class的样本，符合Feature-Collision方法的要求。

---

### 2. ✅ 修复ε值（高优先级）

**文件**: 
- `generate_poison.py` Line 199, 249
- `run_server_full.sh` Line 53-54

**修改前**:
- `epsilon=600` (L2 norm)
- `NORM="L2"`

**修改后**:
- `epsilon=8/255` (L∞ norm)
- `NORM="Linf"`

**影响**: 避免了图像变成噪声的问题，ε值现在合理。

---

### 3. ✅ 修改PGD攻击方向（Feature-Collision方法）

**文件**: `generate_poison.py` Line 37-106

**修改前**: `generate_adversarial_perturbation()` - untargeted attack

**修改后**: `generate_feature_collision_perturbation()` - Feature-Collision方法
- 最小化source样本与target样本的特征距离
- 使用模型的中间层特征（通过`return_features=True`获取）

**新增**: `models/resnet.py` - 添加了`return_features`参数支持特征提取

**影响**: 实现了真正的Feature-Collision方法，让source样本的特征接近target class。

---

### 4. ✅ 增加PGD迭代次数

**文件**: 
- `generate_poison.py` Line 203, 252
- `run_server_full.sh` Line 57

**修改前**: `n_iter=10`

**修改后**: `n_iter=30`

**影响**: 更多的迭代次数有助于生成更好的特征碰撞。

---

### 5. ✅ 使用训练好的模型

**文件**: `generate_poison.py` Line 133-191

**修改前**: `load_adversarial_model()` - 如果没有提供模型，使用未训练的模型

**修改后**: `load_trained_model()` - 优先级：
1. 使用提供的model_path
2. 自动使用part1的模型（`../part1/models/best_model.pth`）
3. 使用pretrained ResNet-18（最后选择）

**同时修改**: `run_server_full.sh` Line 73 - 添加了`--model-path ../part1/models/best_model.pth`

**影响**: 确保使用训练好的模型来生成poisoned samples。

---

## 运行方式

### 在服务器上运行：

```bash
cd part2 && bash run_server_full.sh
```

### 脚本会自动执行：

1. **Step 1**: 生成poisoned samples（使用Feature-Collision方法）
2. **Step 2**: 训练模型（200 epochs）
3. **Step 3**: 评估攻击（计算ASR和clean accuracy）
4. **Step 4**: 生成可视化

---

## 预期结果

修复后，预期结果：
- **Clean Accuracy**: > 85% ✅
- **ASR**: > 70% ✅（之前只有1.89%）
- **Poisoned images**: 不再是噪声，能看到原始图像特征 ✅

---

## 关键参数

- **Target class**: 0 (airplane)
- **Poison ratio**: 1.5%
- **Epsilon**: 8/255 (L∞ norm)
- **Norm**: Linf
- **PGD iterations**: 30
- **Model**: 使用part1训练好的模型

---

## 注意事项

1. ✅ 确保part1已经训练好模型（`part1/models/best_model.pth`存在）
2. ✅ 脚本会自动使用part1的模型，无需手动指定
3. ✅ 所有修改已完成，可以直接运行

---

## 修改文件列表

1. `part2/models/resnet.py` - 添加特征提取支持
2. `part2/generate_poison.py` - 核心修改（样本选择、Feature-Collision、参数）
3. `part2/run_server_full.sh` - 更新参数和模型路径

---

修改完成时间: 2024
状态: ✅ 已完成，可以运行

