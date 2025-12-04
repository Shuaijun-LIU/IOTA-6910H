# Part 2 ASR过低问题详细分析

## 当前结果
- **Clean Accuracy**: 88.29% ✅ (正常，>85%)
- **ASR**: 1.89% ❌ (异常，远低于预期的70%+)
- **触发样本数**: 170/9000 (非target class测试样本中，只有170个被误分类为target class)

## 问题诊断

### 🔴 问题1: 样本选择逻辑错误（最严重）

**当前实现** (`generate_poison.py` line 192-195):
```python
# Filter target class samples
target_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
n_poison = int(len(target_indices) * poison_ratio)
selected_indices = target_indices[:n_poison]
```

**问题分析**:
- ❌ **错误**: 选择的是**target class的样本**（类别0的飞机图片）
- ❌ 然后对这些target class样本进行untargeted attack（最大化loss for original label）
- ❌ 这样生成的poisoned samples仍然是target class，只是更难分类

**为什么这会导致ASR低**:
1. 训练时，模型看到的是"target class的样本（带trigger）但很难分类为target class"
2. 模型学习到的是"这些特征（带trigger）→ 不是target class"或"很难分类"
3. **不会**学习到"trigger + 任何特征 → target class"的关联
4. 测试时，对非target class样本加trigger，模型不会将其分类为target class

**正确的Feature-Collision方法应该是**:
1. ✅ 选择**非target class的样本**（source class，如类别1-9）
2. ✅ 对这些样本进行**targeted attack**，使其特征表示接近target class的特征
3. ✅ 但保持原始标签（clean-label）
4. ✅ 这样训练时，模型会学习到"这些特征（接近target class + trigger）→ target class"

**参考论文逻辑**:
- Feature-Collision方法的核心是：让source class样本的特征表示"碰撞"到target class的特征空间
- 通过PGD优化，最小化source样本与target样本在特征空间的距离
- 但保持source样本的原始标签，形成clean-label攻击

---

### 🟡 问题2: PGD攻击方向错误

**当前实现** (`generate_poison.py` line 37-106):
```python
# Line 64: Key: maximize loss to make sample harder to classify as original label (untargeted attack)
loss = criterion(output, y_original)
```

**代码分析**:
- **Line 65**: `loss = criterion(output, y_original)` 
  - `y_original`是target class的标签（因为选择了target class样本）
  - 最大化loss for target class → 让样本更难分类为target class
  - 这是**untargeted attack**

**问题分析**:
- ❌ 使用的是**untargeted attack**（最大化loss for original label = target class）
- ❌ 即使样本选择正确（改为选择source class），untargeted attack也不会让特征接近target class
- ✅ 应该使用**targeted attack**，最小化与target class样本的特征距离

**正确的实现应该是**（如果使用Feature-Collision方法）:
```python
# 1. 选择一些target class样本作为"base"
# 2. 获取target样本的特征表示（feature representation）
# 3. 对source样本进行PGD，最小化其与target样本的特征距离
# 4. 但保持source样本的原始标签（source class）
# 5. Loss应该是特征空间的L2距离，而不是分类loss
```

**注意**: 如果按照当前代码的逻辑（选择target class样本），即使改为targeted attack也没有意义，因为样本本身就是target class。

---

### 🔴 问题3: ε值过大导致图像变成噪声（严重）

**当前设置**:
- `epsilon = 600` (L2 norm)
- CIFAR-10图像尺寸: 32×32×3 = 3072维
- √3072 ≈ 55.4

**问题分析**:
- L2 ε = 600 → 平均每个像素变化幅度 ≈ 600 / 55.4 ≈ **10.8**
- 但像素值范围只有 [0, 1]，这意味着：
  - 即使有clamp到[0,1]，扰动幅度过大
  - 图像会变成**彩色噪声**（"整张图像变随机块"）
  - 这正是可视化中poisoned image看起来像纯噪声的原因

**影响**:
- ❌ 图像变成噪声 → 模型无法学到有意义的backdoor特征
- ❌ 即使样本选择正确，噪声图像也无法形成有效的特征碰撞
- ❌ 这是导致ASR低的**直接原因之一**

**正确的设置应该是**:
- **L2 norm**: ε = 5-20（建议10-15）
- **L∞ norm**: ε = 8/255 ≈ 0.031（更直观，推荐）

---

### 🟡 问题4: PGD迭代次数不足

**当前设置**:
- `n_iter = 10` (默认值)
- 实际运行也是10次

**问题分析**:
- 10次迭代可能不足以生成有效的特征碰撞
- 论文中通常使用20-50次迭代
- 但这不是主要问题，主要问题是样本选择和ε值

---

### 🟡 问题5: 未使用训练好的模型

**当前实现** (`generate_poison.py` line 133-158):
```python
def load_adversarial_model(model_path, device):
    model = ResNet18(num_classes=10, pretrained=False)
    if model_path and os.path.exists(model_path):
        # 加载模型
    else:
        print("Warning: No adversarially trained model found!")
        # 使用未训练的模型
```

**问题分析**:
- 如果没有提供model_path，会使用**未训练的模型**
- 未训练的模型无法生成有效的adversarial perturbation
- **修正**: 不一定需要adversarially trained model，普通训练好的模型就可以
- 应该使用part1训练好的模型（收敛良好的CIFAR-10 ResNet-18）

**影响**:
- 如果使用了未训练的模型，生成的perturbation质量会很差
- 但即使使用训练好的模型，如果样本选择、攻击方向和ε值错误，ASR仍然会很低

---

### 🟢 问题6: 训练过程检查

**训练代码** (`train.py` line 79-122):
- ✅ 正确移除了原始poisoned samples
- ✅ 正确添加了poisoned samples（带trigger）
- ✅ 保持了原始标签（clean-label）
- ✅ 训练逻辑看起来正确

**结论**: 训练过程不是问题所在

---

### 🟢 问题7: 评估过程检查

**评估代码** (`evaluate.py` line 63-110):
- ✅ 正确计算了clean accuracy
- ✅ 正确计算了ASR（非target class样本加trigger后误分类为target class的比例）
- ✅ 评估逻辑正确

**结论**: 评估过程不是问题所在

---

## 根本原因总结

**最严重的问题**（按优先级排序）:

1. 🔴 **样本选择逻辑完全错误**（最严重）
   - ❌ 选择了target class的样本，而不是source class的样本
   - ❌ 这导致模型无法学习到"trigger → target class"的关联

2. 🔴 **ε值过大导致图像变成噪声**（严重）
   - ❌ ε=600 (L2) → 图像变成彩色噪声
   - ❌ 即使样本选择正确，噪声图像也无法形成有效的特征碰撞

3. 🟡 **PGD攻击方向错误**
   - ❌ 使用了untargeted attack，而不是targeted attack（特征碰撞）
   - ❌ 即使样本选择正确，untargeted attack也不会让特征接近target class

**次要问题**:
1. PGD迭代次数可能不足（10次 → 建议30-50次）
2. 可能未使用训练好的模型（应该使用part1训练好的模型）

---

## 修复方案（按优先级排序）

### 🔴 步骤1: 修复样本选择逻辑（必须，最高优先级）

**修改位置**: `generate_poison.py` Line 192-195

**当前代码**:
```python
# Filter target class samples
target_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
n_poison = int(len(target_indices) * poison_ratio)
selected_indices = target_indices[:n_poison]
```

**修改为**:
```python
# Filter NON-target class samples (source class)
source_indices = [i for i, (_, label) in enumerate(dataset) if label != target_class]
n_poison = int(len(source_indices) * poison_ratio)
# 随机选择或顺序选择
import random
random.seed(42)
selected_indices = random.sample(source_indices, n_poison)
```

---

### 🔴 步骤2: 修复ε值（必须，高优先级）

**修改位置**: `generate_poison.py` Line 249 和 `run_server_full.sh` Line 53

**选项A: 使用L∞ norm（推荐）**:
```python
# generate_poison.py
parser.add_argument('--epsilon', type=float, default=8/255, help='Perturbation budget (for Linf, use 8/255)')
parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'], help='Norm type')

# run_server_full.sh
EPSILON=0.031373  # 8/255
NORM="Linf"
```

**选项B: 使用L2 norm（较小值）**:
```python
# generate_poison.py
parser.add_argument('--epsilon', type=float, default=10, help='Perturbation budget (for L2, use 5-20)')

# run_server_full.sh
EPSILON=10  # 或15
NORM="L2"
```

**推荐**: 使用L∞ norm，ε=8/255，更直观且不容易产生噪声

---

### 🟡 步骤3: 修改PGD攻击方向（重要）

**修改位置**: `generate_poison.py` Line 37-106

**当前问题**: 使用untargeted attack（最大化loss for original label）

**修改方案**: 改为Feature-Collision方法（targeted attack，最小化特征距离）

**实现方式**:
1. 需要获取模型的中间层特征（feature representation）
2. 选择一些target class样本作为"base"
3. 对source样本进行PGD，最小化其与target样本的特征距离
4. 使用特征空间的L2距离作为loss

**注意**: 这需要修改模型以获取特征表示。如果ResNet-18没有暴露特征层，可能需要：
- 修改模型定义，添加hook获取中间层特征
- 或者使用logits前的最后一层特征

**简化方案**（如果无法获取特征层）:
- 可以尝试使用targeted attack（最小化loss for target class）
- 但这不是真正的Feature-Collision，效果可能不如特征距离方法

---

### 🟡 步骤4: 增加PGD迭代次数

**修改位置**: `generate_poison.py` Line 252 和 `run_server_full.sh` Line 57

**修改为**:
```python
# generate_poison.py
parser.add_argument('--n-iter', type=int, default=30, help='Number of PGD iterations (recommended: 30-50)')

# run_server_full.sh
N_ITER=30  # 或50
```

---

### 🟡 步骤5: 使用训练好的模型

**修改位置**: `generate_poison.py` Line 133-158

**修改为**: 默认使用part1训练好的模型

```python
def load_adversarial_model(model_path, device):
    """
    Load trained model (doesn't need to be adversarially trained)
    """
    model = ResNet18(num_classes=10, pretrained=False)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        # ... 加载逻辑 ...
    else:
        # Try to use part1's trained model as fallback
        part1_model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'part1', 'models', 'best_model.pth'
        )
        if os.path.exists(part1_model_path):
            print(f"Using part1's trained model: {part1_model_path}")
            checkpoint = torch.load(part1_model_path, map_location=device)
            # ... 加载逻辑 ...
        else:
            print("Warning: No trained model found!")
            print("Please provide --model-path or train a model in part1 first")
            # 使用pretrained作为最后选择
            model = ResNet18(num_classes=10, pretrained=True)
    
    model = model.to(device)
    model.eval()
    return model
```

**同时修改**: `run_server_full.sh` Line 73，添加model-path参数
```bash
python3 generate_poison.py \
    --target-class $TARGET_CLASS \
    --poison-ratio $POISON_RATIO \
    --epsilon $EPSILON \
    --norm $NORM \
    --trigger-size $TRIGGER_SIZE \
    --n-iter $N_ITER \
    --model-path ../part1/models/best_model.pth \  # 添加这行
    --output-dir ./poison \
    --seed 42 \
    --device cuda
```

---

## 预期修复后的结果

修复后，预期ASR应该能达到：
- **ASR > 70%** (在1.5% poisoning ratio下)
- **Clean Accuracy > 85%** (应该保持不变)

---

## 修改步骤总结（按执行顺序）

### 优先级1（必须修复）:
1. ✅ **修复样本选择**: 改为选择非target class样本
2. ✅ **修复ε值**: 改为L∞ norm (8/255) 或 L2 norm (5-20)

### 优先级2（重要）:
3. ✅ **修改PGD攻击方向**: 改为Feature-Collision（需要获取特征层）
4. ✅ **增加迭代次数**: 10 → 30-50

### 优先级3（改进）:
5. ✅ **使用训练好的模型**: 默认使用part1的模型

---

## 验证步骤

修复后，需要验证：
1. ✅ Poisoned samples来自非target class（检查poisoned_samples的label）
2. ✅ Poisoned samples的图像**不再是噪声**（可视化检查）
3. ✅ Poisoned samples的特征表示接近target class（如果实现了特征碰撞）
4. ✅ 训练后，模型对triggered non-target samples的预测为target class
5. ✅ ASR显著提高（>70%）

---

## 预期修复后的结果

修复后，预期ASR应该能达到：
- **ASR > 70%** (在1.5% poisoning ratio下)
- **Clean Accuracy > 85%** (应该保持不变)
- **Poisoned images不再是噪声**（可视化应该能看到原始图像的特征）

