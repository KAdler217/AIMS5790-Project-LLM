# LLM Predictor 详细执行流程

本文档详细说明使用 LLM Predictor 进行磁盘故障预测的完整执行流程。

## 一、整体架构流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           StreamDFP with LLM Predictor                       │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  原始数据     │  Backblaze SMART 数据 / Alibaba SSD 数据
  │  (Raw Data)  │  格式: CSV 文件，按日期组织
  └──────┬───────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Step 1: Python 预处理 (pyloader/) - 原有模块，保持不变                    │
  │ ───────────────────────────────────────────────────────────────────────  │
  │  输入: ~/trace/smart/all/2015-01-*.csv                                   │
  │  处理:                                                                   │
  │    1. 特征提取 (Feature Extraction)                                       │
  │       - 读取 SMART 属性 (smart_1, smart_5, smart_9, smart_187等)          │
  │    2. 数据缓冲 (Buffering)                                                │
  │       - 按 serial_number 聚合数据                                        │
  │       - 滑动窗口管理 (默认30天)                                           │
  │    3. 样本标记 (Labeling)                                                 │
  │       - 正样本: 故障磁盘在故障前 window 内的记录                          │
  │       - 负样本: 正常磁盘的记录                                            │
  │    4. 第一阶段降采样 (Downsampling)                                        │
  │       - 处理类别不平衡问题                                                │
  │  输出: ARFF/CSV 格式文件                                                   │
  │    - ./pyloader/train/2015-01-30.arff  (训练数据)                        │
  │    - ./pyloader/test/2015-01-30.arff   (测试数据)                        │
  └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Step 2: LLM 预测 (llm_predictor/) - 替代原有 MOA Java 模块               │
  │ ───────────────────────────────────────────────────────────────────────  │
  │  输入: ARFF/CSV 文件                                                      │
  │  处理:                                                                   │
  │    1. 数据加载 (Data Loading)                                             │
  │    2. 数据压缩 (Compression)                                              │
  │    3. Prompt 构建 (Prompt Building)                                       │
  │    4. API 调用 (SiliconFlow + DeepSeek-V3.2)                              │
  │    5. 结果解析 (Response Parsing)                                         │
  │    6. 评估计算 (Evaluation)                                               │
  │  输出: 预测结果和评估指标                                                  │
  │    - ./hi7_llm/results.json                                              │
  │    - 控制台输出 FP, FPR, F1-score, Precision, Recall                     │
  └─────────────────────────────────────────────────────────────────────────┘
```

## 二、详细执行步骤

### 阶段 1: 环境准备

```bash
# 1.1 安装依赖
pip install pandas numpy requests pyyaml

# 1.2 设置 API 密钥
export SILICONFLOW_API_KEY="sk-your-api-key-here"

# 1.3 验证 API 连接
python llm_predictor/test_api.py
```

**关键检查点:**
- [ ] Python 3.x 已安装
- [ ] 依赖包安装成功
- [ ] API 密钥有效且有余额
- [ ] API 连接测试通过

---

### 阶段 2: 数据预处理 (pyloader)

#### 2.1 准备特征配置文件

```
pyloader/features_erg/hi7_all.txt
─────────────────────────────────
smart_1_normalized
smart_5_raw
smart_5_normalized
smart_9_raw
smart_187_raw
smart_197_raw
smart_197_normalized
```

#### 2.2 运行预处理脚本

```bash
cd pyloader

python run.py \
    -s "2015-01-30" \              # 开始日期
    -a 6 \                         # 标记窗口天数 (label_days)
    -p "~/trace/smart/all/" \      # 原始数据路径
    -r "../pyloader/hi7_train/" \  # 训练数据输出路径
    -e "../pyloader/hi7_test/" \   # 测试数据输出路径
    -c "features_erg/hi7_all.txt" \ # 特征配置文件
    -d "HDS722020ALA330" \          # 磁盘型号
    -i 10 \                         # 迭代天数
    -t "sliding" \                  # 遗忘类型: sliding/no
    -w 30 \                         # 正样本窗口大小
    -L 7 \                          # 负样本窗口大小
    -V 30                           # 验证窗口大小
```

#### 2.3 预处理内部流程

```
run.py
  │
  ├──> Simulate.__init__()
  │      │
  │      ├──> Memory.__init__()     # 初始化内存管理
  │      │      ├──> 设置路径、日期、窗口大小
  │      │      └──> 初始化数据缓冲区
  │      │
  │      └──> Memory.buffering()    # 数据缓冲
  │             │
  │             ├──> 读取原始 CSV 文件
  │             │      └──> 筛选指定型号的磁盘
  │             │
  │             ├──> 特征提取
  │             │      └──> 保留配置文件中指定的 SMART 属性
  │             │
  │             ├──> 数据标记
  │             │      ├──> 正样本: failure=1 的磁盘
  │             │      └──> 负样本: failure=0 的磁盘
  │             │
  │             └──> 降采样处理
  │                    └──> 平衡正负样本比例
  │
  └──> run_simulating()            # 主循环
         │
         ├──> 每天生成一个数据文件
         │      ├──> train/2015-01-30.arff
         │      ├──> train/2015-01-31.arff
         │      └──> test/2015-01-30.arff ...
         │
         └──> 保存为 ARFF 格式
```

#### 2.4 输出文件结构

```
pyloader/
├── hi7_train/
│   ├── 2015-01-30.arff      # 第1天训练数据
│   ├── 2015-01-31.arff      # 第2天训练数据
│   ├── 2015-02-01.arff
│   └── ...
│
└── hi7_test/
    ├── 2015-01-30.arff      # 第1天测试数据
    ├── 2015-01-31.arff      # 第2天测试数据
    └── ...
```

**ARFF 文件格式示例:**
```arff
@relation 2015-01-30

@attribute serial_number {SN001, SN002, SN003, ...}
@attribute failure {c0, c1}           # c0=正常, c1=故障
@attribute smart_1_normalized numeric
@attribute smart_5_raw numeric
@attribute smart_5_normalized numeric
@attribute smart_9_raw numeric
@attribute smart_187_raw numeric
@attribute smart_197_raw numeric
@attribute smart_197_normalized numeric

@data
'SN001','c0',114,0,100,1234,0,0,100
'SN002','c0',110,0,100,5678,0,0,100
'SN003','c1',95,10,50,9999,5,3,80
...
```

**关键检查点:**
- [ ] 训练数据文件已生成
- [ ] 测试数据文件已生成
- [ ] ARFF 文件格式正确
- [ ] 文件数量和迭代天数匹配

---

### 阶段 3: LLM 预测 (llm_predictor)

#### 3.1 初始化流程

```python
# predictor.py: LLMPredictor.__init__()

1. 创建 SiliconFlowClient
   ├──> 读取 API key
   ├──> 设置 base_url: https://api.siliconflow.cn/v1
   └──> 设置 model: deepseek-ai/DeepSeek-V3.2

2. 创建 DiskFailurePromptBuilder
   └──> 加载 SYSTEM_MESSAGE (SMART 数据分析专家角色)

3. 创建 DataCompressor (可选)
   ├──> method: gzip
   └──> level: 6

4. 创建 Evaluators
   ├──> global_evaluator: ClassificationEvaluator
   └──> local_evaluator: ClassificationEvaluator

5. 初始化跟踪变量
   ├──> keep_delay: {}  # 延迟评估跟踪
   ├──> iteration: 0
   └──> current_date: None
```

#### 3.2 主预测循环

```python
# predictor.py: run_simulation()

for iteration in range(iterations):
    
    # Step 1: 设置当前日期
    current_date = start_date + timedelta(days=iteration)
    
    # Step 2: 加载测试数据
    test_data = load_data_file(current_date.isoformat(), 'test')
    
    # Step 3: 处理测试数据
    metrics = process_test_data(test_data)
    
    # Step 4: 延迟评估 (如果启用)
    if bl_delay:
        delay_evaluate()
    
    # Step 5: 输出指标
    print_metrics()
    
    # Step 6: 保存结果
    save_results()
```

#### 3.3 单条数据处理流程

```python
# process_test_data() 详细流程

输入: test_data (DataFrame)
  │
  ├──> 提取 serial_numbers
  │      └──> test_data['serial_number']
  │
  ├──> 提取实际标签 (actual labels)
  │      └──> test_data['failure']
  │          └──> c0 -> 0, c1 -> 1
  │
  ├──> 提取特征列
  │      └──> 排除 [serial_number, failure, date, model]
  │
  ├──> 构建批次
  │      └──> batch_size = 10 (默认)
  │
  ├──> 对每个批次:
  │      │
  │      ├──> 构建 Prompt
  │      │      │
  │      │      ├──> System Message (角色定义)
  │      │      │      └──> "你是磁盘故障预测专家..."
  │      │      │
  │      │      └──> User Prompt (具体数据)
  │      │             └──> "分析以下磁盘 SMART 数据..."
  │      │                    + JSON 格式的 SMART 数据
  │      │
  │      ├──> 数据压缩 (可选)
  │      │      └──> gzip compress -> base64 encode
  │      │
  │      ├──> 调用 API
  │      │      │
  │      │      ├──> POST https://api.siliconflow.cn/v1/chat/completions
  │      │      ├──> Headers:
  │      │      │      Authorization: Bearer {API_KEY}
  │      │      │      Content-Type: application/json
  │      │      ├──> Body:
  │      │      │      {
  │      │      │        "model": "deepseek-ai/DeepSeek-V3.2",
  │      │      │        "messages": [
  │      │      │          {"role": "system", "content": SYSTEM_MESSAGE},
  │      │      │          {"role": "user", "content": prompt}
  │      │      │        ],
  │      │      │        "temperature": 0.1,
  │      │      │        "max_tokens": 500
  ��      │      │      }
  │      │      │
  │      │      └──> Response:
  │      │             {
  │      │               "choices": [{
  │      │                 "message": {
  │      │                   "content": "{...JSON...}"
  │      │                 }
  │      │               }],
  │      │               "usage": {
  │      │                 "prompt_tokens": 1200,
  │      │                 "completion_tokens": 150,
  │      │                 "total_tokens": 1350
  │      │               }
  │      │             }
  │      │
  │      ├──> 解析响应
  │      │      │
  │      │      ├──> 提取 JSON 内容
  │      │      │      └──> {
  │      │      │            "failure_probability": 0.85,
  │      │      │            "risk_level": "high",
  │      │      │            "reasoning": "smart_5_raw > 0 indicates...",
  │      │      │            "confidence": 0.92
  │      │      │          }
  │      │      │
  │      │      └──> 提取 failure_probability
  │      │             └──> predicted_prob = 0.85
  │      │
  │      └──> 保存预测结果
  │             └──> predictions.append(predicated_prob)
  │
  ├──> 更新评估器
  │      │
  │      ├──> for each disk:
  │      │      ├──> global_evaluator.add_result(actual, predicted_prob, sn)
  │      │      ├──> local_evaluator.add_result(actual, predicted_prob, sn)
  │      │      └──> delay_evaluator.add_instance(sn, actual, predicted_prob, day)
  │      │
  │      └──> 计算指标
  │             ├──> TP, FP, TN, FN
  │             ├──> Precision = TP / (TP + FP)
  │             ├──> Recall = TP / (TP + FN)
  │             ├──> F1 = 2 * (Precision * Recall) / (Precision + Recall)
  │             └──> FPR = FP / (FP + TN)
  │
  └──> 返回 metrics
```

#### 3.4 Prompt 构建示例

**System Message:**
```
You are an expert in disk failure prediction using SMART (Self-Monitoring, 
Analysis and Reporting Technology) data. Your task is to analyze disk SMART 
attributes and predict the probability of disk failure.

Key SMART attributes interpretation:
- smart_5_raw / Reallocated_Sector_Ct: Count of reallocated sectors. > 0 indicates potential failure
- smart_187_raw / Reported_Uncorrectable_Errors: Uncorrectable errors. > 0 is critical
- smart_197_raw / Current_Pending_Sector_Ct: Pending sectors to be remapped. > 0 is concerning
- smart_198_raw / Offline_Uncorrectable: Uncorrectable sectors found during offline scan. > 0 is critical
- smart_1_normalized / Raw_Read_Error_Rate: Lower normalized value indicates more errors

Failure risk indicators (in order of severity):
1. CRITICAL: smart_5_raw > 0 OR smart_187_raw > 0 OR smart_198_raw > 0
2. HIGH: smart_197_raw > 0
3. MEDIUM: smart_1_normalized < 100 OR other normalized values trending down
4. LOW: All attributes within normal ranges

Output format - return ONLY a JSON object:
{
    "failure_probability": <float 0.0-1.0>,
    "risk_level": "low|medium|high|critical",
    "reasoning": "<brief explanation based on SMART attributes>",
    "confidence": <float 0.0-1.0>
}
```

**User Prompt (Single Disk):**
```
Analyze the following disk SMART data and predict failure probability:

Current SMART readings:
{
  "serial_number": "SN123456",
  "smart_1_normalized": 114,
  "smart_5_raw": 0,
  "smart_5_normalized": 100,
  "smart_9_raw": 1234,
  "smart_187_raw": 0,
  "smart_197_raw": 0,
  "smart_197_normalized": 100
}

Provide your prediction in the specified JSON format.
```

**LLM Response:**
```json
{
  "failure_probability": 0.12,
  "risk_level": "low",
  "reasoning": "All critical SMART attributes are within normal ranges. smart_5_raw=0, smart_187_raw=0, smart_197_raw=0 indicate no reallocated or pending sectors. smart_1_normalized=114 is above threshold.",
  "confidence": 0.89
}
```

---

### 阶段 4: 结果解析

```python
# parse_results.py

输入: results.json
  │
  ├──> 读取所有迭代结果
  │
  ├──> 提取最终指标
  │      ├──> FP (False Positives)
  │      ├──> FPR (False Positive Rate)
  │      ├──> F1-score
  │      ├──> Precision
  │      └──> Recall
  │
  ├──> 计算平均值
  │      └──> 所有迭代的平均指标
  │
  └──> 输出表格
         └──> 类似原始 MOA 的格式
```

**输出示例:**
```
======================================================================
FINAL EVALUATION METRICS
======================================================================

Date: 2015-02-08
Iteration: 9

----------------------------------------------------------------------
Metric                          Value     Description                 
----------------------------------------------------------------------
FP                              22.00     False Positives - incorrectly predicted failures
FPR                             0.473107  False Positive Rate - FP / (FP + TN)
F1-score                        26.220090 F1 Score - harmonic mean of precision and recall
Precision                       16.235855 Precision - TP / (TP + FP)
Recall                          68.095238 Recall - TP / (TP + FN)
----------------------------------------------------------------------

======================================================================
ITERATION SUMMARY
======================================================================

      Date        FP      FPR       F1    Precision  Recall
2015-01-30      3.00   0.2345   45.2345     32.4567  78.9012
2015-01-31      4.00   0.2890   42.1234     30.5678  76.5432
...

======================================================================
AVERAGE METRICS ACROSS ALL ITERATIONS
======================================================================
Metric                          Average         Std Dev        
----------------------------------------------------------------------
FP                              22.000000       5.234567
FPR                             0.473107        0.123456
F1-score                        26.220090       8.901234
Precision                       16.235855       6.543210
Recall                          68.095238       12.345678
======================================================================
```

---

## 三、完整执行命令示例

### 方式 1: 使用一键脚本

```bash
# 1. 设置 API 密钥
export SILICONFLOW_API_KEY="sk-xxxxxxxxxxxxxxxx"

# 2. 执行脚本
./run_hi7_llm.sh
```

### 方式 2: 分步执行

```bash
# ========== Step 1: 预处理 ==========
cd pyloader

python run.py \
    -s "2015-01-30" \
    -a 6 \
    -p "~/trace/smart/all/" \
    -r "../hi7_llm/train/" \
    -e "../hi7_llm/test/" \
    -c "features_erg/hi7_all.txt" \
    -d "HDS722020ALA330" \
    -i 10 \
    -t "sliding" \
    -w 30 \
    -L 7 \
    -V 30

cd ..

# ========== Step 2: LLM 预测 ==========
export SILICONFLOW_API_KEY="sk-xxxxxxxxxxxxxxxx"

python llm_predictor/predictor.py \
    -s "2015-01-30" \
    -p "./hi7_llm/train/" \
    -t "./hi7_llm/test/" \
    -i 10 \
    -m "deepseek-ai/DeepSeek-V3.2" \
    --threshold 0.5 \
    --batch-size 10 \
    -o "./hi7_llm/results.json" \
    2>&1 | tee "./hi7_llm/prediction.log"

# ========== Step 3: 解析结果 ==========
python llm_predictor/parse_results.py "./hi7_llm/results.json"
```

---

## 四、关键配置参数说明

| 参数 | 位置 | 说明 | 建议值 |
|------|------|------|--------|
| `SILICONFLOW_API_KEY` | 环境变量 | API 密钥 | 从官网获取 |
| `-s, --start-date` | predictor.py | 开始日期 | 2015-01-30 |
| `-p, --train-path` | predictor.py | 训练数据路径 | ./train/ |
| `-t, --test-path` | predictor.py | 测试数据路径 | ./test/ |
| `-i, --iterations` | predictor.py | 迭代天数 | 10-30 |
| `--threshold` | predictor.py | 分类阈值 | 0.5 |
| `--batch-size` | predictor.py | API 批次大小 | 5-20 |
| `-a, --label_days` | run.py | 标记窗口 | 6 |
| `-w` | run.py | 正样本窗口 | 30 |
| `-L` | run.py | 负样本窗口 | 7 |
| `-V` | run.py | 验证窗口 | 30 |

---

## 五、故障排查流程

```
问题: API 调用失败
  │
  ├──> 检查 API 密钥
  │      └──> echo $SILICONFLOW_API_KEY
  │
  ├──> 测试 API 连接
  │      └──> python llm_predictor/test_api.py
  │
  └──> 检查网络连接
         └──> curl https://api.siliconflow.cn/v1/models

问题: 数据文件未找到
  │
  ├──> 检查 pyloader 输出
  │      └──> ls -la pyloader/train/
  │
  ├──> 检查路径配置
  │      └──> 确认 -r 和 -e 参数
  │
  └──> 重新运行 pyloader

问题: 指标异常
  │
  ├──> 检查数据格式
  │      └──> head -5 pyloader/train/2015-01-30.arff
  │
  ├──> 检查阈值设置
  │      └──> --threshold 0.5
  │
  └──> 检查标签映射
         └──> c0/c1 -> 0/1
```

---

## 六、性能优化建议

1. **API 成本控制**
   - 增大 batch-size 减少 API 调用次数
   - 使用压缩减少 token 数量
   - 限制 max_tokens 避免过长响应

2. **速度优化**
   - 使用本地缓存避免重复调用
   - 启用并发批量处理
   - 预加载数据文件

3. **准确性优化**
   - 调整 threshold 平衡 Precision/Recall
   - 优化 Prompt 提供更多信息
   - 使用历史数据增强上下文
