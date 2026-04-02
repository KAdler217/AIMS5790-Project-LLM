# LLM Predictor 执行流程图

## 1. 整体流程图

```mermaid
flowchart TB
    Start([开始]) --> Init[环境准备<br/>pip install + API Key]
    Init --> Preprocess[Step 1: pyloader 预处理]
    
    subgraph Preprocessing [数据预处理阶段]
        Preprocess --> LoadRaw[加载原始 SMART 数据]
        LoadRaw --> FeatureExtract[特征提取]
        FeatureExtract --> Buffer[数据缓冲<br/>滑动窗口]
        Buffer --> Label[样本标记<br/>正/负样本]
        Label --> Downsample[降采样处理]
        Downsample --> SaveARFF[保存 ARFF/CSV 文件]
    end
    
    SaveARFF --> Predict[Step 2: LLM 预测]
    
    subgraph Prediction [LLM 预测阶段]
        Predict --> LoadARFF[加载 ARFF 文件]
        LoadARFF --> Compress[数据压缩<br/>gzip]
        Compress --> BuildPrompt[构建 Prompt]
        BuildPrompt --> API[调用 SiliconFlow API<br/>DeepSeek-V3.2]
        API --> Parse[解析响应]
        Parse --> Evaluate[计算评估指标]
    end
    
    Evaluate --> Save[保存结果]
    Save --> ParseResult[Step 3: 解析结果]
    
    subgraph Result [结果处理阶段]
        ParseResult --> Metrics[提取指标<br/>FP/FPR/F1/Precision/Recall]
        Metrics --> Visual[可视化输出]
    end
    
    Visual --> End([结束])
```

## 2. 详细数据流图

```mermaid
flowchart LR
    subgraph Input [输入数据]
        Raw[原始 SMART 日志<br/>CSV 格式]
        Config[特征配置<br/>hi7_all.txt]
    end
    
    subgraph Pyloader [pyloader 模块]
        Run[run.py] --> Simulate[Simulate 类]
        Simulate --> Memory[Memory 类<br/>数据管理]
        Memory --> Buffer[buffering<br/>缓冲处理]
        Buffer --> Arff[ARFF 生成器]
    end
    
    subgraph LLM [llm_predictor 模块]
        Loader[data_loader.py<br/>加载 ARFF] --> Predictor[predictor.py<br/>主预测类]
        Predictor --> Client[llm_client.py<br/>API 客户端]
        Predictor --> Compressor[compressor.py<br/>数据压缩]
        Predictor --> Evaluator[evaluator.py<br/>评估计算]
        Client --> SF[SiliconFlow API]
    end
    
    subgraph Output [输出结果]
        JSON[results.json<br/>详细结果]
        Log[prediction.log<br/>执行日志]
        Table[指标表格<br/>FP/FPR/F1/...]
    end
    
    Raw --> Run
    Config --> Run
    Arff --> Loader
    SF --> Predictor
    Predictor --> JSON
    Predictor --> Log
    Evaluator --> Table
```

## 3. LLM API 调用详细流程

```mermaid
sequenceDiagram
    participant P as Predictor
    participant C as DataCompressor
    participant B as PromptBuilder
    participant A as API Client
    participant S as SiliconFlow
    participant E as Evaluator
    
    loop 每天迭代
        P->>P: load_data_file(date)
        P->>P: extract_features()
        
        loop 每个批次
            P->>C: compress(batch_data)
            C-->>P: compressed_data
            
            P->>B: build_prompt(disk_data)
            B-->>P: prompt
            
            P->>A: predict(prompt)
            
            A->>S: POST /chat/completions
            Note right of A: {model: DeepSeek-V3.2,<br/>messages: [system, user]}
            
            S-->>A: {choices: [...], usage: {...}}
            A-->>P: {text, usage}
            
            P->>P: parse_response()
            P->>E: add_result(actual, predicted)
        end
        
        P->>E: get_metrics()
        E-->>P: {FP, FPR, F1, ...}
        P->>P: save_results()
    end
```

## 4. 评估指标计算流程

```mermaid
flowchart TD
    Start[开始评估] --> Confusion[构建混淆矩阵]
    
    subgraph ConfusionMatrix [混淆矩阵]
        TP[TP<br/>预测故障<br/>实际故障]
        FP[FP<br/>预测故障<br/>实际正常]
        TN[TN<br/>预测正常<br/>实际正常]
        FN[FN<br/>预测正常<br/>实际故障]
    end
    
    Confusion --> TP & FP & TN & FN
    
    TP & FP --> Precision[Precision = TP / (TP + FP)]
    TP & FN --> Recall[Recall = TP / (TP + FN)]
    FP & TN --> FPR[FPR = FP / (FP + TN)]
    
    Precision & Recall --> F1[F1 = 2 * P * R / (P + R)]
    
    F1 & FPR --> Output[输出指标表格]
```

## 5. Prompt 构建流程

```mermaid
flowchart LR
    subgraph Input [输入]
        Data[磁盘 SMART 数据]
        History[历史数据<br/>可选]
    end
    
    subgraph Builder [PromptBuilder]
        System[SYSTEM_MESSAGE<br/>专家角色定义]
        Template[构建模板]
        Json[JSON 格式化<br/>数据]
        Combine[组合 Prompt]
    end
    
    subgraph Output [输出]
        Final[完整 Prompt<br/>System + User]
    end
    
    Data --> Json
    History --> Template
    System --> Combine
    Template --> Combine
    Json --> Combine
    Combine --> Final
```

## 6. 错误处理流程

```mermaid
flowchart TD
    Start[API 调用] --> Check{检查}
    
    Check -->|API Key 无效| Error1[错误: API Key 未设置]
    Check -->|网络错误| Error2[错误: 连接失败]
    Check -->|Rate Limit| Error3[错误: 频率限制]
    Check -->|正常| Process[继续处理]
    
    Error1 --> Fix1[设置环境变量<br/>SILICONFLOW_API_KEY]
    Error2 --> Fix2[检查网络<br/>重试]
    Error3 --> Fix3[增加延迟<br/>降低 batch_size]
    
    Fix1 & Fix2 & Fix3 --> Retry[重试调用]
    Retry --> Check
    
    Process --> Parse[解析响应]
    Parse --> Valid{JSON 有效?}
    
    Valid -->|无效| Default[使用默认值<br/>probability=0.5]
    Valid -->|有效| Extract[提取概率值]
    
    Default & Extract --> End[返回结果]
```

## 7. 完整执行时序图

```mermaid
sequenceDiagram
    actor User
    participant Shell as run_hi7_llm.sh
    participant Pyloader as pyloader/run.py
    participant Predictor as llm_predictor/predictor.py
    participant API as SiliconFlow API
    participant Parser as parse_results.py
    
    User->>Shell: 执行脚本
    Shell->>Shell: 检查 SILICONFLOW_API_KEY
    
    Shell->>Pyloader: 调用预处理
    activate Pyloader
    
    loop 每天数据处理
        Pyloader->>Pyloader: 读取原始 CSV
        Pyloader->>Pyloader: 特征提取
        Pyloader->>Pyloader: 标记样本
        Pyloader->>Pyloader: 生成 ARFF
    end
    
    Pyloader-->>Shell: 完成
    deactivate Pyloader
    
    Shell->>Predictor: 调用 LLM 预测
    activate Predictor
    
    Predictor->>Predictor: 初始化 Client
    
    loop iterations 天
        Predictor->>Predictor: 加载当天 ARFF
        
        loop 每个 batch
            Predictor->>Predictor: 构建 Prompt
            Predictor->>API: POST /chat/completions
            activate API
            API-->>Predictor: 返回预测结果
            deactivate API
            Predictor->>Predictor: 解析 JSON
        end
        
        Predictor->>Predictor: 更新评估指标
        Predictor->>Predictor: 输出当天结果
    end
    
    Predictor->>Predictor: 保存 results.json
    Predictor-->>Shell: 完成
    deactivate Predictor
    
    Shell->>Parser: 调用结果解析
    activate Parser
    Parser->>Parser: 读取 results.json
    Parser->>Parser: 计算统计指标
    Parser-->>User: 显示最终表格
    deactivate Parser
```

## 8. 模块依赖关系

```mermaid
graph TD
    subgraph Main [主模块]
        Predictor[predictor.py<br/>主预测类]
    end
    
    subgraph Core [核心模块]
        Loader[data_loader.py<br/>数据加载]
        Client[llm_client.py<br/>API 客户端]
        Eval[evaluator.py<br/>评估器]
        Comp[compressor.py<br/>压缩器]
    end
    
    subgraph Utils [工具模块]
        Parse[parse_results.py<br/>结果解析]
        Test[test_api.py<br/>API 测试]
    end
    
    Predictor --> Loader
    Predictor --> Client
    Predictor --> Eval
    Predictor --> Comp
    
    Client --> |调用| SF[SiliconFlow API]
    
    Parse --> |读取| Results[results.json]
    Test --> |测试| Client
    
    style Predictor fill:#f9f,stroke:#333,stroke-width:4px
    style SF fill:#bbf,stroke:#333,stroke-width:2px
```

## 9. 配置流程

```mermaid
flowchart LR
    subgraph Env [环境配置]
        Key[设置 API Key<br/>export SILICONFLOW_API_KEY=...]
        Dep[安装依赖<br/>pip install ...]
    end
    
    subgraph Data [数据配置]
        Path[配置数据路径<br/>config.yaml]
        Feature[选择特征<br/>hi7_all.txt]
    end
    
    subgraph Model [模型配置]
        ModelName[模型名称<br/>deepseek-ai/DeepSeek-V3.2]
        Threshold[分类阈值<br/>0.5]
        Batch[批次大小<br/>10]
    end
    
    subgraph Run [运行配置]
        StartDate[开始日期<br/>2015-01-30]
        Iter[迭代次数<br/>10]
        Window[窗口大小<br/>30]
    end
    
    Key & Dep --> Ready[准备就绪]
    Path & Feature --> Ready
    ModelName & Threshold & Batch --> Ready
    StartDate & Iter & Window --> Ready
    Ready --> Execute[执行预测]
```


class MultiLLMPredictor:
    def __init__(self, d_model,num_heads,dropout):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_nodel, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)


