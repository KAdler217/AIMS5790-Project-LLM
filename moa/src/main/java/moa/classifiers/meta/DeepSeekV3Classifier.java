/*
 * DeepSeekV3Classifier.java
 * DeepSeekV3大模型分类器，通过硅基流动平台API调用
 */
package moa.classifiers.meta;

import com.github.javacliparser.*;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import moa.core.MiscUtils;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;

/**
 * DeepSeekV3大模型分类器，通过硅基流动平台API调用
 * 
 * 该分类器将磁盘SMART数据转换为提示词，通过DeepSeekV3 API进行分类预测。
 * 支持在线学习和提示词更新，适应概念漂移场景。
 * 
 * @author Generated for MOA integration
 */
public class DeepSeekV3Classifier extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "DeepSeekV3: 大语言模型分类器，通过硅基流动平台API调用";
    }

    // API配置选项
    public StringOption apiKeyOption = new StringOption("apiKey", 'k',
            "硅基流动平台API密钥", "");
    
    public StringOption modelOption = new StringOption("model", 'm',
            "模型名称", "deepseek-ai/DeepSeek-V3");
    
    public StringOption endpointOption = new StringOption("endpoint", 'e',
            "API端点URL", "https://api.siliconflow.cn/v1/chat/completions");
    
    public FloatOption temperatureOption = new FloatOption("temperature", 't',
            "温度参数，控制随机性", 0.1, 0.0, 2.0);
    
    public IntOption maxTokensOption = new IntOption("maxTokens", 'x',
            "最大生成token数", 30, 1, 100);  // 进一步减少最大生成token数
    
    public IntOption batchSizeOption = new IntOption("batchSize", 'b',
            "批处理大小", 1, 1, 3);  // 进一步减小批处理大小
    
    public IntOption updateFrequencyOption = new IntOption("updateFrequency", 'u',
            "提示词更新频率", 50, 1, Integer.MAX_VALUE);
    
    public FlagOption onlineLearningOption = new FlagOption("onlineLearning", 'o',
            "启用在线学习");
    
    public StringOption systemPromptOption = new StringOption("systemPrompt", 'p',
            "系统提示词", "预测(0/1)");  // 极简系统提示词

    // 内部状态
    protected String apiKey;
    protected String model;
    protected String endpoint;
    protected double temperature;
    protected int maxTokens;
    protected int batchSize;
    protected int updateFrequency;
    protected boolean onlineLearning;
    protected String systemPrompt;
    
    // 训练缓冲区
    protected List<Instance> trainingBuffer;
    protected List<String> examplePrompts;
    protected int instancesSinceUpdate;
    
    // 性能统计
    protected int apiCallCount;
    protected long totalApiTime;
    protected int errorCount;

    @Override
    public void resetLearningImpl() {
        // 初始化参数
        this.apiKey = apiKeyOption.getValue();
        this.model = modelOption.getValue();
        this.endpoint = endpointOption.getValue();
        this.temperature = (float)temperatureOption.getValue();
        this.maxTokens = maxTokensOption.getValue();
        this.batchSize = batchSizeOption.getValue();
        this.updateFrequency = updateFrequencyOption.getValue();
        this.onlineLearning = onlineLearningOption.isSet();
        this.systemPrompt = systemPromptOption.getValue();
        
        // 初始化内部状态
        this.trainingBuffer = new ArrayList<>();
        this.examplePrompts = new ArrayList<>();
        this.instancesSinceUpdate = 0;
        
        // 重置性能统计
        this.apiCallCount = 0;
        this.totalApiTime = 0;
        this.errorCount = 0;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        // 添加实例到训练缓冲区
        trainingBuffer.add(inst);
        
        // 当缓冲区满时处理批次
        if (trainingBuffer.size() >= batchSize) {
            processBatch();
            trainingBuffer.clear();
        }
        
        // 如果启用在线学习，定期更新提示词
        if (onlineLearning) {
            instancesSinceUpdate++;
            if (instancesSinceUpdate >= updateFrequency) {
                updatePrompt();
                instancesSinceUpdate = 0;
            }
        }
    }
    
    protected void processBatch() {
        // 为每个实例创建提示词
        List<String> prompts = new ArrayList<>();
        List<String> expectedOutputs = new ArrayList<>();
        
        for (Instance inst : trainingBuffer) {
            String prompt = createPromptFromInstance(inst);
            prompts.add(prompt);
            expectedOutputs.add(String.valueOf((int) inst.classValue()));
        }
        
        // 存储示例用于少样本学习
        for (int i = 0; i < prompts.size(); i++) {
            examplePrompts.add(prompts.get(i) + " -> " + expectedOutputs.get(i));
        }
        
        // 保持示例数量在合理范围内
        if (examplePrompts.size() > 20) {
            examplePrompts = examplePrompts.subList(examplePrompts.size() - 20, examplePrompts.size());
        }
    }
    
    protected String createPromptFromInstance(Instance inst) {
        StringBuilder prompt = new StringBuilder();
        
        // 使用极简提示词格式
        prompt.append("预测:");
        
        // 只使用最重要的3个特征
        int maxFeatures = Math.min(3, inst.numAttributes() - 1);
        for (int i = 0; i < maxFeatures; i++) {
            double value = inst.value(i);
            prompt.append(String.format("%.1f", value));
            if (i < maxFeatures - 1) {
                prompt.append(",");
            }
        }
        
        // 确保提示词不超过500字符
        String promptStr = prompt.toString();
        if (promptStr.length() > 500) {
            promptStr = promptStr.substring(0, 500);
        }
        
        return promptStr;
    }
    
    protected void updatePrompt() {
        // 基于最新示例更新系统提示词
        if (examplePrompts.size() > 0) {
            StringBuilder newPrompt = new StringBuilder(systemPrompt);
            newPrompt.append("\n\n以下是一些示例：\n");
            
            // 添加最近的示例
            int numExamples = Math.min(5, examplePrompts.size());
            for (int i = examplePrompts.size() - numExamples; i < examplePrompts.size(); i++) {
                newPrompt.append(examplePrompts.get(i)).append("\n");
            }
            
            systemPrompt = newPrompt.toString();
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        // 创建提示词
        String prompt = createPromptFromInstance(inst);
        
        // 直接使用极简提示词，不使用少样本学习以减少token
        String fullPrompt = systemPrompt + " " + prompt;
        
        // 调用API获取预测
        String response = callDeepSeekAPI(fullPrompt);
        
        // 解析响应为类别概率
        return parseResponseToVotes(response);
    }
    
    protected String callDeepSeekAPI(String prompt) {
        try {
            long startTime = System.currentTimeMillis();
            
            // 构建请求体
            String requestBody = String.format(
                "{\"model\": \"%s\", \"messages\": [{\"role\": \"user\", \"content\": \"%s\"}], \"temperature\": %f, \"max_tokens\": %d}",
                model, escapeJson(prompt), temperature, maxTokens
            );
            
            // 创建连接
            URL url = new URL(endpoint);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("Authorization", "Bearer " + apiKey);
            connection.setDoOutput(true);
            
            // 发送请求
            try (OutputStream os = connection.getOutputStream()) {
                byte[] input = requestBody.getBytes(StandardCharsets.UTF_8);
                os.write(input, 0, input.length);
            }
            
            // 读取响应
            int responseCode = connection.getResponseCode();
            StringBuilder response = new StringBuilder();
            
            if (responseCode == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8))) {
                    String responseLine;
                    while ((responseLine = br.readLine()) != null) {
                        response.append(responseLine.trim());
                    }
                }
            } else {
                errorCount++;
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(connection.getErrorStream(), StandardCharsets.UTF_8))) {
                    String responseLine;
                    while ((responseLine = br.readLine()) != null) {
                        response.append(responseLine.trim());
                    }
                }
                System.err.println("API错误 (" + responseCode + "): " + response.toString());
                return "0"; // 默认返回正常
            }
            
            // 更新统计
            long endTime = System.currentTimeMillis();
            apiCallCount++;
            totalApiTime += (endTime - startTime);
            
            // 解析JSON响应提取内容
            return extractContentFromJson(response.toString());
            
        } catch (Exception e) {
            errorCount++;
            System.err.println("API调用异常: " + e.getMessage());
            return "0"; // 默认返回正常
        }
    }
    
    protected String extractContentFromJson(String jsonResponse) {
        try {
            // 简单的JSON解析，提取choices[0].message.content
            int contentStart = jsonResponse.indexOf("\"content\":\"") + 11;
            int contentEnd = jsonResponse.indexOf("\"", contentStart);
            
            if (contentStart > 10 && contentEnd > contentStart) {
                return jsonResponse.substring(contentStart, contentEnd);
            }
            
            // 如果简单解析失败，返回原始响应
            return jsonResponse;
        } catch (Exception e) {
            System.err.println("JSON解析错误: " + e.getMessage());
            return jsonResponse;
        }
    }
    
    protected String escapeJson(String text) {
        return text.replace("\\", "\\\\")
                  .replace("\"", "\\\"")
                  .replace("\n", "\\n")
                  .replace("\r", "\\r")
                  .replace("\t", "\\t");
    }
    
    protected double[] parseResponseToVotes(String response) {
        // 尝试从响应中提取数字
        try {
            // 清理响应，只保留数字和可能的负号
            String cleanResponse = response.replaceAll("[^0-9]", "");
            
            if (cleanResponse.isEmpty()) {
                // 如果没有数字，返回均匀分布
                double[] votes = new double[2];
                votes[0] = 0.5;
                votes[1] = 0.5;
                return votes;
            }
            
            int predictedClass = Integer.parseInt(cleanResponse);
            double[] votes = new double[2];
            
            if (predictedClass == 1) {
                votes[0] = 0.0;
                votes[1] = 1.0;
            } else {
                votes[0] = 1.0;
                votes[1] = 0.0;
            }
            
            return votes;
        } catch (NumberFormatException e) {
            // 解析失败，返回均匀分布
            double[] votes = new double[2];
            votes[0] = 0.5;
            votes[1] = 0.5;
            return votes;
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[] {
            new Measurement("API调用次数", apiCallCount),
            new Measurement("平均API响应时间(ms)", 
                apiCallCount > 0 ? (double) totalApiTime / apiCallCount : 0.0),
            new Measurement("API错误次数", errorCount),
            new Measurement("训练缓冲区大小", trainingBuffer.size()),
            new Measurement("示例数量", examplePrompts.size()),
            new Measurement("自上次更新实例数", instancesSinceUpdate)
        };
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // 添加模型描述
        StringUtils.appendIndented(out, indent, "DeepSeekV3大模型分类器");
        StringUtils.appendNewline(out);
        StringUtils.appendIndented(out, indent + 1, "模型: " + model);
        StringUtils.appendNewline(out);
        StringUtils.appendIndented(out, indent + 1, "端点: " + endpoint);
        StringUtils.appendNewline(out);
        StringUtils.appendIndented(out, indent + 1, "批处理大小: " + batchSize);
        StringUtils.appendNewline(out);
        StringUtils.appendIndented(out, indent + 1, "在线学习: " + onlineLearning);
        StringUtils.appendNewline(out);
        StringUtils.appendIndented(out, indent + 1, "API调用次数: " + apiCallCount);
        StringUtils.appendNewline(out);
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}