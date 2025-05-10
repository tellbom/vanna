# Vanna MySQL Qdrant 扩展 API 文档

## 目录
1. [API 概述](#api-概述)
2. [身份验证](#身份验证)
3. [检索上下文 API](#检索上下文-api)
4. [文档管理 API](#文档管理-api)
   - [上传 Markdown 文档](#上传-markdown-文档)
   - [上传 DDL 文件](#上传-ddl-文件)
   - [上传 SQL 示例](#上传-sql-示例)
   - [上传术语词典](#上传术语词典)
5. [任务管理 API](#任务管理-api)
6. [与 Dify 集成](#与-dify-集成)
7. [生产环境部署注意事项](#生产环境部署注意事项)

---

## API 概述

Vanna MySQL Qdrant 扩展 API 提供了一系列端点，用于检索 SQL 上下文、管理文档、DDL 和 SQL 示例，以及监控异步任务进度。这些 API 设计用于支持高效的文本到 SQL 转换服务，特别是在机械制造和设备管理等专业领域。

**基础 URL**: `http://your-server:8084`

---

## 身份验证

所有 API 请求需要通过 `Authorization` 头进行身份验证。

**身份验证头格式**:
```
Authorization: Bearer YOUR_AUTH_TOKEN
```

**说明**: 
- 替换 `YOUR_AUTH_TOKEN` 为您的实际认证令牌
- 无效的认证令牌将返回 401 Unauthorized 响应

---

## 检索上下文 API

### 获取检索上下文

检索与问题相关的 SQL 示例、表定义和文档，用于支持 SQL 生成。

**端点**: `GET /api/v0/get_retrieval_context`

**参数**:

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| question | String | 是 | 用户问题文本 |
| max_results | Integer | 否 | 每种类型最大返回结果数，默认值 10 |
| include_prompt | Boolean | 否 | 是否包含生成的提示词，默认值 true |
| enhance_query | Boolean | 否 | 是否增强查询，默认值 true |
| use_rerank | Boolean | 否 | 是否使用重排序，默认值 true |
| intent_type | String | 否 | 查询意图类型，支持的值包括: status, time, type, count, aggregation, comparison, equipment, component, material, specification, process, maintenance, troubleshooting, quality, efficiency, cost, safety, supply_chain |

**请求示例**:
```bash
curl -X GET \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
  "http://your-server:8084/api/v0/get_retrieval_context?question=%E6%9F%A5%E8%AF%A2%E8%BF%87%E5%8E%BB%E4%B8%80%E5%B9%B4%E4%B8%AD%E7%BB%B4%E4%BF%AE%E6%AC%A1%E6%95%B0%E6%9C%80%E5%A4%9A%E7%9A%84%E4%B8%89%E5%8F%B0%E8%AE%BE%E5%A4%87%E5%8F%8A%E5%85%B6%E7%BB%B4%E4%BF%AE%E6%80%BB%E8%B4%B9%E7%94%A8&max_results=15&include_prompt=true&enhance_query=true&use_rerank=true&intent_type=maintenance"
```

**响应格式**:
```json
{
  "type": "retrieval_context",
  "question": "查询过去一年中维修次数最多的三台设备及其维修总费用",
  "context": {
    "question_sql_list": [
      {
        "question": "查询过去一年中维修次数最多的三台设备及其维修总费用",
        "sql": "SELECT e.equipment_id, e.equipment_name, COUNT(m.maintenance_id) AS maintenance_count, SUM(m.maintenance_cost) AS total_cost FROM equipment_info e JOIN maintenance_record m ON e.equipment_id = m.equipment_id WHERE m.start_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR) GROUP BY e.equipment_id, e.equipment_name ORDER BY maintenance_count DESC, total_cost DESC LIMIT 3"
      }
    ],
    "ddl_list": [
      "CREATE TABLE equipment_info (\n  equipment_id VARCHAR(20) PRIMARY KEY,\n  equipment_name VARCHAR(100) NOT NULL,\n  model VARCHAR(50),\n  manufacturer VARCHAR(100),\n  purchase_date DATE,\n  installation_date DATE,\n  service_life INT COMMENT '预计使用年限(年)',\n  status ENUM('正常', '维修中', '停用'),\n  department_id VARCHAR(20),\n  location VARCHAR(100),\n  last_maintenance_date DATE,\n  FOREIGN KEY (department_id) REFERENCES department_info(department_id)\n);"
    ],
    "doc_list": [
      "# 表名词:设备信息表\n设备信息表存储了所有生产设备的基本信息，是整个系统的核心表。\n\n| 字段名 | 数据类型 | 说明 | 备注 |\n|--------|----------|------|------|\n| equipment_id | VARCHAR(20) | 设备编号 | 主键 |\n| equipment_name | VARCHAR(100) | 设备名称 | 非空 |"
    ],
    "table_relationships": "表 equipment_info 的列 department_id 引用 表 department_info 的列 department_id\n表 maintenance_record 的列 equipment_id 引用 表 equipment_info 的列 equipment_id"
  },
  "dialect": "MySQL",
  "retrieval_stats": {
    "enhanced_question": "查询过去一年中维修次数最多的三台设备及其维修总费用 maintenance maintenance_count repair service",
    "sql_retrieval_time": 0.528,
    "sql_results_count": 1,
    "ddl_retrieval_time": 0.175,
    "ddl_results_count": 3,
    "doc_retrieval_time": 0.412,
    "doc_results_count": 5,
    "prompt_generation_time": 0.038
  },
  "total_time": 1.153,
  "detected_intent": "maintenance",
  "prompt": [
    {
      "role": "system",
      "content": "你是一位精通MySQL的专家..."
    },
    {
      "role": "user",
      "content": "查询过去一年中维修次数最多的三台设备及其维修总费用"
    }
  ]
}
```

**错误响应**:
```json
{
  "type": "error",
  "error": "错误消息",
  "details": "详细错误信息（仅在调试模式下提供）"
}
```

---

## 文档管理 API

### 上传 Markdown 文档

上传包含表定义、业务场景和规则的 Markdown 文档。

**端点**: `POST /api/v0/upload_documentation`

**请求体**:
- `Content-Type`: `multipart/form-data`
- `file`: Markdown 文件 (.md)

**请求示例**:
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
  -F "file=@equipment_management.md" \
  http://your-server:8084/api/v0/upload_documentation
```

**响应格式**:
```json
{
  "type": "task_submitted",
  "task_id": "9b1c8fd3-7a23-4b5c-8f9e-d8f631a2e0b7",
  "message": "文档上传已开始。系统将在后台处理文档分块和训练。"
}
```

**文档格式要求**:
- 文档应使用 Markdown 格式
- 表定义使用 `# 表名词:表名` 格式
- 表结构使用 Markdown 表格格式
- 业务场景使用 `业务场景:` 前缀
- 业务规则使用 `业务规则:` 前缀
- 示例数据使用 `示例数据:` 前缀，可包含 JSON 格式示例

### 上传 DDL 文件

上传包含表定义的 DDL 文件。

**端点**: `POST /api/v0/upload_ddl`

**请求体**:
- `Content-Type`: `multipart/form-data`
- `file`: SQL 文件 (.sql, .ddl) 或文本文件 (.txt)，包含 CREATE TABLE 语句

**请求示例**:
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
  -F "file=@equipment_management.sql" \
  http://your-server:8084/api/v0/upload_ddl
```

**响应格式**:
```json
{
  "type": "task_submitted",
  "task_id": "7d8e9f0a-1b2c-3d4e-5f6g-7h8i9j0k1l2m",
  "message": "DDL上传已开始。系统将在后台提取和训练表定义。"
}
```

**DDL 文件格式示例**:
```sql
-- 设备管理系统数据库结构
-- 创建部门信息表
CREATE TABLE department_info (
    department_id VARCHAR(20) PRIMARY KEY,
    department_name VARCHAR(50) NOT NULL,
    manager VARCHAR(50),
    contact_phone VARCHAR(20),
    location VARCHAR(100)
);

-- 创建设备信息表
CREATE TABLE equipment_info (
    equipment_id VARCHAR(20) PRIMARY KEY,
    equipment_name VARCHAR(100) NOT NULL,
    model VARCHAR(50),
    manufacturer VARCHAR(100),
    purchase_date DATE,
    installation_date DATE,
    service_life INT COMMENT '预计使用年限(年)',
    status ENUM('正常', '维修中', '停用'),
    department_id VARCHAR(20),
    location VARCHAR(100),
    last_maintenance_date DATE,
    FOREIGN KEY (department_id) REFERENCES department_info(department_id)
);

-- 创建维护记录表
CREATE TABLE maintenance_record (
    maintenance_id VARCHAR(20) PRIMARY KEY,
    equipment_id VARCHAR(20),
    maintenance_type ENUM('定期维护', '故障维修'),
    start_date DATETIME,
    end_date DATETIME,
    maintenance_staff VARCHAR(50),
    fault_description TEXT,
    maintenance_content TEXT,
    parts_replaced TEXT COMMENT '多个零部件用逗号分隔',
    maintenance_cost DECIMAL(10,2) COMMENT '单位：元',
    status ENUM('计划中', '进行中', '已完成'),
    FOREIGN KEY (equipment_id) REFERENCES equipment_info(equipment_id)
);
```

### 上传 SQL 示例

上传包含问题和对应 SQL 查询的示例文件。

**端点**: `POST /api/v0/upload_sql_examples`

**请求体**:
- `Content-Type`: `multipart/form-data`
- `file`: Excel 文件 (.xlsx) 或 CSV 文件 (.csv)，包含 question 和 sql 列

**请求示例**:
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
  -F "file=@equipment_sql_examples.csv" \
  http://your-server:8084/api/v0/upload_sql_examples
```

**响应格式**:
```json
{
  "type": "task_submitted",
  "task_id": "3c4d5e6f-7g8h-9i0j-1k2l-3m4n5o6p7q8r",
  "message": "SQL示例上传已开始。系统将在后台处理和训练示例。"
}
```

**SQL 示例文件格式**:
CSV 文件必须包含 `question` 和 `sql` 两列，示例:
```csv
question,sql
"查询过去一年中维修次数最多的三台设备及其维修总费用","SELECT e.equipment_id, e.equipment_name, COUNT(m.maintenance_id) AS maintenance_count, SUM(m.maintenance_cost) AS total_cost FROM equipment_info e JOIN maintenance_record m ON e.equipment_id = m.equipment_id WHERE m.start_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR) GROUP BY e.equipment_id, e.equipment_name ORDER BY maintenance_count DESC, total_cost DESC LIMIT 3"
"查询所有服役超过预计使用年限但仍处于正常状态的设备","SELECT equipment_id, equipment_name, model, manufacturer, purchase_date, service_life, TIMESTAMPDIFF(YEAR, purchase_date, CURDATE()) AS actual_years FROM equipment_info WHERE status = '正常' AND TIMESTAMPDIFF(YEAR, purchase_date, CURDATE()) > service_life ORDER BY (TIMESTAMPDIFF(YEAR, purchase_date, CURDATE()) - service_life) DESC"
```

### 上传术语词典

上传包含机械制造领域专业术语的词典文件。

**端点**: `POST /api/v0/upload_terminology`

**请求体**:
- `Content-Type`: `multipart/form-data`
- `file`: TXT 文件 (.txt)、Excel 文件 (.xlsx) 或 CSV 文件 (.csv)

**请求示例**:
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
  -F "file=@mechanical_terms.txt" \
  http://your-server:8084/api/v0/upload_terminology
```

**响应格式**:
```json
{
  "type": "terminology_upload_result",
  "path": "/dictionary/MechanicalWords.txt",
  "terms": 320,
  "status": "success"
}
```

**术语词典格式**:
- **TXT 文件**: 一行英文一行中文，交替排列
- **Excel/CSV 文件**: 必须包含 `english` 和 `chinese` 两列

---

## 任务管理 API

### 获取任务状态

查询异步任务的执行状态和结果。

**端点**: `GET /api/v0/task_status`

**参数**:

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| task_id | String | 是 | 任务 ID |

**请求示例**:
```bash
curl -X GET \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
  "http://your-server:8084/api/v0/task_status?task_id=9b1c8fd3-7a23-4b5c-8f9e-d8f631a2e0b7"
```

**响应格式**:
```json
{
  "type": "task_status",
  "task_id": "9b1c8fd3-7a23-4b5c-8f9e-d8f631a2e0b7",
  "status": {
    "status": "completed",
    "result": ["doc-123456"]
  }
}
```

**可能的状态值**:
- `pending`: 任务等待中
- `running`: 任务执行中
- `completed`: 任务已完成
- `failed`: 任务执行失败
- `not_found`: 任务不存在

---

## 与 Dify 集成

### 在 Dify 工作流中使用检索 API

以下示例展示如何在 Dify 工作流中集成 Vanna 检索 API：

1. **创建新工作流**:
   - 在 Dify 平台创建新工作流
   - 添加"发送 HTTP 请求"节点

2. **配置 HTTP 请求节点**:
   - 方法: `GET`
   - URL: `http://your-server:8084/api/v0/get_retrieval_context`
   - 头信息:
     ```
     Authorization: Bearer YOUR_AUTH_TOKEN
     ```
   - 查询参数:
     ```
     question: {{$input.query}}
     max_results: 10
     include_prompt: true
     enhance_query: true
     use_rerank: true
     ```

3. **添加条件分支节点**:
   - 条件: `{{$httpResponse.type}} == "retrieval_context"`
   - 成功路径: 连接到 LLM 调用节点
   - 失败路径: 连接到错误处理节点

4. **配置 LLM 调用节点**:
   - 提示词: 使用 `{{$httpResponse.prompt}}` 传递检索到的完整提示词
   - 或者构建自定义提示词:
     ```
     我需要把这个问题转换为SQL: {{$input.query}}

     以下是相关的表结构:
     {{$httpResponse.context.ddl_list}}

     表之间的关系:
     {{$httpResponse.context.table_relationships}}

     类似的SQL示例:
     {{#each $httpResponse.context.question_sql_list}}
     问题: {{this.question}}
     SQL: {{this.sql}}
     {{/each}}

     请生成解决问题的SQL查询:
     ```

### Dify 工作流集成最佳实践

1. **缓存检索结果**:
   - 使用 Dify 的变量存储功能缓存相似问题的检索结果
   - 设置合理的缓存过期时间（如 1 小时）

2. **错误处理**:
   - 为 HTTP 请求添加超时设置（建议 10-15 秒）
   - 添加重试逻辑（最多 3 次重试，指数退避）
   - 当 API 不可用时提供优雅的降级方案

3. **意图分类**:
   - 在 Dify 工作流前添加意图分类步骤
   - 将检测到的意图传递给检索 API: `intent_type: {{$intent}}`

4. **参数调优**:
   - 对简单问题设置较小的 `max_results`（3-5）
   - 对复杂问题设置较大的 `max_results`（10-15）
   - 根据问题复杂度动态调整参数

---

## 生产环境部署注意事项

### 系统要求

- **服务器配置**:
  - CPU: 最低 4 核，推荐 8 核或更高
  - 内存: 最低 16GB，推荐 32GB 或更高
  - 存储: 最低 100GB SSD，用于模型、索引和数据存储
  - GPU: 可选，但推荐用于加速嵌入生成

- **软件依赖**:
  - Docker 和 Docker Compose
  - Qdrant 向量数据库（推荐 v1.1.0 或更高）
  - Elasticsearch（推荐 7.14 或更高）
  - Python 3.10 或更高

### 安全配置

1. **API 认证**:
   - 使用强密钥生成 JWT 令牌
   - 实现令牌轮换机制（每 30 天更换一次）
   - 根据需要实现 IP 白名单过滤

2. **网络安全**:
   - 使用 HTTPS 加密所有通信
   - 配置反向代理（如 Nginx）限制直接访问
   - 实现请求速率限制（如每分钟 100 请求）

3. **数据安全**:
   - 定期备份 Qdrant 和 Elasticsearch 数据
   - 实施数据访问审计日志
   - 敏感数据加密存储

### 监控与日志

1. **设置监控**:
   - 使用 Prometheus 监控系统指标
   - 配置 Grafana 面板可视化性能
   - 设置关键指标告警（如高延迟、高错误率）

2. **日志管理**:
   - 使用 ELK 栈或类似工具集中管理日志
   - 配置日志轮换防止磁盘空间耗尽
   - 实现关键错误的实时告警

### 扩展性考虑

1. **水平扩展**:
   - 将 API 服务部署在多个容器/节点上
   - 使用负载均衡器分配请求
   - 考虑使用 Kubernetes 进行编排

2. **高可用性**:
   - 实现服务的自动恢复机制
   - 配置数据库主从复制或集群
   - 考虑多区域部署以提高可用性

### 预热与性能调优

1. **系统预热**:
   - 启动时预加载嵌入模型
   - 执行常见查询预热缓存
   - 配置 Qdrant 和 Elasticsearch 索引优化

2. **性能参数调优**:
   - 根据硬件调整批处理大小
   - 优化缓存策略和大小
   - 调整并发连接数和线程池大小

---

通过本文档中详述的 API 和最佳实践，您可以将 Vanna MySQL Qdrant 扩展系统成功集成到生产环境中，并与 Dify 等工作流平台无缝对接，为用户提供高效、准确的自然语言到 SQL 转换服务。​​​​​​​​​​​​​​​​
