# Embeddings生成器独立脚本

这个独立的Python脚本可以用来检测和生成项目中items的embeddings，无需启动完整的Web服务。

## 功能

- 🔍 **检测功能**: 检测项目中哪些items缺少embeddings
- 🚀 **生成功能**: 为缺少embeddings的items生成向量embeddings
- 📊 **统计报告**: 提供详细的处理结果统计
- ⚡ **高效处理**: 支持并发处理以提高效率

## 环境要求

### 必需的环境变量

在运行脚本前，请确保设置以下环境变量：

```bash
# 数据库连接
export DATABASE_URL="postgresql://username:password@host:port/database"

# Google AI服务
export GEMINI_API_KEY="your-gemini-api-key"
export GOOGLE_CREDS='{"type": "service_account", ...}'  # JSON格式的Google Cloud认证信息
export VERTEX_AI_PROJECT="your-vertex-ai-project-id"
export VERTEX_AI_LOCATION="us-central1"  # 可选，默认值

# 可选配置
export TEXT_EMBEDDING_MODEL="text-multilingual-embedding-002"  # 默认值
```

### Python依赖

脚本会自动使用项目现有的依赖，确保你在项目虚拟环境中运行：

```bash
source venv/bin/activate  # 激活虚拟环境
```

## 使用方法

### 基本用法

```bash
# 为整个项目检测并生成embeddings
python generate_embeddings_standalone.py --project-id "your-project-id"

# 指定数据库URL（如果没有设置环境变量）
python generate_embeddings_standalone.py --project-id "your-project-id" --database-url "postgresql://..."
```

### 高级用法

```bash
# 仅检测缺少embeddings的items（不生成）
python generate_embeddings_standalone.py --project-id "your-project-id" --check-only

# 为特定的items生成embeddings
python generate_embeddings_standalone.py --project-id "your-project-id" --item-ids "item-id-1" "item-id-2" "item-id-3"

# 仅检测特定items
python generate_embeddings_standalone.py --project-id "your-project-id" --item-ids "item-id-1" "item-id-2" --check-only
```

## 输出示例

### 检测阶段输出

```
2025-01-20 10:30:15 - INFO - 数据库连接初始化成功
2025-01-20 10:30:16 - INFO - Vecs客户端初始化成功
2025-01-20 10:30:16 - INFO - ItemsVectorService初始化成功 (项目ID: project-123)
2025-01-20 10:30:17 - INFO - 开始检测缺少embeddings的items...
2025-01-20 10:30:20 - INFO - 检测结果:
2025-01-20 10:30:20 - INFO -   - 总计items: 150
2025-01-20 10:30:20 - INFO -   - 缺少embeddings的items: 25
2025-01-20 10:30:20 - INFO -   - 缺少图片embeddings: 15
2025-01-20 10:30:20 - INFO -   - 缺少文本embeddings: 10
```

### 生成阶段输出

```
2025-01-20 10:30:21 - INFO - 发现 25 个items缺少embeddings，开始生成...
2025-01-20 10:30:21 - INFO - 开始生成embeddings...
2025-01-20 10:32:45 - INFO - 生成结果:
2025-01-20 10:32:45 - INFO -   - 处理的items总数: 25
2025-01-20 10:32:45 - INFO -   - 成功生成图片embeddings: 15
2025-01-20 10:32:45 - INFO -   - 成功生成文本embeddings: 10
2025-01-20 10:32:45 - INFO -   - 失败的items: 0
2025-01-20 10:32:45 - INFO - ✅ Embeddings检测和生成完成!
```

## 命令行参数

| 参数 | 必需 | 描述 |
|------|------|------|
| `--project-id` | ✅ | 项目ID |
| `--database-url` | ❌ | 数据库连接URL（也可通过环境变量设置） |
| `--item-ids` | ❌ | 要处理的特定item ID列表（空格分隔） |
| `--check-only` | ❌ | 仅检测模式，不生成embeddings |

## 错误处理

脚本包含完善的错误处理机制：

- **环境变量检查**: 启动前验证所有必需的环境变量
- **数据库连接**: 自动处理数据库连接错误
- **资源清理**: 确保程序退出时正确关闭数据库连接
- **详细日志**: 提供详细的错误信息和堆栈跟踪

## 性能考虑

- 脚本使用异步处理以提高效率
- 内置并发控制（默认最多4个并发任务）
- 支持批量处理以减少API调用次数
- 自动处理大型项目的内存管理

## 故障排除

### S3访问诊断工具

如果遇到S3文件下载问题（如404错误），可以使用专门的诊断工具：

```bash
# 基本S3连接测试
python debug_s3_access.py

# 测试特定文件
python debug_s3_access.py --test-file "your-s3-file-path"
```

### 常见问题

1. **数据库连接失败**
   ```
   确保DATABASE_URL格式正确：postgresql://username:password@host:port/database
   ```

2. **S3文件下载失败（404错误）**
   ```bash
   # 可能的原因：
   # 1. AWS凭证配置错误
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export AWS_REGION="your-region"
   export AWS_BUCKET_NAME="your-bucket"
   
   # 2. 文件路径不正确
   # 检查数据库中的s3_path字段是否正确
   
   # 3. 权限不足
   # 确保AWS用户有s3:GetObject权限
   
   # 4. 存储桶区域不匹配
   # 确保AWS_REGION与实际存储桶区域一致
   ```

3. **Google API认证失败**
   ```
   检查GOOGLE_CREDS环境变量是否包含有效的JSON格式认证信息
   确保VERTEX_AI_PROJECT设置正确
   ```

4. **权限错误**
   ```
   确保Google Cloud服务账户有访问Vertex AI和Gemini API的权限
   确保AWS用户有访问S3存储桶的权限
   ```

5. **内存不足**
   ```
   对于大型项目，考虑分批处理，使用--item-ids参数指定要处理的items子集
   ```

### S3权限配置示例

确保你的AWS用户或角色有以下权限：

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name",
                "arn:aws:s3:::your-bucket-name/*"
            ]
        }
    ]
}
```

## 与原始endpoint的对比

| 特性 | Web Endpoint | 独立脚本 |
|------|-------------|----------|
| 用户认证 | ✅ 需要 | ❌ 不需要 |
| 后台任务 | ✅ 异步 | ✅ 直接执行 |
| 进度跟踪 | ✅ 任务ID | ✅ 实时日志 |
| 批量处理 | ✅ 支持 | ✅ 支持 |
| 独立运行 | ❌ 需要服务 | ✅ 完全独立 |
| 调试友好 | ❌ 较难 | ✅ 详细输出 |

这个独立脚本非常适合：
- 数据维护任务
- 批量处理
- 调试和测试
- 自动化脚本集成
