# 视频场景检测和帧抽取功能

本文档介绍如何使用新增的 PySceneDetect 功能来检测视频场景切换点并抽取关键帧。

## 功能概述

新增的功能可以：
1. 使用 PySceneDetect 自动检测视频中的场景切换点
2. 为每个检测到的场景抽取三个关键帧：
   - 第一帧（场景开始）
   - 中间帧（场景中点）
   - 最后一帧（场景结束前）
3. 可选择将帧保存为 JPEG 文件
4. 生成详细的场景信息 JSON 文件

## 安装依赖

确保已安装 PySceneDetect：

```bash
pip install scenedetect[opencv]
```

或者通过 requirements.txt 安装：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基本用法

```python
from aimage_supervision.utils.video_utils import detect_video_scenes_and_extract_frames

# 检测场景并抽取帧
result = detect_video_scenes_and_extract_frames(
    video_path="path/to/your/video.mp4",
    detector_type="adaptive",  # 可选: "adaptive", "content"
    output_dir="output_frames/"  # 可选: 保存帧图片的目录
)

print(f"检测到 {result['total_scenes']} 个场景")
for scene in result['scenes']:
    print(f"场景 {scene['scene_number']}: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s")
```

### 2. 获取特定场景的帧

```python
from aimage_supervision.utils.video_utils import get_scene_representative_frames

# 获取第1个场景的关键帧
frames = get_scene_representative_frames(
    video_path="path/to/your/video.mp4",
    scene_number=1,
    detector_type="adaptive"
)

# frames 包含三个 bytes 对象：
# - frames['first_frame']
# - frames['middle_frame'] 
# - frames['last_frame']
```

## 函数参数说明

### `detect_video_scenes_and_extract_frames`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `video_path` | str | 必填 | 视频文件路径 |
| `detector_type` | str | "adaptive" | 检测器类型：`"adaptive"` 或 `"content"` |
| `threshold` | float | None | 检测阈值（可选，使用默认值） |
| `output_dir` | str | None | 输出目录（可选，不提供则不保存文件） |

### `get_scene_representative_frames`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `video_path` | str | 必填 | 视频文件路径 |
| `scene_number` | int | 必填 | 场景编号（从1开始） |
| `detector_type` | str | "adaptive" | 检测器类型 |

## 检测器类型

### Adaptive Detector（推荐）
- 自适应内容检测器
- 适合处理有快速摄像机运动的视频
- 使用相邻帧变化的滚动平均值
- 默认阈值：3.0

### Content Detector
- 内容感知检测器
- 检测相邻帧之间的显著差异
- 适合检测快速切换的场景
- 默认阈值：27.0

## 返回数据结构

### `detect_video_scenes_and_extract_frames` 返回值

```python
{
    "scenes": [
        {
            "scene_number": 1,
            "start_time": 0.0,
            "end_time": 5.2,
            "duration": 5.2,
            "start_frame": 0,
            "end_frame": 156,
            "frame_count": 156,
            "first_frame_bytes": bytes,
            "middle_frame_bytes": bytes,
            "last_frame_bytes": bytes,
            # 如果提供了 output_dir，还会包含以下字段：
            "first_frame_path": "output/video_scene001_first.jpg",
            "middle_frame_path": "output/video_scene001_middle.jpg",
            "last_frame_path": "output/video_scene001_last.jpg"
        },
        # ... 更多场景
    ],
    "total_scenes": 5,
    "detector_used": "adaptive",
    "threshold_used": 3.0,
    # 如果提供了 output_dir，还会包含：
    "scenes_info_path": "output/video_scenes_info.json"
}
```

## 示例脚本

运行提供的示例脚本：

```bash
cd examples/
python scene_detection_example.py path/to/video.mp4 output_directory/
```

## 错误处理

函数会抛出以下异常：

- `ImportError`: 未安装 scenedetect 库
- `FileNotFoundError`: 视频文件不存在
- `ValueError`: 无效的检测器类型或场景编号
- `Exception`: 视频处理失败

## 性能考虑

- 大视频文件可能需要较长处理时间
- 建议先用较小的测试视频验证参数
- 可以调整阈值来优化检测结果
- 如果只需要帧数据不需要保存文件，不要提供 `output_dir` 参数

## API 端点

新增了以下 REST API 端点：

### 1. 分析视频场景 
`POST /api/v1/video/scenes/analyze`

**参数:**
- `file`: 视频文件 (multipart/form-data)
- `detector_type`: 检测器类型 ("adaptive" 或 "content")
- `threshold`: 检测阈值 (可选)

**响应:**
```json
{
  "scenes": [
    {
      "scene_number": 1,
      "start_time": 0.0,
      "end_time": 5.2,
      "duration": 5.2,
      "start_frame": 0,
      "end_frame": 156,
      "frame_count": 156
    }
  ],
  "total_scenes": 5,
  "detector_used": "adaptive",
  "threshold_used": 3.0
}
```

### 2. 获取场景关键帧
`POST /api/v1/video/scenes/{scene_number}/frames`

**参数:**
- `scene_number`: 场景编号 (路径参数)
- `file`: 视频文件 (multipart/form-data)
- `detector_type`: 检测器类型

**响应:**
```json
{
  "first_frame": "base64_encoded_image_data",
  "middle_frame": "base64_encoded_image_data", 
  "last_frame": "base64_encoded_image_data",
  "scene_number": 1
}
```

### 3. 获取场景关键帧和详细信息
`POST /api/v1/video/scenes/{scene_number}/frames-with-info`

**参数:**
- `scene_number`: 场景编号 (路径参数)
- `file`: 视频文件 (multipart/form-data)
- `detector_type`: 检测器类型
- `threshold`: 检测阈值 (可选)

**响应:**
```json
{
  "frames": {
    "first_frame": "base64_encoded_image_data",
    "middle_frame": "base64_encoded_image_data",
    "last_frame": "base64_encoded_image_data",
    "scene_number": 1
  },
  "scene_info": {
    "scene_number": 1,
    "start_time": 0.0,
    "end_time": 5.2,
    "duration": 5.2,
    "start_frame": 0,
    "end_frame": 156,
    "frame_count": 156
  }
}
```

### API 测试

使用提供的测试脚本：

```bash
cd examples/
python video_api_test.py path/to/video.mp4
```

**注意:** 在使用API前需要：
1. 获取有效的JWT认证令牌
2. 在请求头中包含 `Authorization: Bearer <token>`

## 示例用例

### 1. 视频内容分析 (Python SDK)
```python
# 快速分析视频内容结构
result = detect_video_scenes_and_extract_frames("movie.mp4")
for scene in result['scenes']:
    if scene['duration'] > 10:  # 找出长于10秒的场景
        print(f"长场景: {scene['scene_number']} ({scene['duration']:.1f}s)")
```

### 2. 生成视频预览 (Python SDK)
```python
# 为每个场景生成预览图
result = detect_video_scenes_and_extract_frames(
    "presentation.mp4", 
    output_dir="thumbnails/"
)
print(f"生成了 {result['total_scenes'] * 3} 张预览图")
```

### 3. 场景统计分析 (Python SDK)
```python
result = detect_video_scenes_and_extract_frames("video.mp4")
durations = [scene['duration'] for scene in result['scenes']]
print(f"平均场景长度: {sum(durations) / len(durations):.2f}s")
print(f"最短场景: {min(durations):.2f}s")
print(f"最长场景: {max(durations):.2f}s")
```

### 4. API 调用示例 (JavaScript)
```javascript
// 使用 JavaScript 调用场景分析 API
const formData = new FormData();
formData.append('file', videoFile);
formData.append('detector_type', 'adaptive');

const response = await fetch('/api/v1/video/scenes/analyze', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${authToken}`
    },
    body: formData
});

const result = await response.json();
console.log(`检测到 ${result.total_scenes} 个场景`);
``` 