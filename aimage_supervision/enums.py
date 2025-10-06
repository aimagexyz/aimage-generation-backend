from enum import Enum


class UserRole(str, Enum):
    USER = 'user'
    ADMIN = 'admin'


class SubtaskStatus(str, Enum):
    PENDING = 'pending'
    DENIED = 'denied'
    ACCEPTED = 'accepted'


class SubtaskType(str, Enum):
    PICTURE = 'picture'
    VIDEO = 'video'
    TEXT = 'text'
    AUDIO = 'audio'
    WORD = 'word'
    EXCEL = 'excel'


class MarkedType(str, Enum):
    CIRCLE = 'circle'
    LINE = 'line'
    TEXT = 'text'
    SELECT_TEXT = 'select_text'
    AUDIO = 'audio'
    VIDEO = 'video'


class AssetStatus(str, Enum):
    UPLOADING = 'uploading'
    PENDING = 'pending'
    PROCESSING = 'processing'
    DONE = 'done'
    FAILED = 'failed'


class AIClassificationStatus(str, Enum):
    UNCLASSIFIED = 'unclassified'
    IN_PROGRESS = 'in_progress'
    CLASSIFIED = 'classified'


class BatchJobStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class AiReviewMode(str, Enum):
    QUALITY = 'quality'
    SPEED = 'speed'


class AiReviewProcessingStatus(str, Enum):
    PENDING = "pending"        # 刚创建，等待处理
    PROCESSING = "processing"  # 正在处理中
    COMPLETED = "completed"    # 处理完成
    FAILED = "failed"         # 处理失败
    CANCELLED = "cancelled"    # 用户中断
