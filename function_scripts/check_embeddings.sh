# 设置环境变量
export DATABASE_URL=""
export GEMINI_API_KEY=""
export GOOGLE_CREDS=''
export VERTEX_AI_PROJECT=""
export VERTEX_AI_LOCATION=""

export AWS_ACCESS_KEY_ID=""
export AWS_BUCKET_NAME=""
export AWS_REGION="ap-northeast-1"
export AWS_SECRET_ACCESS_KEY=""

# 运行脚本
python function_scripts/generate_embeddings_standalone.py --project-id "0db99f96-1722-48c1-9a25-27971a3ff9f5"