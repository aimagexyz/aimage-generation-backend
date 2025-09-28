from aimage_supervision.clients.aws_s3 import get_s3_url_from_path


async def get_image(user_id, s3_path, annotation):
    if s3_path.startswith('s3://'):
        parts = s3_path.split('/', 3)
        if len(parts) > 3:
            s3_path = parts[3]
    url = await get_s3_url_from_path(s3_path)
    return url  # avoid to use cv2 in AWS App runner.