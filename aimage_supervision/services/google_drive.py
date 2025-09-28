import io

import httpx

from aimage_supervision.clients.aws_s3 import upload_file_to_s3
from aimage_supervision.enums import AssetStatus
from aimage_supervision.models import Asset


async def download_file_from_google_drive(
    file_id: str,
    google_access_token: str,
) -> bytes:
    '''Google Drive からファイルをダウンロードする'''
    url = f'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media'
    headers = {'Authorization': f'Bearer {google_access_token}'}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        file_binary = response.content
        return file_binary


async def upload_file_from_google_drive_task(
    file_id: str,
    file_name: str,
    google_access_token: str,
) -> str:
    '''ファイルを Google Drive にアップロードする'''
    asset = await Asset.filter(drive_file_id=file_id).first()
    if not asset:
        raise Exception('Asset not found')
    try:
        file_binary = await download_file_from_google_drive(file_id, google_access_token)
    except:
        asset.status = AssetStatus.FAILED
        await asset.save()
        raise Exception('Failed to download file from Google Drive')
    file_binary_io = io.BytesIO(file_binary)
    await upload_file_to_s3(file_binary_io, asset.s3_path)
    asset.status = AssetStatus.PENDING
    await asset.save()
    return asset.s3_path
