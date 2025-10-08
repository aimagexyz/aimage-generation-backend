import io
import uuid
from typing import Annotated, List

from fastapi import (APIRouter, File, HTTPException, Security, UploadFile,
                     status)
from fastapi_pagination import Page

from aimage_supervision.clients.aws_s3 import (get_s3_url_from_path,
                                               upload_file_to_s3)
from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.middlewares.tortoise_paginate import tortoise_paginate
from aimage_supervision.models import (Document, Project, ProjectIn,
                                       ProjectOut, ProjectSimpleOut, User)
from aimage_supervision.settings import logger

router = APIRouter(prefix='', tags=['Projects'])

# Constants for image upload
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_CONTENT_TYPE_PREFIX = "image/"


@router.post('/projects')
async def create_project(
    user: Annotated[User, Security(get_current_user)],
    payload: ProjectIn,
) -> ProjectOut:
    project_name: str = payload.name
    description: str | None = payload.description

    existed = await Project.filter(owner=user, name=project_name).exists()

    if existed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Project with this name already exists',
        )

    project = await Project.create(
        owner=user,
        name=project_name,
        description=description,
    )

    response = await ProjectOut.from_tortoise_orm(project)
    return response


@router.get('/projects')
async def list_projects(
    user: Annotated[User, Security(get_current_user)],
) -> Page[ProjectSimpleOut]:
    # 使用优化的查询，只获取 id 和 name 字段
    projects_queryset = Project.of_user_simple(user)
    response = await tortoise_paginate(
        projects_queryset,
        model=ProjectSimpleOut,  # type: ignore
    )
    return response


@router.get('/projects/{project_id}')
async def get_project(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
) -> ProjectOut:
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found',
        )

    response = await ProjectOut.from_tortoise_orm(project)
    return response


@router.delete('/projects/{project_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_or_quit_project(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
):
    project = await Project.get_or_none(id=project_id)
    if not project or not project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found',
        )

    if project.is_user_owner(user):
        # Delete project
        await project.delete()
        return None
    # Quit project
    if project.is_user_in_coop_members(user):
        await project.coop_members.remove(user)
        return None

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail='You cannot quit this project, because your organization is participating in it',
    )


@router.post("/projects/{project_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload_project_documents(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    file: UploadFile = File(..., description="document file"),
):
    # verify project permission
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or no access permission",
        )

    # upload file to s3
    cloud_file_path = f"projects/{project_id}/documents/{file.filename}"

    # check if file already exists

    await upload_file_to_s3(file.file, cloud_file_path=cloud_file_path)

    # check if file already exists in database
    document = await Document.get_or_none(file_path=cloud_file_path)
    if document:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File already exists",
        )
    # save file metadata to database.
    document = await Document.create(
        project=project,
        file_path=cloud_file_path,
        file_name=file.filename,
        file_size=file.size,
        file_type=file.content_type,
    )

    logger.info(f"Uploaded {file.filename} documents to project {project_id}")

    return status.HTTP_201_CREATED


@router.get("/projects/{project_id}/documents/{document_id}/url")
async def get_document_url(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    document_id: str,
) -> dict[str, str]:
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    document = await Document.get_or_none(id=document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    await document.fetch_s3_url()

    return {"file_url": document.file_url()}
