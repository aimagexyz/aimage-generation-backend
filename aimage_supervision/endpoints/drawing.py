"""
Drawing annotation endpoints for handwritten corrections/annotations on images
"""
import uuid
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Security, status
from pydantic import BaseModel

from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.models import Subtask, User
from aimage_supervision.schemas import SubtaskAnnotation

router = APIRouter(prefix='/drawing', tags=['Drawing'])


class DrawingData(BaseModel):
    """Drawing data from fabric.js canvas"""
    drawing_data: str  # JSON string from fabric.js
    
class DrawingResponse(BaseModel):
    """Response after saving drawing"""
    id: str
    subtask_id: str
    drawing_data: str  # Added field to return the actual drawing data
    created_at: str
    

@router.get('/subtasks/{subtask_id}')
async def get_subtask_drawing(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
) -> Optional[DrawingResponse]:
    """Get the drawing annotation for a subtask if it exists."""
    
    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Subtask not found',
        )
    
    # Look for drawing annotation in the annotations array
    if subtask.annotations:
        for annotation in subtask.annotations:
            # annotation は dict として保存されているので、直接アクセスできる
            if isinstance(annotation, dict):
                if annotation.get('tool') == 'pen' and annotation.get('drawing_data'):
                    return DrawingResponse(
                        id=annotation.get('id'),
                        subtask_id=subtask_id,
                        drawing_data=annotation.get('drawing_data'),
                        created_at=annotation.get('timestamp', '')
                    )
    
    return None


@router.put('/subtasks/{subtask_id}')
async def save_subtask_drawing(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
    payload: DrawingData,
) -> DrawingResponse:
    """Save or update drawing annotation for a subtask."""
    
    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Subtask not found',
        )
    
    if not subtask.annotations:
        subtask.annotations = []
    
    # Remove existing drawing annotation if any
    subtask.annotations = [
        a for a in subtask.annotations 
        if not (
            (isinstance(a, dict) and a.get('tool') == 'pen' and a.get('drawing_data')) or
            (hasattr(a, 'tool') and a.tool == 'pen' and hasattr(a, 'drawing_data') and a.drawing_data)
        )
    ]
    
    # Create new drawing annotation
    drawing_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    drawing_annotation = SubtaskAnnotation(
        id=drawing_id,
        type='annotation',
        tool='pen',
        text='手書き修正',  # "Handwritten correction" in Japanese
        drawing_data=payload.drawing_data,
        author=str(user.id),
        timestamp=timestamp,
        version=subtask.version(),
    )
    
    subtask.annotations.append(drawing_annotation.model_dump())  # type: ignore
    await subtask.save()
    
    return DrawingResponse(
        id=drawing_id,
        subtask_id=subtask_id,
        drawing_data=payload.drawing_data,
        created_at=timestamp
    )


@router.delete('/subtasks/{subtask_id}')
async def delete_subtask_drawing(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
) -> dict:
    """Delete drawing annotation from a subtask."""
    
    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Subtask not found',
        )
    
    if subtask.annotations:
        # Remove drawing annotation
        original_count = len(subtask.annotations)
        subtask.annotations = [
            a for a in subtask.annotations 
            if not (
                (isinstance(a, dict) and a.get('tool') == 'pen' and a.get('drawing_data')) or
                (hasattr(a, 'tool') and a.tool == 'pen' and hasattr(a, 'drawing_data') and a.drawing_data)
            )
        ]
        
        if len(subtask.annotations) < original_count:
            await subtask.save()
            return {"message": "Drawing annotation deleted successfully"}
    
    return {"message": "No drawing annotation found"}