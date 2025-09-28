import asyncio
import datetime
from aimage_supervision.settings import logger
from uuid import UUID

import httpx
from tortoise.transactions import in_transaction

from aimage_supervision.enums import AIClassificationStatus
# Assuming User might be needed later for history or created_by
from aimage_supervision.models import Subtask, User
# To type hint content and history
from aimage_supervision.schemas import SubtaskContent
from aimage_supervision.settings import \
    AWS_BUCKET_NAME  # 添加 AWS_BUCKET_NAME 导入
from aimage_supervision.settings import (SUPERVISION_BATCH_API_PASSWORD,
                                         SUPERVISION_BATCH_API_URL,
                                         SUPERVISION_BATCH_API_USER)


# Placeholder for database dependency, adjust if your project uses a different pattern
# For example, if you have a specific get_db or similar utility


async def get_db_transaction():
    # This is a simplified example. In a real scenario, you'd integrate
    # with your project's transaction management (e.g., Tortoise's `atomic` or `in_transaction`).
    # For now, we'll assume the function using this will be wrapped in `in_transaction`.
    pass


async def process_subtask_ai_classification(
    sub_task_id: UUID,
    # db: AsyncTransactionWrapper = Depends(get_db_transaction) # Correct dependency might be different
) -> None:
    """
    Processes AI classification for a given subtask.
    Updates status, simulates AI processing, updates content with a mock S3 URL,
    and manages content versioning in history.
    """
    processed_relative_s3_path = None
    original_relative_s3_path = None  # Store original path for logging/fallback

    try:
        # First, fetch necessary info from subtask (outside main transaction if possible, or minimal transaction)
        # For this refactor, we'll fetch initial s3_path and then do AI call
        # then start the main transaction for updates.

        subtask_for_info = await Subtask.get_or_none(id=sub_task_id)
        if not subtask_for_info:
            logger.error(
                f"Subtask with ID {sub_task_id} not found for AI classification.")
            return

        if subtask_for_info.content and hasattr(subtask_for_info.content, 's3_path'):
            original_relative_s3_path = subtask_for_info.content.s3_path
            processed_relative_s3_path = original_relative_s3_path  # Default to original

        if original_relative_s3_path and AWS_BUCKET_NAME:
            full_s3_path = f's3://{AWS_BUCKET_NAME}/{original_relative_s3_path}'
            try:
                async with httpx.AsyncClient() as client:
                    api_url = f'{SUPERVISION_BATCH_API_URL}/api/v1/ai/process-file'
                    params = {'s3_path': full_s3_path}
                    auth = (SUPERVISION_BATCH_API_USER,
                            SUPERVISION_BATCH_API_PASSWORD)

                    logger.info(
                        f"Calling AI processing API: {api_url} with s3_path: {full_s3_path}")
                    # 10 minutes timeout, adjusted from 600.0 based on previous context
                    response = await client.get(api_url, params=params, auth=auth, timeout=1000.0)
                    response.raise_for_status()

                    processed_s3_path_from_api = response.text
                    if processed_s3_path_from_api and processed_s3_path_from_api.startswith(f's3://{AWS_BUCKET_NAME}/'):
                        processed_relative_s3_path = processed_s3_path_from_api[len(
                            f's3://{AWS_BUCKET_NAME}/'):]
                        logger.info(
                            f"Received processed S3 path: {processed_s3_path_from_api} from AI API. Storing relative path: {processed_relative_s3_path}")
                    else:
                        logger.warning(
                            f"AI processing API returned an invalid S3 path: {processed_s3_path_from_api}. Using original relative S3 path: {original_relative_s3_path}")
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error calling AI processing API for {full_s3_path}: {e.response.status_code} - {e.response.text}", exc_info=True)
                # Fallback to original path, status will be updated accordingly later
            except Exception as e:  # Includes httpx.ReadTimeout
                logger.error(
                    f"Error calling AI processing API for {full_s3_path}: {e}", exc_info=True)
                # Fallback to original path
        else:
            if not original_relative_s3_path:
                logger.warning(
                    f"Subtask {sub_task_id} has no s3_path in content. Skipping AI processing API call.")
            elif not AWS_BUCKET_NAME:
                logger.warning(
                    f"AWS_BUCKET_NAME is not provided. Skipping AI processing API call for subtask {sub_task_id}.")

        # Now, start the database transaction for all database updates
        async with in_transaction() as connection:
            # Re-fetch inside transaction
            subtask = await Subtask.get_or_none(id=sub_task_id, using_db=connection)

            if not subtask:  # Should not happen if subtask_for_info was found, but good practice
                logger.error(
                    f"Subtask with ID {sub_task_id} disappeared before transaction.")
                return

            # 1. Update status to IN_PROGRESS (or directly to CLASSIFIED/FAILED if AI call was made)
            # If AI call was attempted, we might skip IN_PROGRESS if it was quick
            # For simplicity and robustness, we'll always set IN_PROGRESS first then CLASSIFIED
            subtask.ai_classification_status = AIClassificationStatus.IN_PROGRESS
            # Save only status first
            await subtask.save(using_db=connection, update_fields=['ai_classification_status'])
            logger.info(
                f"AI classification status for Subtask ID: {sub_task_id} set to IN_PROGRESS.")

            if subtask.content:  # Ensure content exists before trying to update it
                if subtask.history is None:
                    subtask.history = []

                # Create history from current content *before* updating it
                content_history_snapshot = SubtaskContent.model_validate(
                    subtask.content)
                subtask.history.append(
                    content_history_snapshot.model_dump())  # type: ignore

                # Update current content with new values
                # Ensure all required fields for SubtaskContent are present
                current_content_dict = dict(subtask.content)
                current_content_dict.update({
                    # Use the path determined (original or processed)
                    's3_path': processed_relative_s3_path,
                    'created_at': datetime.datetime.now().isoformat(),  # Update timestamp
                    # Ensure title, description, task_type are present or provide defaults
                    'title': current_content_dict.get('title', subtask.name),
                    'description': current_content_dict.get('description', subtask.description or ''),
                    'task_type': current_content_dict.get('task_type', subtask.task_type),
                })

                content_model = SubtaskContent(**current_content_dict)
                subtask.content = content_model.model_dump()  # type: ignore

                # 4. Update status to CLASSIFIED (or a FAILED status if API call failed and we want to reflect that)
                # Assuming success or fallback to original path is "classified"
                subtask.ai_classification_status = AIClassificationStatus.CLASSIFIED
            else:
                # If there was no initial content, we might want to create it here
                # For now, log a warning if this state is unexpected
                logger.warning(
                    f"Subtask {sub_task_id} had no content object to update after AI processing call.")

            # Save all changes (content, history, final status)
            await subtask.save(using_db=connection)
            logger.info(
                f"AI classification finished for Subtask ID: {sub_task_id}. Status set to CLASSIFIED. S3 Path: {processed_relative_s3_path}")

    except Exception as e:
        logger.error(
            f"Outer error during AI classification for Subtask ID {sub_task_id}: {e}", exc_info=True)
        # Attempt to set a FAILED status if possible, outside the potentially failed transaction
        try:
            async with in_transaction() as conn_fail:  # New transaction for failure update
                subtask_fail = await Subtask.get_or_none(id=sub_task_id, using_db=conn_fail)
                if subtask_fail:
                    # Or a dedicated FAILED status
                    subtask_fail.ai_classification_status = AIClassificationStatus.UNCLASSIFIED
                    await subtask_fail.save(using_db=conn_fail, update_fields=['ai_classification_status'])
                    logger.info(
                        f"Set AI classification status to UNCLASSIFIED for Subtask {sub_task_id} due to error.")
        except Exception as e_fail_update:
            logger.error(
                f"Failed to update subtask {sub_task_id} status to FAILED/UNCLASSIFIED: {e_fail_update}", exc_info=True)
        # The original transaction will be rolled back if an error occurred within it.
        # The pass here allows the background task to complete without re-raising,
        # as FastAPI's BackgroundTask might not handle re-raised exceptions well.
        pass
