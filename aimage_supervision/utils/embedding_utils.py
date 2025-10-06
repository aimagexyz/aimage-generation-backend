from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from tqdm import tqdm
from vertexai.vision_models import Image

from aimage_supervision.clients.google import multimodal_embedding_model, text_embedding_model
from aimage_supervision.settings import (EMBEDDING_BATCH_SIZE, IMAGE_EMBEDDING_BATCH_SIZE, IMAGE_EMBEDDING_DIM,
                      IMAGE_EMBEDDING_WORKERS, logger)


# --- Embedding Functions (Matching Notebook Logic) ---
def get_text_embeddings(texts: List[str]):
    """Get text embeddings using Vertex AI (matching notebook batching)."""
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []

    all_embeddings = []
    try:
        # Process in batches as per notebook example (batch size from settings)
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            # Use the globally initialized text_embedding_model
            response = text_embedding_model.get_embeddings(texts=batch)
            # response should be a list of embeddings
            all_embeddings.extend(response)
        # Return the list of embedding objects
        return all_embeddings
    except Exception as e:
        return []  # Return empty list on error


def get_image_embedding(image_input):
    """Get image embedding using Vertex AI multimodal model.

    Args:
        image_input: Either a file path string or a file-like object (e.g., from Streamlit's file_uploader)

    Returns:
        The image embedding vector or None if there was an error
    """
    try:
        from vertexai.vision_models import Image

        # Handle either a file path or a file-like object (e.g., from Streamlit's file_uploader)
        if hasattr(image_input, 'read'):
            # It's a file-like object
            file_bytes = image_input.read()
            # Reset file pointer for potential future use
            image_input.seek(0)
            image = Image(image_bytes=file_bytes)
        else:
            # Assume it's a file path
            with open(image_input, 'rb') as image_file:
                image = Image(image_bytes=image_file.read())

        # Use the globally initialized multimodal_embedding_model
        embedding = multimodal_embedding_model.get_embeddings(
            image=image,
            dimension=IMAGE_EMBEDDING_DIM,  # Use constant
        )

        # Return the specific image_embedding attribute as in the notebook
        if hasattr(embedding, 'image_embedding'):
            # Ensure we're returning a list, not None
            return embedding.image_embedding if embedding.image_embedding is not None else []
        else:
            return []
    except FileNotFoundError:
        logger.error(f"File not found: {image_input}")
        return []
    except Exception as e:
        logger.error(f"Error in get_image_embedding: {str(e)}")
        return []


def get_averaged_embedding(text_embedding, image_embedding):
    """Get averaged embedding using Vertex AI multimodal model."""
    return np.mean([text_embedding, image_embedding], axis=0)


def get_multimodal_embedding(text_input, image_input):
    """Get text embedding using Vertex AI multimodal model."""
    try:
        if hasattr(image_input, 'read'):
            # It's a file-like object
            file_bytes = image_input.read()
            # Reset file pointer for potential future use
            image_input.seek(0)
            image = Image(image_bytes=file_bytes)
        else:
            # Assume it's a file path
            with open(image_input, 'rb') as image_file:
                image = Image(image_bytes=image_file.read())

        embeddings = multimodal_embedding_model.get_embeddings(
            image=image,
            contextual_text=text_input,
            dimension=1408,
        )
        text_embedding = embeddings.text_embedding
        image_embedding = embeddings.image_embedding
        averaged_embedding = get_averaged_embedding(
            text_embedding, image_embedding)

        return averaged_embedding
    except Exception as e:
        logger.error(f"Error in get_text_embedding_multimodal: {str(e)}")
        return []

# --- DataFrame Embedding Functions (Matching Notebook Logic) ---


def generate_text_embeddings_df(df, output_pkl_path, column_name='reviewcomment'):
    """Generates text embeddings for 'reviewcomment' and saves DataFrame (matching notebook)."""
    column_values = df[column_name].tolist()

    # Call the batch embedding function
    embeddings_list = get_text_embeddings(column_values)

    if len(embeddings_list) != len(df):
        logger.error(
            f"Error: Mismatch between number of comments ({len(df)}) and generated embeddings ({len(embeddings_list)}).")
        return None  # Indicate error

    # Extract the .values attribute from each embedding object, as done in the notebook
    try:
        df[f"{column_name}_embedding"] = [emb.values for emb in embeddings_list]
    except AttributeError:
        logger.error(
            "Error: Embedding objects do not have 'values' attribute. Check Vertex AI SDK version/response.")
        # Fallback: Store the whole object or handle differently
        # df["reviewcomment_embedding"] = embeddings_list
        return None
    except Exception as e:
        logger.error(f"Error assigning embeddings to DataFrame: {e}")
        return None

    logger.info(f"Saving DataFrame with text embeddings to: {output_pkl_path}")
    try:
        df.to_pickle(output_pkl_path)
        logger.info("DataFrame saved.")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {output_pkl_path}: {e}")

    return df


def generate_image_embeddings_df(df, output_pkl_path, image_column_name='url', max_workers=IMAGE_EMBEDDING_WORKERS):
    """Generates image embeddings concurrently and saves DataFrame (matching notebook)."""
    logger.info(
        f"Generating image embeddings for {len(df)} images using {max_workers} workers...")
    image_files = df[image_column_name].tolist()
    total_files = len(image_files)
    image_embeddings = [None] * total_files  # Initialize results list

    # Inner function to process one image
    def process_image(args):
        idx, file_path = args
        try:
            # get_image_embedding returns the embedding vector directly or None
            embedding = get_image_embedding(file_path)
            if embedding is not None and len(embedding) > 0:
                return idx, True, embedding
            else:
                # Embedding function already prints errors, just return failure
                return idx, False, f"Embedding failed for {file_path}"
        except Exception as e:
            # Catch unexpected errors during the call itself
            error_msg = f"Error in process_image for {file_path}: {str(e)}"
            logger.error(error_msg)
            return idx, False, error_msg

    tasks = list(enumerate(image_files))
    processed_count = 0
    failed_indices = []

    # Use ThreadPoolExecutor and tqdm as in notebook
    with tqdm(total=total_files, desc="Image Embeddings") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_image, task) for task in tasks]

            for future in futures:
                try:
                    idx, success, result = future.result()
                    if success:
                        image_embeddings[idx] = result
                    else:
                        logger.warning(
                            f"Task failed for index {idx}: {result}")
                        failed_indices.append(idx)
                    processed_count += 1
                    pbar.update(1)

                    # Checkpointing logic from notebook (optional but good for long runs)
                    # Note: Notebook checkpointing saves based on 'i' in a loop,
                    # here we use processed_count. Adjust batch_size if needed.
                    checkpoint_batch_size = 10 * IMAGE_EMBEDDING_BATCH_SIZE  # Example size
                    if processed_count > 0 and processed_count % checkpoint_batch_size == 0:
                        try:
                            temp_df = df.copy()
                            # Assign potentially incomplete embeddings
                            temp_df['image_embedding'] = image_embeddings
                            checkpoint_path = f"{output_pkl_path}_checkpoint_{processed_count}.pkl"
                            logger.info(
                                f"Saving checkpoint ({processed_count}/{total_files}) to {checkpoint_path}")
                            temp_df.to_pickle(checkpoint_path)
                        except Exception as cp_error:
                            logger.warning(
                                f"Warning: Failed to save checkpoint: {cp_error}")

                except Exception as e:
                    # Error retrieving result from future (less common)
                    logger.critical(
                        f"Critical error retrieving future result: {e}")
                    pbar.update(1)  # Still update progress

    # Assign results to DataFrame
    df['image_embedding'] = image_embeddings

    # Handle failures - Optional: remove rows with None embeddings
    if failed_indices:
        logger.warning(
            f"Warning: {len(failed_indices)} images failed embedding generation.")
        # Option 1: Keep rows with None (as notebook likely does implicitly)
        # Option 2: Drop rows with failed embeddings
        # df.drop(failed_indices, inplace=True)
        # logger.info(f"Removed {len(failed_indices)} rows with failed embeddings.")
        pass  # Keep rows with None for now

    logger.info(
        f"Saving final DataFrame with image embeddings to: {output_pkl_path}")
    try:
        df.to_pickle(output_pkl_path)
        logger.info("DataFrame saved.")
    except Exception as e:
        logger.error(f"Error saving final DataFrame to {output_pkl_path}: {e}")

    return df
