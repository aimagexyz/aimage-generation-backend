from aimage_supervision.settings import logger
import subprocess
import tempfile
import shutil
from pathlib import Path


def convert_pptx_to_pdf(pptx_bytes: bytes) -> bytes:
    """
    Converts PPTX file bytes to PDF file bytes using LibreOffice.

    Args:
        pptx_bytes: The byte content of the PPTX file.

    Returns:
        The byte content of the converted PDF file.

    Raises:
        FileNotFoundError: If the PDF file is not created by LibreOffice.
        Exception: For any other errors during the conversion process.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        pptx_path = temp_dir_path / "presentation.pptx"

        # Write the PPTX bytes to a temporary file
        with open(pptx_path, "wb") as f:
            f.write(pptx_bytes)

        logger.info(f"Converting PPTX to PDF in temporary directory: {temp_dir}")

        try:
            # Determine which LibreOffice executable is available ("soffice" or "libreoffice")
            libreoffice_executable = shutil.which("soffice") or shutil.which("libreoffice")

            if not libreoffice_executable:
                logger.error(
                    "Neither 'soffice' nor 'libreoffice' command was found. Is LibreOffice installed in the environment?"
                )
                raise FileNotFoundError("LibreOffice executable not found in PATH.")

            # Execute the LibreOffice conversion command
            subprocess.run(
                [
                    libreoffice_executable,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(temp_dir_path),
                    str(pptx_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,  # Add a timeout to prevent hanging
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"LibreOffice conversion failed. STDERR: {e.stderr}")
            raise Exception(f"PDF conversion failed: {e.stderr}")
        except FileNotFoundError:
            logger.error(
                "The 'soffice' command was not found. Is LibreOffice installed in the environment?"
            )
            raise
        except subprocess.TimeoutExpired:
            logger.error("LibreOffice conversion timed out after 60 seconds.")
            raise Exception("PDF conversion timed out.")

        pdf_path = temp_dir_path / "presentation.pdf"

        if not pdf_path.exists():
            raise FileNotFoundError("Converted PDF file not found after LibreOffice execution.")

        # Read the converted PDF bytes
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        logger.info("Successfully converted PPTX to PDF.")
        return pdf_bytes
