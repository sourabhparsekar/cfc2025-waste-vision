from fastapi import APIRouter, File, UploadFile, HTTPException
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


router = APIRouter()

# Initialize the Gemini client
client = genai.Client()

IMAGE_ANALYSIS_SYS_PROMPT = """
You are an expert in analysing images to identify items that are potentially garbage or waste, or items that would end up as garbage or waste at the end of its lifecycle.
These items need to be recycled or replaced with sustainable alternatives.
Identify each item present along with percentage of the item that makes up the total items seen in the image.
"""

IMAGE_ANALYSIS_USER_PROMPT = "Analyse the given image and identify the items"


class ItemsIdentified(BaseModel):
    items: list[str] = Field(..., description="List of items identified in the image")
    percentages: list[float] = Field(
        ..., description="Percentage of each item in the image"
    )


@router.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
) -> ItemsIdentified:
    """
    Upload an image and analyze it with Vision Model
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}",
        )

    try:
        # Read the uploaded file as bytes
        image_bytes = await file.read()

        # Generate content using Gemini
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
                IMAGE_ANALYSIS_USER_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=IMAGE_ANALYSIS_SYS_PROMPT,
                response_mime_type="application/json",
                response_schema=ItemsIdentified,
            ),
        )
        parsed_response: ItemsIdentified = response.parsed

        return parsed_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Close the file
        await file.close()
