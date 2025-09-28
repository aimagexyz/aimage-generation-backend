generate_ng_review_prompt = """
I want you to generate a prompt for me to detect the ng items.
Please generate a description for the ng item using the image provided, so that others can detect the object or anything similar to the object in other images.

visual_characteristics: visual characteristics that describe the object details. including shape, color, texture, size, etc.
key_considerations: what to look for when detecting the object. What does it possibly similar to? What's the situation that it usually appears in? etc.

## example
for example, if I want to detect the cross, the description should be:
### Tag: cross
· Key Visual Characteristics:
Basic Form: Two lines or bars intersecting, typically at a right angle.
Latin Cross: Vertical bar is longer than the horizontal bar, with the intersection above the midpoint of the vertical bar.
Greek Cross: Bars are of equal length, intersecting at their midpoints.
Impression: Can range from an explicit religious symbol to a simple geometric shape.
· Key Considerations & What to Look For:
Explicit religious symbols on characters, buildings (churches), or as objects (e.g., necklaces).
Stylized crosses:
Abstract patterns where lines clearly form a prominent cross shape.
Simple "+" signs if they are prominent and could be interpreted as a cross in context.
Decorative elements, window panes, sword hilts, or even misaligned sparkles (like the potential very faint suggestion in Image 2 with Ralts, though that example is extremely tenuous and likely not an NG – but illustrates thinking about abstract forms).
Distinguish from the Celtic Cross (see item 6) which has an additional circle.

### Tag:SS Mark (SSマーク - SS māku, Nazi symbol):
· Key Visual Characteristics:
Sig Runes: Composed of two stylized 'S' shapes that resemble lightning bolts (Armanen Sig runes).
Arrangement: Typically side-by-side.
Shape: Sharp, angular, distinct zig-zag form.
Impression: Aggressive, sharp, lightning-like.
· Key Considerations & What to Look For:
Explicit depiction of two side-by-side Sig runes.
Individual lightning bolt shapes that strongly resemble the characteristic angular, zig-zag form of a Sig rune (as highlighted in Image 4 with Plusle/Minun, where the feedback points out that even single or grouped lightning bolts looking like "SSマークに見えてしまう" – appearing as an SS mark – require modification).
Focus on the specific sharp, angular, "runic S" shape of the lightning bolt(s), not just any generic cartoon lightning. The zig-zags should be few and sharp.
"""


generate_visual_review_prompt = """
User's input will be about some visual details he want's to check in an image, and you will be offering the input and the image.
Please transform users' simple inputs into complete, professional, and more effective prompts.

Task Requirements:
1. Describe the characteristics in the image of the mentioned details from user input.
2. Break down the user's input into several steps that helps AI to understand how to check the details.
3. Maintain the original intent while adding necessary details and structure
4. return both English and Japanese

Transformation Principles:
- Be Specific and Clear: Avoid vague expressions, use concrete descriptions
- Structure Clearly: Organize the prompt's structure and logic reasonably
- Easy to Understand: Use clear language and formatting
- Function-Oriented: Ensure the prompt can effectively guide AI to complete tasks

Please analyze the original prompt and provide the transformation result.
"""

generate_settings_review_prompt = """
Please analyze the user's input and the image, and generate a prompt to check whether the image fits the settings.
"""

generate_design_review_prompt = """
Please analyze the user's input and the image, and generate the detailed points that need to be checked in the design, please be descriptive not abstract.
"""

generate_text_review_prompt = """
Please analyze the user's input and the image, and generate the detailed points that need to be checked in the text, please be descriptive not abstract.
"""

generate_copyright_review_prompt = """
Please recognize the words in the copyright mark from user's input or image and use the words to generate a prompt. The words might be detailed,
please be sure not to miss any words.
"""


prompt_group = {
    "general_ng_review": generate_ng_review_prompt,
    "visual_review": generate_visual_review_prompt,
    "settings_review": generate_settings_review_prompt,
    "design_review": generate_design_review_prompt,
    "text_review": generate_text_review_prompt,
    "copyright_review": generate_copyright_review_prompt
}


general_prompt_rewrite_prompt = """
You are a professional prompt writer. Please rewrite the provided prompt in a more professional and detailed way: 
The prompt is used for {type} review. 

Context information: {context_info}

Please don't miss anything mentioned in the user's input. Context information only provide the direction of rewrite.
"""