import base64
import io
import json
import os
import re

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps


# =========================================================
# STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Instagram Carousel Machine",
    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

LOGO_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "logo.png"
)

FONT_PATH = os.path.join(
    BASE_DIR,
    "fonts",
    "Anton-Regular.ttf"
)


# =========================================================
# CLOUDFLARE SECRETS
# =========================================================

def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, "")


CLOUDFLARE_ACCOUNT_ID = get_secret(
    "CLOUDFLARE_ACCOUNT_ID"
)

CLOUDFLARE_API_TOKEN = get_secret(
    "CLOUDFLARE_API_TOKEN"
)


if not CLOUDFLARE_ACCOUNT_ID:
    st.error(
        "CLOUDFLARE_ACCOUNT_ID is missing from Streamlit Secrets."
    )
    st.stop()


if not CLOUDFLARE_API_TOKEN:
    st.error(
        "CLOUDFLARE_API_TOKEN is missing from Streamlit Secrets."
    )
    st.stop()


# =========================================================
# CLOUDFLARE WORKERS AI
# =========================================================

CLOUDFLARE_MODEL = (
    "@cf/meta/llama-3.2-11b-vision-instruct"
)

CLOUDFLARE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/"
    f"{CLOUDFLARE_MODEL}"
)


# =========================================================
# OUTPUT QUALITY
# =========================================================

BASE_WIDTH = 1080
BASE_HEIGHT = 1440

RENDER_SCALE = 2

WIDTH = BASE_WIDTH * RENDER_SCALE
HEIGHT = BASE_HEIGHT * RENDER_SCALE


def px(value):
    return int(
        value * RENDER_SCALE
    )


# =========================================================
# COVER DESIGN
# =========================================================

PHOTO_HEIGHT = px(1040)

FADE_HEIGHT = px(180)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 170, 255)


# =========================================================
# CHECK ASSETS
# =========================================================

if not os.path.exists(FONT_PATH):
    st.error(
        "Font missing. Expected: fonts/Anton-Regular.ttf"
    )
    st.stop()


if not os.path.exists(LOGO_PATH):
    st.warning(
        "Logo missing at assets/logo.png"
    )


# =========================================================
# PREPARE IMAGE FOR AI
# =========================================================

def prepare_ai_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # AI does not need the full 2160x2880 source.
    # This saves Cloudflare usage and upload size.
    image.thumbnail(
        (1280, 1280),
        Image.Resampling.LANCZOS
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=88,
        optimize=True
    )

    return buffer.getvalue()


def image_to_data_url(image_bytes):

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# =========================================================
# AI PROMPT
# =========================================================

def build_prompt(custom_headline=""):

    prompt = """
You are a human social media editor running a highly engaging viral Instagram page.

Study the supplied image carefully.

Return VALID JSON ONLY.

Do not use markdown.
Do not use a code block.
Do not write anything outside the JSON.

Use exactly this structure:

{
  "headline": "maximum 8 words",
  "suggested_highlight": ["word1", "word2"],
  "paragraph_1": "4 to 6 natural sentences",
  "paragraph_2": "4 to 6 natural sentences",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]
}

HEADLINE:

Maximum 8 words.

Make it bold, interesting, attention grabbing, dramatic, and appropriate for a viral Instagram page.

Do not invent specific facts that cannot reasonably be determined from the image.

SUGGESTED HIGHLIGHT:

Choose one or two important words that already exist exactly inside the headline.

CAPTION:

Write exactly two paragraphs.

Each paragraph should contain approximately 4 to 6 natural sentences.

The caption should deliberately over explain the situation.

Make the writing expressive, entertaining, energetic, dramatic, engaging, conversational, and attention grabbing.

Write like a real human running a viral Instagram media page.

Talk about the mood, reaction, visual details, atmosphere, situation, and why the subject feels interesting.

You may exaggerate emotion, humor, surprise, shock, tension, drama, intensity, or scale when appropriate.

Do not make it sound academic.

Do not make it sound like a technical report.

Avoid robotic AI sounding phrases.

Do not start with phrases such as "This image shows".

Do not say "It is important to note".

VERY IMPORTANT CAPTION RULES:

Never use hyphens.

Never use the character "-".

Never use em dashes.

Never use en dashes.

Never use bullet points.

Never use dash separated phrases.

Use natural commas, periods, exclamation marks, and question marks.

Do not invent statistics.

Do not invent dates.

Do not invent names.

Do not invent locations.

Do not invent world records.

Do not invent factual details that cannot reasonably be determined from the image.

HASHTAGS:

Return exactly 5 relevant hashtags.

Every hashtag must begin with #.

Do not use hyphens inside hashtags.

Do not return more than 5 hashtags.
"""

    if custom_headline.strip():

        prompt += f"""

The user provided this exact headline:

{custom_headline.strip()}

Use that headline EXACTLY.

Do not rewrite it.

The suggested_highlight words must exist inside that exact headline.
"""

    return prompt


# =========================================================
# CLOUDFLARE REQUEST
# =========================================================

def ask_cloudflare(
    original_image_bytes,
    custom_headline=""
):

    ai_image = prepare_ai_image(
        original_image_bytes
    )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an experienced human "
                    "Instagram social media editor."
                )
            },
            {
                "role": "user",
                "content": build_prompt(
                    custom_headline
                )
            }
        ],

        "image": image_to_data_url(
            ai_image
        ),

        "max_tokens": 1200,

        "temperature": 0.7
    }


    headers = {
        "Authorization": (
            f"Bearer {CLOUDFLARE_API_TOKEN}"
        ),
        "Content-Type": "application/json"
    }


    response = requests.post(
        CLOUDFLARE_URL,
        headers=headers,
        json=payload,
        timeout=180
    )


    if not response.ok:

        raise RuntimeError(
            "Cloudflare Workers AI error:\n\n"
            + response.text
        )


    data = response.json()


    if not data.get(
        "success",
        False
    ):

        raise RuntimeError(
            "Cloudflare request failed:\n\n"
            + json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            )
        )


    result = data.get(
        "result",
        {}
    )


    # =====================================================
    # IMPORTANT FIX
    #
    # Cloudflare may return:
    #
    # response = "text"
    #
    # OR:
    #
    # response = {...JSON object...}
    #
    # Both are accepted.
    # =====================================================

    if isinstance(result, dict):

        if "response" in result:

            output = result[
                "response"
            ]

        elif "output" in result:

            output = result[
                "output"
            ]

        elif "text" in result:

            output = result[
                "text"
            ]

        else:

            output = result

    else:

        output = result


    if output is None:

        raise RuntimeError(
            "Cloudflare returned an empty response."
        )


    return output


# =========================================================
# RESPONSE HELPERS
# =========================================================

def value_to_text(value):

    if value is None:
        return ""

    if isinstance(
        value,
        str
    ):
        return value.strip()

    if isinstance(
        value,
        (int, float, bool)
    ):
        return str(value)

    if isinstance(
        value,
        list
    ):

        return " ".join(
            value_to_text(item)
            for item in value
        ).strip()

    if isinstance(
        value,
        dict
    ):

        # Try common text fields first.

        for key in (
            "text",
            "response",
            "content",
            "value"
        ):

            if key in value:

                return value_to_text(
                    value[key]
                )

        return json.dumps(
            value,
            ensure_ascii=False
        )

    return str(value)


def strip_code_fence(text):

    # FIX:
    # Never call .strip() directly
    # unless we know it is a string.

    if isinstance(
        text,
        (dict, list)
    ):

        text = json.dumps(
            text,
            ensure_ascii=False
        )

    if text is None:
        return ""

    text = str(text).strip()


    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )


    text = re.sub(
        r"\s*```$",
        "",
        text
    )


    return text.strip()


# =========================================================
# NORMALIZE STRUCTURED RESPONSE
# =========================================================

def find_content_dict(value):

    target_keys = {
        "headline",
        "paragraph_1",
        "paragraph_2",
        "hashtags"
    }


    if isinstance(
        value,
        dict
    ):

        if (
            "headline" in value
            or
            len(
                target_keys.intersection(
                    value.keys()
                )
            ) >= 2
        ):

            return value


        for key in (
            "response",
            "output",
            "text",
            "content",
            "result"
        ):

            if key in value:

                found = find_content_dict(
                    value[key]
                )

                if found is not None:
                    return found


        for child in value.values():

            found = find_content_dict(
                child
            )

            if found is not None:
                return found


    elif isinstance(
        value,
        list
    ):

        for item in value:

            found = find_content_dict(
                item
            )

            if found is not None:
                return found


    return None


# =========================================================
# AI RESPONSE PARSER
# =========================================================

def parse_ai_response(raw_response):

    # -----------------------------------------------------
    # CASE 1:
    # Cloudflare already returned parsed JSON/dict.
    # -----------------------------------------------------

    content_dict = find_content_dict(
        raw_response
    )


    if content_dict is not None:

        headline = value_to_text(
            content_dict.get(
                "headline",
                ""
            )
        )


        highlights = content_dict.get(
            "suggested_highlight",
            []
        )


        if isinstance(
            highlights,
            str
        ):

            highlights = [
                highlights
            ]

        elif isinstance(
            highlights,
            dict
        ):

            highlights = list(
                highlights.values()
            )

        elif not isinstance(
            highlights,
            list
        ):

            highlights = [
                str(highlights)
            ]


        highlights = [
            value_to_text(item)
            for item in highlights
            if value_to_text(item)
        ]


        paragraph_1 = value_to_text(
            content_dict.get(
                "paragraph_1",
                ""
            )
        )


        paragraph_2 = value_to_text(
            content_dict.get(
                "paragraph_2",
                ""
            )
        )


        hashtags = content_dict.get(
            "hashtags",
            []
        )


        if isinstance(
            hashtags,
            str
        ):

            hashtags = hashtags.split()

        elif isinstance(
            hashtags,
            dict
        ):

            hashtags = list(
                hashtags.values()
            )

        elif not isinstance(
            hashtags,
            list
        ):

            hashtags = [
                str(hashtags)
            ]


        return (
            headline,
            highlights,
            paragraph_1,
            paragraph_2,
            hashtags
        )


    # -----------------------------------------------------
    # CASE 2:
    # Response is text containing JSON.
    # -----------------------------------------------------

    cleaned = strip_code_fence(
        raw_response
    )


    start = cleaned.find("{")
    end = cleaned.rfind("}")


    if (
        start != -1
        and end != -1
        and end > start
    ):

        json_text = cleaned[
            start:
            end + 1
        ]


        try:

            parsed = json.loads(
                json_text
            )


            return parse_ai_response(
                parsed
            )

        except Exception:

            pass


    # -----------------------------------------------------
    # CASE 3:
    # Fallback old style text parser.
    # -----------------------------------------------------

    headline = ""
    highlights = []
    paragraph_1 = ""
    paragraph_2 = ""
    hashtags = []


    for line in cleaned.splitlines():

        stripped = line.strip()

        upper = stripped.upper()


        if upper.startswith(
            "HEADLINE:"
        ):

            headline = stripped.split(
                ":",
                1
            )[1].strip()


        elif upper.startswith(
            "SUGGESTED_HIGHLIGHT:"
        ):

            value = stripped.split(
                ":",
                1
            )[1].strip()

            highlights = value.split()


        elif upper.startswith(
            "PARAGRAPH_1:"
        ):

            paragraph_1 = stripped.split(
                ":",
                1
            )[1].strip()


        elif upper.startswith(
            "PARAGRAPH_2:"
        ):

            paragraph_2 = stripped.split(
                ":",
                1
            )[1].strip()


        elif upper.startswith(
            "HASHTAGS:"
        ):

            hashtags = stripped.split(
                ":",
                1
            )[1].strip().split()


    return (
        headline,
        highlights,
        paragraph_1,
        paragraph_2,
        hashtags
    )


# =========================================================
# CAPTION CLEANUP
# =========================================================

def clean_paragraph(text):

    text = value_to_text(
        text
    )


    text = text.replace(
        "—",
        ", "
    )

    text = text.replace(
        "–",
        ", "
    )

    text = text.replace(
        "-",
        " "
    )


    for bullet in (
        "•",
        "●",
        "▪",
        "◦"
    ):

        text = text.replace(
            bullet,
            ""
        )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


def clean_hashtag(tag):

    tag = value_to_text(
        tag
    )


    tag = (
        tag
        .replace("-", "")
        .replace("—", "")
        .replace("–", "")
    )


    tag = re.sub(
        r"[^A-Za-z0-9_#]",
        "",
        tag
    )


    if not tag:
        return ""


    if not tag.startswith("#"):

        tag = (
            "#"
            + tag
        )


    return tag


def ensure_five_hashtags(
    hashtags,
    headline
):

    final_tags = []


    if isinstance(
        hashtags,
        str
    ):

        hashtags = hashtags.split()


    if not isinstance(
        hashtags,
        list
    ):

        hashtags = [
            hashtags
        ]


    for tag in hashtags:

        cleaned = clean_hashtag(
            tag
        )


        if (
            cleaned
            and cleaned not in final_tags
        ):

            final_tags.append(
                cleaned
            )


        if len(
            final_tags
        ) == 5:

            break


    # -----------------------------------------------------
    # If AI produced fewer than 5,
    # create relevant tags from headline.
    # -----------------------------------------------------

    stop_words = {
        "THE",
        "A",
        "AN",
        "AND",
        "OR",
        "OF",
        "IN",
        "ON",
        "AT",
        "TO",
        "IS",
        "ARE",
        "THIS",
        "THAT",
        "WITH",
        "FOR",
        "WILL"
    }


    headline_words = re.findall(
        r"[A-Za-z0-9]+",
        headline.upper()
    )


    for word in headline_words:

        if word in stop_words:
            continue


        tag = (
            "#"
            + word.title()
        )


        if tag not in final_tags:

            final_tags.append(
                tag
            )


        if len(
            final_tags
        ) == 5:

            break


    fallback_tags = [
        "#Viral",
        "#Trending",
        "#Explore",
        "#Photo",
        "#Instagram"
    ]


    for tag in fallback_tags:

        if tag not in final_tags:

            final_tags.append(
                tag
            )


        if len(
            final_tags
        ) == 5:

            break


    return final_tags[:5]


def build_final_caption(
    paragraph_1,
    paragraph_2,
    hashtags,
    headline
):

    p1 = clean_paragraph(
        paragraph_1
    )

    p2 = clean_paragraph(
        paragraph_2
    )


    tags = ensure_five_hashtags(
        hashtags,
        headline
    )


    parts = []


    if p1:
        parts.append(
            p1
        )


    if p2:
        parts.append(
            p2
        )


    parts.append(
        " ".join(tags)
    )


    return "\n\n".join(
        parts
    ).strip()


# =========================================================
# HEADLINE HELPERS
# =========================================================

def clean_word(word):

    return (
        value_to_text(word)
        .replace(",", "")
        .replace(".", "")
        .replace("!", "")
        .replace("?", "")
        .replace(":", "")
        .replace(";", "")
        .replace('"', "")
        .replace("'", "")
        .upper()
    )


def get_headline_word_options(
    headline
):

    options = []
    seen = set()


    for word in headline.split():

        cleaned = clean_word(
            word
        )


        if (
            cleaned
            and cleaned not in seen
        ):

            options.append(
                cleaned
            )

            seen.add(
                cleaned
            )


    return options


# =========================================================
# TEXT MEASUREMENT
# =========================================================

def text_width(
    draw,
    text,
    font
):

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    return (
        bbox[2]
        - bbox[0]
    )


def wrap_headline(
    draw,
    headline,
    font,
    max_width
):

    words = (
        headline
        .upper()
        .split()
    )


    lines = []
    current = []


    for word in words:

        candidate = " ".join(
            current
            + [word]
        )


        if text_width(
            draw,
            candidate,
            font
        ) <= max_width:

            current.append(
                word
            )


        else:

            if current:

                lines.append(
                    current
                )


            current = [
                word
            ]


    if current:

        lines.append(
            current
        )


    return lines


# =========================================================
# DRAW HEADLINE
# =========================================================

def draw_highlighted_line(
    draw,
    words,
    highlight_words,
    font,
    center_x,
    y
):

    space_width = text_width(
        draw,
        " ",
        font
    )


    widths = [

        text_width(
            draw,
            word,
            font
        )

        for word in words
    ]


    total_width = (
        sum(widths)
        + space_width
        * (
            len(words)
            - 1
        )
    )


    x = int(
        center_x
        - total_width / 2
    )


    highlight_set = {

        clean_word(word)

        for word in highlight_words
    }


    for word, width in zip(
        words,
        widths
    ):

        color = (
            BLUE
            if clean_word(
                word
            ) in highlight_set
            else WHITE
        )


        draw.text(
            (
                int(x),
                int(y)
            ),
            word,
            font=font,
            fill=color
        )


        x += (
            width
            + space_width
        )


# =========================================================
# HIGH QUALITY IMAGE FIT
# =========================================================

def high_quality_fit(
    source,
    size,
    centering=(0.5, 0.5)
):

    return ImageOps.fit(
        source.convert("RGB"),
        size,
        method=Image.Resampling.LANCZOS,
        centering=centering
    )


def make_full_slide(source):

    return high_quality_fit(
        source,
        (
            WIDTH,
            HEIGHT
        )
    )


# =========================================================
# BLACK FADE
# =========================================================

def add_black_fade(canvas):

    fade_start = (
        PHOTO_HEIGHT
        - FADE_HEIGHT
    )


    mask = Image.new(
        "L",
        (
            1,
            FADE_HEIGHT
        )
    )


    pixels = mask.load()


    for y in range(
        FADE_HEIGHT
    ):

        t = y / max(
            FADE_HEIGHT - 1,
            1
        )


        smooth = (
            t
            * t
            * (
                3
                - 2 * t
            )
        )


        pixels[
            0,
            y
        ] = int(
            smooth
            * 255
        )


    mask = mask.resize(
        (
            WIDTH,
            FADE_HEIGHT
        ),
        Image.Resampling.BILINEAR
    )


    black_strip = Image.new(
        "RGB",
        (
            WIDTH,
            FADE_HEIGHT
        ),
        BLACK
    )


    canvas.paste(
        black_strip,
        (
            0,
            fade_start
        ),
        mask
    )


# =========================================================
# COVER GENERATOR
# =========================================================

def create_cover(
    source,
    headline,
    highlight_words
):

    canvas = Image.new(
        "RGB",
        (
            WIDTH,
            HEIGHT
        ),
        BLACK
    )


    # PHOTO

    photo = high_quality_fit(
        source,
        (
            WIDTH,
            PHOTO_HEIGHT
        )
    )


    canvas.paste(
        photo,
        (
            0,
            0
        )
    )


    # FADE

    add_black_fade(
        canvas
    )


    draw = ImageDraw.Draw(
        canvas
    )


    # LOWER BLACK PANEL

    draw.rectangle(
        [
            0,
            PHOTO_HEIGHT,
            WIDTH,
            HEIGHT
        ],
        fill=BLACK
    )


    # LOGO

    if os.path.exists(
        LOGO_PATH
    ):

        logo = Image.open(
            LOGO_PATH
        ).convert(
            "RGBA"
        )


        logo.thumbnail(
            (
                px(760),
                px(760)
            ),
            Image.Resampling.LANCZOS
        )


        logo_x = (
            WIDTH
            - logo.width
        ) // 2


        logo_y = (
            PHOTO_HEIGHT
            - logo.height // 2
            - px(8)
        )


        canvas.paste(
            logo,
            (
                logo_x,
                logo_y
            ),
            logo
        )


    # DIVIDER

    divider_y = (
        PHOTO_HEIGHT
        + px(145)
    )


    draw.line(
        [
            px(50),
            divider_y,
            WIDTH
            - px(50),
            divider_y
        ],
        fill=WHITE,
        width=px(3)
    )


    # HEADLINE

    font_size = px(78)


    font = ImageFont.truetype(
        FONT_PATH,
        font_size
    )


    max_text_width = (
        WIDTH
        - px(40)
    )


    lines = wrap_headline(
        draw,
        headline,
        font,
        max_text_width
    )


    while (
        len(lines) > 2
        and font_size > px(52)
    ):

        font_size -= px(3)


        font = ImageFont.truetype(
            FONT_PATH,
            font_size
        )


        lines = wrap_headline(
            draw,
            headline,
            font,
            max_text_width
        )


    line_height = int(
        font_size
        * 0.90
    )


    total_text_height = (
        len(lines)
        * line_height
    )


    headline_y = (
        divider_y
        + px(12)
    )


    bottom = (
        HEIGHT
        - px(18)
    )


    if (
        headline_y
        + total_text_height
        > bottom
    ):

        headline_y = (
            bottom
            - total_text_height
        )


    for line_number, words in enumerate(
        lines
    ):

        draw_highlighted_line(
            draw=draw,
            words=words,
            highlight_words=highlight_words,
            font=font,
            center_x=WIDTH // 2,
            y=(
                headline_y
                + line_number
                * line_height
            )
        )


    return canvas


# =========================================================
# PNG EXPORT
# =========================================================

def image_to_png_bytes(image):

    buffer = io.BytesIO()


    image.save(
        buffer,
        format="PNG",
        optimize=False
    )


    return buffer.getvalue()


# =========================================================
# APP HEADER
# =========================================================

st.title(
    "Instagram Carousel Machine"
)

st.caption(
    "Cloud AI • High Quality • 3:4 • 2160 × 2880"
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "ai_ready": False,
    "headline": "",
    "suggested_highlight": [],
    "caption": "",
    "ai_response": "",
    "rendered_slides": []
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# =========================================================
# UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "Upload carousel images",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    accept_multiple_files=True
)


custom_headline = st.text_input(
    "Headline (optional)",
    placeholder=(
        "Leave empty and AI "
        "will create the headline"
    )
)


# =========================================================
# UPLOADED PREVIEW
# =========================================================

if uploaded_files:

    st.subheader(
        "Uploaded Images"
    )


    columns = st.columns(
        min(
            len(uploaded_files),
            4
        )
    )


    for index, file in enumerate(
        uploaded_files
    ):

        source = Image.open(
            io.BytesIO(
                file.getvalue()
            )
        )


        with columns[
            index
            % len(columns)
        ]:

            st.image(
                source,
                caption=(
                    f"Slide {index + 1}"
                ),
                use_container_width=True
            )


    # =====================================================
    # GENERATE
    # =====================================================

    if st.button(
        "GENERATE AI TEXT",
        type="primary"
    ):

        with st.spinner(
            "Cloud AI is analyzing the first image..."
        ):

            try:

                first_bytes = (
                    uploaded_files[
                        0
                    ].getvalue()
                )


                ai_response = (
                    ask_cloudflare(
                        first_bytes,
                        custom_headline
                    )
                )


                (
                    headline,
                    suggested,
                    paragraph_1,
                    paragraph_2,
                    hashtags
                ) = parse_ai_response(
                    ai_response
                )


                if custom_headline.strip():

                    headline = (
                        custom_headline
                        .strip()
                    )


                if not headline:

                    headline = (
                        "UNTITLED STORY"
                    )


                final_caption = (
                    build_final_caption(
                        paragraph_1,
                        paragraph_2,
                        hashtags,
                        headline
                    )
                )


                st.session_state[
                    "headline"
                ] = headline


                st.session_state[
                    "suggested_highlight"
                ] = suggested


                st.session_state[
                    "caption"
                ] = final_caption


                st.session_state[
                    "ai_response"
                ] = ai_response


                st.session_state[
                    "ai_ready"
                ] = True


                st.session_state[
                    "rendered_slides"
                ] = []


                st.success(
                    "AI text generated!"
                )


            except Exception as error:

                st.error(
                    str(error)
                )


# =========================================================
# CUSTOMIZE
# =========================================================

if (
    uploaded_files
    and st.session_state[
        "ai_ready"
    ]
):

    st.divider()

    st.header(
        "Customize Cover"
    )


    edited_headline = st.text_input(
        "Headline",
        value=(
            st.session_state[
                "headline"
            ]
        ),
        key="edited_headline"
    )


    word_options = (
        get_headline_word_options(
            edited_headline
        )
    )


    suggested_words = {

        clean_word(word)

        for word in
        st.session_state[
            "suggested_highlight"
        ]
    }


    default_highlights = [

        word

        for word in word_options

        if word in suggested_words
    ]


    selected_highlights = (
        st.multiselect(
            "Choose words to make BLUE",
            options=word_options,
            default=default_highlights
        )
    )


    st.caption(
        "You decide which words become blue."
    )


    # CAPTION

    st.subheader(
        "Instagram Caption"
    )


    edited_caption = st.text_area(
        "Edit caption if needed",
        value=(
            st.session_state[
                "caption"
            ]
        ),
        height=450
    )


    # =====================================================
    # RENDER
    # =====================================================

    if st.button(
        "RENDER CAROUSEL",
        type="primary"
    ):

        try:

            first_image = Image.open(
                io.BytesIO(
                    uploaded_files[
                        0
                    ].getvalue()
                )
            )


            generated_slides = []


            cover = create_cover(
                first_image,
                edited_headline,
                selected_highlights
            )


            generated_slides.append(
                image_to_png_bytes(
                    cover
                )
            )


            for file in (
                uploaded_files[1:]
            ):

                source = Image.open(
                    io.BytesIO(
                        file.getvalue()
                    )
                )


                slide = (
                    make_full_slide(
                        source
                    )
                )


                generated_slides.append(
                    image_to_png_bytes(
                        slide
                    )
                )


            st.session_state[
                "rendered_slides"
            ] = generated_slides


            st.session_state[
                "caption"
            ] = edited_caption


            st.success(
                "High quality carousel generated!"
            )


        except Exception as error:

            st.error(
                str(error)
            )


    # =====================================================
    # RESULTS
    # =====================================================

    if st.session_state[
        "rendered_slides"
    ]:

        st.header(
            "Preview & Download"
        )


        st.caption(
            "Downloaded files are full 2160 × 2880 PNG masters."
        )


        for index, png_data in enumerate(
            st.session_state[
                "rendered_slides"
            ],
            start=1
        ):

            st.markdown(
                f"## Slide {index}"
            )


            st.image(
                png_data,
                width=650
            )


            st.download_button(
                label=(
                    f"Download Slide "
                    f"{index} PNG"
                ),
                data=png_data,
                file_name=(
                    f"slide_"
                    f"{index:02d}.png"
                ),
                mime="image/png",
                key=(
                    f"download_"
                    f"{index}"
                ),
                on_click="ignore"
            )


            st.divider()


        st.subheader(
            "Final Caption"
        )


        st.text_area(
            "Copy this caption",
            value=(
                st.session_state[
                    "caption"
                ]
            ),
            height=450,
            key="final_caption"
        )


        with st.expander(
            "Raw AI Response"
        ):

            raw = st.session_state[
                "ai_response"
            ]


            if isinstance(
                raw,
                (dict, list)
            ):

                st.json(
                    raw
                )

            else:

                st.code(
                    str(raw)
                )
