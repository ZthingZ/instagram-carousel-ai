import base64
import io
import json
import os
import re

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Instagram Carousel Machine",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
# SECRETS
# =========================================================

def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, "")


ACCOUNT_ID = get_secret(
    "CLOUDFLARE_ACCOUNT_ID"
)

API_TOKEN = get_secret(
    "CLOUDFLARE_API_TOKEN"
)

if not ACCOUNT_ID:
    st.error("CLOUDFLARE_ACCOUNT_ID is missing.")
    st.stop()

if not API_TOKEN:
    st.error("CLOUDFLARE_API_TOKEN is missing.")
    st.stop()


# =========================================================
# MODELS
# =========================================================

VISION_MODEL = (
    "@cf/meta/llama-3.2-11b-vision-instruct"
)

# IMPORTANT:
# This model supports Cloudflare JSON Mode.
TEXT_MODEL = (
    "@cf/meta/llama-3.1-8b-instruct"
)


def model_url(model):
    return (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{ACCOUNT_ID}/ai/run/{model}"
    )


HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}


# =========================================================
# OUTPUT / DESIGN
# =========================================================

BASE_WIDTH = 1080
BASE_HEIGHT = 1440

RENDER_SCALE = 2

WIDTH = BASE_WIDTH * RENDER_SCALE
HEIGHT = BASE_HEIGHT * RENDER_SCALE


def px(value):
    return int(value * RENDER_SCALE)


PHOTO_HEIGHT = px(1040)
FADE_HEIGHT = px(180)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 170, 255)


# =========================================================
# ASSET CHECK
# =========================================================

if not os.path.exists(FONT_PATH):
    st.error(
        "Font missing. Expected fonts/Anton-Regular.ttf"
    )
    st.stop()

if not os.path.exists(LOGO_PATH):
    st.warning(
        "Logo missing at assets/logo.png"
    )


# =========================================================
# CLOUDFLARE REQUEST
# =========================================================

def call_cloudflare(model, payload):

    response = requests.post(
        model_url(model),
        headers=HEADERS,
        json=payload,
        timeout=180
    )

    if not response.ok:
        raise RuntimeError(
            "Cloudflare HTTP error:\n\n"
            + response.text
        )

    data = response.json()

    if not data.get("success", False):
        raise RuntimeError(
            "Cloudflare AI error:\n\n"
            + json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            )
        )

    return data.get("result")


# =========================================================
# SAFE TEXT EXTRACTION
# =========================================================

def extract_text(value):

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, list):
        parts = []

        for item in value:
            text = extract_text(item)

            if text:
                parts.append(text)

        return "\n".join(parts).strip()

    if isinstance(value, dict):

        for key in (
            "response",
            "text",
            "content",
            "output",
            "result"
        ):
            if key in value:
                text = extract_text(value[key])

                if text:
                    return text

        parts = []

        for item in value.values():
            text = extract_text(item)

            if text:
                parts.append(text)

        return "\n".join(parts).strip()

    return str(value).strip()


# =========================================================
# IMAGE PREPARATION FOR VISION AI
# =========================================================

def prepare_ai_image(original_bytes):

    image = Image.open(
        io.BytesIO(original_bytes)
    ).convert("RGB")

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


def image_data_url(image_bytes):

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# =========================================================
# STAGE 1
# IMAGE -> DESCRIPTION
# =========================================================

def analyze_image(original_bytes):

    ai_image = prepare_ai_image(
        original_bytes
    )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise visual analyst. "
                    "Describe only what can reasonably be seen."
                )
            },
            {
                "role": "user",
                "content": """
Look carefully at this image.

Describe what is visibly happening in one detailed paragraph.

Include:
the main subject,
what the subject is doing,
the setting,
visible objects,
facial expression or reaction if relevant,
the mood,
anything funny, strange, dramatic, surprising, or visually important.

The description must be detailed enough that another writer who cannot see the image can understand what is happening.

Do not write a headline.
Do not write an Instagram caption.
Do not write hashtags.
Do not use bullet points.

Do not invent names.
Do not invent dates.
Do not invent locations.
Do not invent statistics.
Do not invent facts that cannot reasonably be determined from the image.
"""
            }
        ],

        "image": image_data_url(
            ai_image
        ),

        "max_tokens": 500,
        "temperature": 0.2
    }

    result = call_cloudflare(
        VISION_MODEL,
        payload
    )

    description = extract_text(
        result
    )

    if not description:
        raise RuntimeError(
            "Vision AI returned an empty description."
        )

    return description, result


# =========================================================
# WRITER JSON SCHEMA
# =========================================================

WRITER_SCHEMA = {
    "type": "object",

    "properties": {
        "headline": {
            "type": "string"
        },

        "suggested_highlight": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "paragraph_1": {
            "type": "string"
        },

        "paragraph_2": {
            "type": "string"
        },

        "hashtags": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },

    "required": [
        "headline",
        "suggested_highlight",
        "paragraph_1",
        "paragraph_2",
        "hashtags"
    ],

    "additionalProperties": False
}


# =========================================================
# STAGE 2
# DESCRIPTION -> STRUCTURED INSTAGRAM COPY
# =========================================================

def generate_instagram_copy(
    description,
    custom_headline=""
):

    prompt = f"""
You are a human social media editor running a viral Instagram media page.

A vision model inspected the image and produced this accurate visual description:

{description}

Using that description as the factual basis, write Instagram carousel content.

HEADLINE:
Maximum 8 words.
Bold, dramatic, entertaining, interesting, and attention grabbing.
It must clearly relate to the image.
Never use "Untitled Story".

SUGGESTED HIGHLIGHT:
Choose one or two important words that already appear exactly in the headline.

PARAGRAPH 1:
Write approximately 4 to 6 natural sentences.
Deliberately over explain what is happening.
Describe the reaction, scene, mood, atmosphere, funny details, dramatic details, or surprising visual elements.

PARAGRAPH 2:
Write approximately 4 to 6 more natural sentences.
Continue the idea instead of repeating paragraph 1.
Make it expressive, conversational, dramatic, entertaining, and suitable for Instagram.

STYLE:
Sound like a real human running an entertaining viral social media page.
Do not sound academic.
Do not sound robotic.
Do not say "This image shows".
Do not say "It is important to note".

VERY IMPORTANT:
Never use hyphens.
Never use the character "-".
Never use em dashes.
Never use en dashes.
Never use bullet points.
Never write dash separated phrases.

Use normal commas, periods, question marks, and exclamation marks.

Do not invent names.
Do not invent dates.
Do not invent locations.
Do not invent precise statistics.
Do not invent world records.
Do not invent facts beyond the supplied visual description.

HASHTAGS:
Exactly 5 hashtags.
Each one must start with #.
All 5 should relate to the actual subject.
Do not use hyphens inside hashtags.
"""

    if custom_headline.strip():
        prompt += f"""

The user already provided this exact headline:

{custom_headline.strip()}

Use that headline EXACTLY.
Do not rewrite it.

suggested_highlight must contain only words that appear inside that exact headline.
"""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an experienced human "
                    "Instagram content writer."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        "response_format": {
            "type": "json_schema",
            "json_schema": WRITER_SCHEMA
        },

        "max_tokens": 1200,
        "temperature": 0.7
    }

    result = call_cloudflare(
        TEXT_MODEL,
        payload
    )

    return parse_writer_result(
        result
    ), result


# =========================================================
# PARSE JSON MODE RESPONSE
# =========================================================

def parse_writer_result(result):

    value = result

    # Typical Cloudflare JSON Mode:
    # result = {"response": {...}}

    if isinstance(value, dict):
        if "response" in value:
            value = value["response"]

    # Sometimes JSON might still arrive as a string.
    if isinstance(value, str):

        cleaned = value.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned
        )

        try:
            value = json.loads(
                cleaned
            )
        except Exception as exc:
            raise RuntimeError(
                "Writer AI returned text instead of valid structured JSON:\n\n"
                + cleaned
            ) from exc

    if not isinstance(value, dict):
        raise RuntimeError(
            "Writer AI returned an unexpected response:\n\n"
            + str(value)
        )

    headline = str(
        value.get("headline", "")
    ).strip()

    paragraph_1 = str(
        value.get("paragraph_1", "")
    ).strip()

    paragraph_2 = str(
        value.get("paragraph_2", "")
    ).strip()

    highlights = value.get(
        "suggested_highlight",
        []
    )

    hashtags = value.get(
        "hashtags",
        []
    )

    if not headline:
        raise RuntimeError(
            "Writer AI returned no headline."
        )

    if not paragraph_1:
        raise RuntimeError(
            "Writer AI returned no first paragraph."
        )

    if not paragraph_2:
        raise RuntimeError(
            "Writer AI returned no second paragraph."
        )

    if isinstance(highlights, str):
        highlights = [highlights]

    if not isinstance(highlights, list):
        highlights = []

    if isinstance(hashtags, str):
        hashtags = hashtags.split()

    if not isinstance(hashtags, list):
        hashtags = []

    return {
        "headline": headline,
        "suggested_highlight": highlights,
        "paragraph_1": paragraph_1,
        "paragraph_2": paragraph_2,
        "hashtags": hashtags
    }


# =========================================================
# CAPTION CLEANUP
# =========================================================

def clean_paragraph(text):

    text = str(text)

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

    tag = str(tag).strip()

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
        tag = "#" + tag

    return tag


def ensure_five_hashtags(
    hashtags,
    headline
):

    tags = []

    for tag in hashtags:

        cleaned = clean_hashtag(
            tag
        )

        if (
            cleaned
            and cleaned not in tags
        ):
            tags.append(
                cleaned
            )

        if len(tags) == 5:
            break

    # Fill only if model returned fewer than 5.

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

    headline_words_list = re.findall(
        r"[A-Za-z0-9]+",
        headline.upper()
    )

    for word in headline_words_list:

        if word in stop_words:
            continue

        tag = "#" + word.title()

        if tag not in tags:
            tags.append(tag)

        if len(tags) == 5:
            break

    fallback = [
        "#Viral",
        "#Trending",
        "#Explore",
        "#Photo",
        "#Instagram"
    ]

    for tag in fallback:

        if len(tags) == 5:
            break

        if tag not in tags:
            tags.append(tag)

    return tags[:5]


def build_caption(data):

    headline = data["headline"]

    p1 = clean_paragraph(
        data["paragraph_1"]
    )

    p2 = clean_paragraph(
        data["paragraph_2"]
    )

    hashtags = ensure_five_hashtags(
        data["hashtags"],
        headline
    )

    return (
        p1
        + "\n\n"
        + p2
        + "\n\n"
        + " ".join(hashtags)
    )


# =========================================================
# HEADLINE HELPERS
# =========================================================

def clean_word(word):

    return (
        str(word)
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


def get_headline_words(headline):

    words = []
    seen = set()

    for word in headline.split():

        cleaned = clean_word(
            word
        )

        if cleaned and cleaned not in seen:

            words.append(
                cleaned
            )

            seen.add(
                cleaned
            )

    return words


# =========================================================
# TEXT RENDER HELPERS
# =========================================================

def text_width(
    draw,
    text,
    font
):

    box = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    return (
        box[2] - box[0]
    )


def wrap_headline(
    draw,
    headline,
    font,
    max_width
):

    words = headline.upper().split()

    lines = []
    current = []

    for word in words:

        candidate = " ".join(
            current + [word]
        )

        if text_width(
            draw,
            candidate,
            font
        ) <= max_width:

            current.append(word)

        else:

            if current:
                lines.append(current)

            current = [word]

    if current:
        lines.append(current)

    return lines


def draw_colored_line(
    draw,
    words,
    highlight_words,
    font,
    center_x,
    y
):

    space = text_width(
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

    total = (
        sum(widths)
        + space * (len(words) - 1)
    )

    x = int(
        center_x
        - total / 2
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
            if clean_word(word)
            in highlight_set
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
            + space
        )


# =========================================================
# IMAGE FIT
# =========================================================

def high_quality_fit(
    source,
    size
):

    return ImageOps.fit(
        source.convert("RGB"),
        size,
        method=Image.Resampling.LANCZOS
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

        t = (
            y
            / max(
                FADE_HEIGHT - 1,
                1
            )
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
            smooth * 255
        )

    mask = mask.resize(
        (
            WIDTH,
            FADE_HEIGHT
        ),
        Image.Resampling.BILINEAR
    )

    black_layer = Image.new(
        "RGB",
        (
            WIDTH,
            FADE_HEIGHT
        ),
        BLACK
    )

    canvas.paste(
        black_layer,
        (
            0,
            fade_start
        ),
        mask
    )


# =========================================================
# COVER
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

    photo = high_quality_fit(
        source,
        (
            WIDTH,
            PHOTO_HEIGHT
        )
    )

    canvas.paste(
        photo,
        (0, 0)
    )

    add_black_fade(
        canvas
    )

    draw = ImageDraw.Draw(
        canvas
    )

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
        ).convert("RGBA")

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
            WIDTH - px(50),
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

    max_width = (
        WIDTH
        - px(40)
    )

    lines = wrap_headline(
        draw,
        headline,
        font,
        max_width
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
            max_width
        )

    line_height = int(
        font_size * 0.90
    )

    total_height = (
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
        + total_height
        > bottom
    ):
        headline_y = (
            bottom
            - total_height
        )

    for index, words in enumerate(
        lines
    ):

        draw_colored_line(
            draw,
            words,
            highlight_words,
            font,
            WIDTH // 2,
            (
                headline_y
                + index
                * line_height
            )
        )

    return canvas


# =========================================================
# OTHER SLIDES
# =========================================================

def make_full_slide(image):

    return high_quality_fit(
        image,
        (
            WIDTH,
            HEIGHT
        )
    )


# =========================================================
# PNG
# =========================================================

def to_png(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=False
    )

    return buffer.getvalue()


# =========================================================
# APP UI
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
    "ready": False,
    "headline": "",
    "suggested": [],
    "caption": "",
    "vision_description": "",
    "vision_raw": "",
    "writer_raw": "",
    "slides": []
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# =========================================================
# UPLOAD
# =========================================================

uploaded = st.file_uploader(
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
# PREVIEW
# =========================================================

if uploaded:

    st.subheader(
        "Uploaded Images"
    )

    cols = st.columns(
        min(
            len(uploaded),
            4
        )
    )

    for index, file in enumerate(
        uploaded
    ):

        image = Image.open(
            io.BytesIO(
                file.getvalue()
            )
        )

        with cols[
            index % len(cols)
        ]:

            st.image(
                image,
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

        try:

            first_bytes = (
                uploaded[
                    0
                ].getvalue()
            )

            # -------------------------
            # STAGE 1
            # -------------------------

            with st.spinner(
                "Step 1 of 2: AI is looking at the image..."
            ):

                description, vision_raw = (
                    analyze_image(
                        first_bytes
                    )
                )

            st.session_state[
                "vision_description"
            ] = description

            st.session_state[
                "vision_raw"
            ] = vision_raw


            # -------------------------
            # STAGE 2
            # -------------------------

            with st.spinner(
                "Step 2 of 2: AI is writing the Instagram post..."
            ):

                writer_data, writer_raw = (
                    generate_instagram_copy(
                        description,
                        custom_headline
                    )
                )

            st.session_state[
                "writer_raw"
            ] = writer_raw


            if custom_headline.strip():

                writer_data[
                    "headline"
                ] = (
                    custom_headline
                    .strip()
                )


            caption = build_caption(
                writer_data
            )


            suggested = [
                clean_word(word)
                for word
                in writer_data[
                    "suggested_highlight"
                ]
            ]


            st.session_state[
                "headline"
            ] = writer_data[
                "headline"
            ]

            st.session_state[
                "suggested"
            ] = suggested

            st.session_state[
                "caption"
            ] = caption

            st.session_state[
                "ready"
            ] = True

            st.session_state[
                "slides"
            ] = []


            st.success(
                "AI successfully analyzed the image and created the post."
            )


        except Exception as error:

            st.session_state[
                "ready"
            ] = False

            st.error(
                str(error)
            )


# =========================================================
# DEBUG VISION EVEN IF WRITER FAILS
# =========================================================

if st.session_state[
    "vision_description"
]:

    with st.expander(
        "What the Vision AI saw"
    ):

        st.write(
            st.session_state[
                "vision_description"
            ]
        )


# =========================================================
# CUSTOMIZE
# =========================================================

if (
    uploaded
    and st.session_state[
        "ready"
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


    options = get_headline_words(
        edited_headline
    )


    default_blue = [
        word
        for word in options
        if word in st.session_state[
            "suggested"
        ]
    ]


    selected_blue = st.multiselect(
        "Choose words to make BLUE",

        options=options,

        default=default_blue
    )


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

        first = Image.open(
            io.BytesIO(
                uploaded[
                    0
                ].getvalue()
            )
        )


        slides = []


        cover = create_cover(
            first,
            edited_headline,
            selected_blue
        )


        slides.append(
            to_png(
                cover
            )
        )


        for file in uploaded[1:]:

            image = Image.open(
                io.BytesIO(
                    file.getvalue()
                )
            )


            slide = make_full_slide(
                image
            )


            slides.append(
                to_png(
                    slide
                )
            )


        st.session_state[
            "slides"
        ] = slides


        st.session_state[
            "caption"
        ] = edited_caption


        st.success(
            "Carousel generated!"
        )


# =========================================================
# DOWNLOAD
# =========================================================

if st.session_state[
    "slides"
]:

    st.header(
        "Preview & Download"
    )


    for index, data in enumerate(
        st.session_state[
            "slides"
        ],
        start=1
    ):

        st.markdown(
            f"## Slide {index}"
        )


        st.image(
            data,
            width=650
        )


        st.download_button(
            label=(
                f"Download Slide "
                f"{index} PNG"
            ),

            data=data,

            file_name=(
                f"slide_"
                f"{index:02d}.png"
            ),

            mime="image/png",

            key=(
                f"download_{index}"
            )
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


# =========================================================
# DEBUG
# =========================================================

with st.expander(
    "Debug AI Responses"
):

    st.markdown(
        "### Vision AI raw response"
    )

    vision_raw = st.session_state[
        "vision_raw"
    ]

    if isinstance(
        vision_raw,
        (dict, list)
    ):
        st.json(
            vision_raw
        )
    else:
        st.code(
            str(vision_raw)
        )


    st.markdown(
        "### Writer AI raw response"
    )

    writer_raw = st.session_state[
        "writer_raw"
    ]

    if isinstance(
        writer_raw,
        (dict, list)
    ):
        st.json(
            writer_raw
        )
    else:
        st.code(
            str(writer_raw)
        )
