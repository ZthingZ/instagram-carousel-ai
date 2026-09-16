import base64
import io
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


ACCOUNT_ID = get_secret(
    "CLOUDFLARE_ACCOUNT_ID"
)

API_TOKEN = get_secret(
    "CLOUDFLARE_API_TOKEN"
)


if not ACCOUNT_ID:
    st.error(
        "CLOUDFLARE_ACCOUNT_ID is missing."
    )
    st.stop()


if not API_TOKEN:
    st.error(
        "CLOUDFLARE_API_TOKEN is missing."
    )
    st.stop()


# =========================================================
# CLOUDFLARE MODELS
# =========================================================

VISION_MODEL = (
    "@cf/meta/llama-3.2-11b-vision-instruct"
)

TEXT_MODEL = (
    "@cf/meta/llama-3.1-8b-instruct-fast"
)


def cloudflare_url(model):

    return (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{ACCOUNT_ID}/ai/run/"
        f"{model}"
    )


HEADERS = {
    "Authorization":
        f"Bearer {API_TOKEN}",
    "Content-Type":
        "application/json"
}


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
# CHECK FILES
# =========================================================

if not os.path.exists(FONT_PATH):

    st.error(
        "Font missing: fonts/Anton-Regular.ttf"
    )

    st.stop()


if not os.path.exists(LOGO_PATH):

    st.warning(
        "Logo missing: assets/logo.png"
    )


# =========================================================
# CLOUDFLARE RESPONSE EXTRACTOR
# =========================================================

def extract_text(value):
    """
    Cloudflare can return:
    string
    dict with response
    dict with text
    nested result
    list

    This function safely extracts readable text.
    """

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

        parts = []

        for item in value:

            text = extract_text(
                item
            )

            if text:

                parts.append(
                    text
                )

        return "\n".join(
            parts
        ).strip()


    if isinstance(
        value,
        dict
    ):

        # Prefer known Cloudflare output keys

        for key in (
            "response",
            "text",
            "content",
            "output",
            "result"
        ):

            if key in value:

                text = extract_text(
                    value[key]
                )

                if text:

                    return text


        # Last resort: recursively inspect values

        parts = []

        for item in value.values():

            text = extract_text(
                item
            )

            if text:

                parts.append(
                    text
                )


        return "\n".join(
            parts
        ).strip()


    return str(
        value
    ).strip()


# =========================================================
# GENERIC CLOUDFLARE REQUEST
# =========================================================

def run_cloudflare(
    model,
    payload
):

    response = requests.post(
        cloudflare_url(model),
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


    if not data.get(
        "success",
        False
    ):

        raise RuntimeError(
            "Cloudflare AI request failed:\n\n"
            + str(data)
        )


    result = data.get(
        "result"
    )


    text = extract_text(
        result
    )


    if not text:

        raise RuntimeError(
            "Cloudflare returned an empty AI response."
        )


    return text


# =========================================================
# PREPARE SMALL IMAGE FOR VISION AI
# =========================================================

def prepare_ai_image(
    original_bytes
):

    image = Image.open(
        io.BytesIO(
            original_bytes
        )
    ).convert(
        "RGB"
    )


    image.thumbnail(
        (
            1280,
            1280
        ),
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


def image_data_url(
    image_bytes
):

    encoded = base64.b64encode(
        image_bytes
    ).decode(
        "utf-8"
    )


    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# =========================================================
# AI STAGE 1
#
# IMAGE → DETAILED DESCRIPTION
# =========================================================

def analyze_image(
    original_bytes
):

    ai_image = prepare_ai_image(
        original_bytes
    )


    prompt = """
Look carefully at the supplied image.

Your ONLY job is to describe what is actually visible.

Write one detailed natural language description.

Mention:

the main subject
what the subject is doing
the environment or setting
important visible objects
facial expression or reaction if relevant
the mood of the scene
anything unusual, funny, dramatic, surprising, or visually interesting

Be specific enough that another writer who cannot see the image could understand what is happening.

Do not write an Instagram headline.

Do not write hashtags.

Do not write an Instagram caption.

Do not use bullet points.

Do not invent names.

Do not invent locations.

Do not invent dates.

Do not invent statistics.

Do not claim facts that cannot reasonably be determined from the image.

Only describe the image.
"""


    payload = {

        "messages": [
            {
                "role":
                    "system",

                "content":
                    (
                        "You are a precise visual "
                        "image analyst."
                    )
            },
            {
                "role":
                    "user",

                "content":
                    prompt
            }
        ],

        "image":
            image_data_url(
                ai_image
            ),

        "max_tokens":
            500,

        "temperature":
            0.2
    }


    return run_cloudflare(
        VISION_MODEL,
        payload
    )


# =========================================================
# AI STAGE 2
#
# DESCRIPTION → INSTAGRAM COPY
# =========================================================

def generate_instagram_copy(
    image_description,
    custom_headline=""
):

    prompt = f"""
You are a human social media editor running a viral Instagram media page.

Another AI carefully inspected an image and produced this visual description:

IMAGE DESCRIPTION:
{image_description}

Using ONLY that description as the factual basis, create Instagram carousel copy.

Return EXACTLY this format:

HEADLINE: your headline
HIGHLIGHT: one or two exact words from the headline
PARAGRAPH_1: first paragraph
PARAGRAPH_2: second paragraph
HASHTAGS: #tag1 #tag2 #tag3 #tag4 #tag5

HEADLINE RULES:

Maximum 8 words.

Make it bold, dramatic, engaging, viral, and attention grabbing.

The headline should strongly relate to what is actually happening in the image.

Do not use "Untitled Story".

HIGHLIGHT RULES:

Choose one or two important words.

The words must exist exactly inside the headline.

CAPTION RULES:

Write exactly TWO paragraphs.

Each paragraph should contain around 4 to 6 natural sentences.

The paragraphs should be deliberately over explanatory.

Explain what is happening.

Expand on the visual reaction, atmosphere, mood, funny detail, surprising detail, tension, or drama.

Write like a real human who runs an entertaining Instagram media page.

Make it expressive, conversational, engaging, and slightly exaggerated.

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

Use commas, periods, exclamation marks, and question marks naturally.

Do not invent names.

Do not invent dates.

Do not invent locations.

Do not invent statistics.

Do not invent world records.

Do not invent factual information beyond the supplied visual description.

HASHTAG RULES:

Exactly 5 hashtags.

Each hashtag begins with #.

All hashtags must relate to the actual subject.

Do not use generic unrelated hashtags unless necessary.

Do not use hyphens inside hashtags.
"""


    if custom_headline.strip():

        prompt += f"""

IMPORTANT:

The user has already provided this exact headline:

{custom_headline.strip()}

Use it EXACTLY as HEADLINE.

Do not change a single word.

HIGHLIGHT must be one or two words that already exist inside that exact headline.
"""


    payload = {

        "messages": [
            {
                "role":
                    "system",

                "content":
                    (
                        "You are an experienced "
                        "human Instagram content writer."
                    )
            },
            {
                "role":
                    "user",

                "content":
                    prompt
            }
        ],

        "max_tokens":
            1200,

        "temperature":
            0.7
    }


    return run_cloudflare(
        TEXT_MODEL,
        payload
    )


# =========================================================
# PARSE STAGE 2 OUTPUT
# =========================================================

def parse_copy(
    text
):

    fields = {
        "headline": "",
        "highlight": "",
        "paragraph_1": "",
        "paragraph_2": "",
        "hashtags": ""
    }


    current_field = None


    label_map = {
        "HEADLINE:":
            "headline",

        "HIGHLIGHT:":
            "highlight",

        "PARAGRAPH_1:":
            "paragraph_1",

        "PARAGRAPH_2:":
            "paragraph_2",

        "HASHTAGS:":
            "hashtags"
    }


    for raw_line in (
        text.splitlines()
    ):

        line = raw_line.strip()


        if not line:

            continue


        matched = False


        for label, key in (
            label_map.items()
        ):

            if line.upper().startswith(
                label
            ):

                fields[
                    key
                ] = line[
                    len(label):
                ].strip()

                current_field = key

                matched = True

                break


        if (
            not matched
            and current_field
        ):

            fields[
                current_field
            ] += (
                " "
                + line
            )


    headline = fields[
        "headline"
    ].strip()


    highlight = fields[
        "highlight"
    ].strip()


    paragraph_1 = fields[
        "paragraph_1"
    ].strip()


    paragraph_2 = fields[
        "paragraph_2"
    ].strip()


    hashtags = fields[
        "hashtags"
    ].strip()


    # IMPORTANT:
    # No more silent UNTITLED STORY fallback.

    if not headline:

        raise ValueError(
            "AI did not return a headline."
        )


    if not paragraph_1:

        raise ValueError(
            "AI did not return paragraph 1."
        )


    if not paragraph_2:

        raise ValueError(
            "AI did not return paragraph 2."
        )


    return (
        headline,
        highlight,
        paragraph_1,
        paragraph_2,
        hashtags
    )


# =========================================================
# CLEAN CAPTION
# =========================================================

def remove_dashes(
    text
):

    text = (
        text
        .replace(
            "—",
            ", "
        )
        .replace(
            "–",
            ", "
        )
        .replace(
            "-",
            " "
        )
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


def extract_hashtags(
    text
):

    tags = re.findall(
        r"#[A-Za-z0-9_]+",
        text
    )


    unique = []


    for tag in tags:

        if tag not in unique:

            unique.append(
                tag
            )


        if len(
            unique
        ) == 5:

            break


    return unique


def build_caption(
    p1,
    p2,
    hashtags,
    headline
):

    p1 = remove_dashes(
        p1
    )

    p2 = remove_dashes(
        p2
    )


    tags = extract_hashtags(
        hashtags
    )


    # Only fill missing hashtags.
    # Never replace actual paragraphs.

    if len(tags) < 5:

        words = re.findall(
            r"[A-Za-z0-9]+",
            headline
        )


        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "with",
            "this",
            "that",
            "is",
            "are"
        }


        for word in words:

            if (
                word.lower()
                in stop_words
            ):

                continue


            tag = (
                "#"
                + word.title()
            )


            if tag not in tags:

                tags.append(
                    tag
                )


            if len(
                tags
            ) == 5:

                break


    fallback = [
        "#Viral",
        "#Trending",
        "#Explore",
        "#Photo",
        "#Instagram"
    ]


    for tag in fallback:

        if len(
            tags
        ) >= 5:

            break


        if tag not in tags:

            tags.append(
                tag
            )


    return (
        p1
        + "\n\n"
        + p2
        + "\n\n"
        + " ".join(
            tags[:5]
        )
    )


# =========================================================
# HEADLINE HELPERS
# =========================================================

def clean_word(
    word
):

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


def headline_words(
    headline
):

    options = []
    seen = set()


    for word in (
        headline.split()
    ):

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
# TEXT DRAWING HELPERS
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
        box[2]
        - box[0]
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

        candidate = (
            " ".join(
                current
                + [word]
            )
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
        + space
        * (
            len(words)
            - 1
        )
    )


    x = int(
        center_x
        - total / 2
    )


    highlight_set = {

        clean_word(word)

        for word
        in highlight_words
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
            + space
        )


# =========================================================
# IMAGE RESIZE
# =========================================================

def high_quality_fit(
    source,
    size
):

    return ImageOps.fit(
        source.convert(
            "RGB"
        ),
        size,
        method=(
            Image.Resampling.LANCZOS
        )
    )


# =========================================================
# BLACK FADE
# =========================================================

def add_black_fade(
    canvas
):

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


    black = Image.new(
        "RGB",
        (
            WIDTH,
            FADE_HEIGHT
        ),
        BLACK
    )


    canvas.paste(
        black,
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
        (
            0,
            0
        )
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


        font = (
            ImageFont.truetype(
                FONT_PATH,
                font_size
            )
        )


        lines = wrap_headline(
            draw,
            headline,
            font,
            max_width
        )


    line_height = int(
        font_size
        * 0.90
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
# FULL IMAGE SLIDE
# =========================================================

def make_full_slide(
    image
):

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

def to_png(
    image
):

    buffer = io.BytesIO()


    image.save(
        buffer,
        format="PNG",
        optimize=False
    )


    return buffer.getvalue()


# =========================================================
# APP
# =========================================================

st.title(
    "Instagram Carousel Machine"
)

st.caption(
    "Cloud AI • High Quality • 3:4 • 2160 × 2880"
)


# =========================================================
# SESSION
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


for key, value in (
    defaults.items()
):

    if key not in (
        st.session_state
    ):

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
            index
            % len(cols)
        ]:

            st.image(
                image,
                caption=(
                    f"Slide {index + 1}"
                ),
                use_container_width=True
            )


    # =====================================================
    # GENERATE AI
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


            # STAGE 1

            with st.spinner(
                "Step 1 of 2: AI is looking at the image..."
            ):

                description = (
                    analyze_image(
                        first_bytes
                    )
                )


            st.session_state[
                "vision_description"
            ] = description


            st.session_state[
                "vision_raw"
            ] = description


            # STAGE 2

            with st.spinner(
                "Step 2 of 2: AI is writing the Instagram post..."
            ):

                writer_response = (
                    generate_instagram_copy(
                        description,
                        custom_headline
                    )
                )


            st.session_state[
                "writer_raw"
            ] = writer_response


            (
                headline,
                highlight,
                p1,
                p2,
                hashtags
            ) = parse_copy(
                writer_response
            )


            if custom_headline.strip():

                headline = (
                    custom_headline.strip()
                )


            caption = build_caption(
                p1,
                p2,
                hashtags,
                headline
            )


            suggested = [
                clean_word(
                    word
                )
                for word
                in highlight.split()
            ]


            st.session_state[
                "headline"
            ] = headline


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

            st.error(
                str(error)
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


    edited_headline = (
        st.text_input(
            "Headline",
            value=(
                st.session_state[
                    "headline"
                ]
            ),
            key=(
                "edited_headline"
            )
        )
    )


    options = headline_words(
        edited_headline
    )


    defaults_blue = [

        word

        for word in options

        if word in (
            st.session_state[
                "suggested"
            ]
        )
    ]


    selected_blue = (
        st.multiselect(
            "Choose words to make BLUE",
            options=options,
            default=defaults_blue
        )
    )


    st.subheader(
        "Instagram Caption"
    )


    edited_caption = (
        st.text_area(
            "Edit caption if needed",
            value=(
                st.session_state[
                    "caption"
                ]
            ),
            height=450
        )
    )


    # =====================================================
    # SHOW VISION DESCRIPTION
    # =====================================================

    with st.expander(
        "What the Vision AI saw"
    ):

        st.write(
            st.session_state[
                "vision_description"
            ]
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


        for file in (
            uploaded[1:]
        ):

            image = Image.open(
                io.BytesIO(
                    file.getvalue()
                )
            )


            slide = (
                make_full_slide(
                    image
                )
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

if (
    uploaded
    and st.session_state[
        "slides"
    ]
):

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
        "Debug AI Responses"
    ):

        st.markdown(
            "### Vision AI"
        )

        st.code(
            st.session_state[
                "vision_raw"
            ]
        )


        st.markdown(
            "### Writing AI"
        )

        st.code(
            st.session_state[
                "writer_raw"
            ]
        )
