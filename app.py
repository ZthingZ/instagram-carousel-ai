import base64
import io
import json
import os
import re

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps


# =========================================================
# PROJECT PATHS
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

# Design size:
# 1080 x 1440
#
# Actual exported master:
# 2160 x 2880

BASE_WIDTH = 1080
BASE_HEIGHT = 1440

RENDER_SCALE = 2

WIDTH = BASE_WIDTH * RENDER_SCALE
HEIGHT = BASE_HEIGHT * RENDER_SCALE


def px(value):
    return int(value * RENDER_SCALE)


# =========================================================
# COVER DESIGN
# =========================================================

# Large photo area = smaller black panel

PHOTO_HEIGHT = px(1040)

# Soft photo to black transition

FADE_HEIGHT = px(180)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Brand blue

BLUE = (0, 170, 255)


# =========================================================
# CHECK ASSETS
# =========================================================

if not os.path.exists(FONT_PATH):
    st.error(
        "Font not found. Expected:\n\n"
        "fonts/Anton-Regular.ttf"
    )
    st.stop()


if not os.path.exists(LOGO_PATH):
    st.warning(
        "Logo not found at assets/logo.png. "
        "The app can still run, but the cover will have no logo."
    )


# =========================================================
# PREPARE IMAGE FOR CLOUD AI
# =========================================================

def prepare_ai_image(image_bytes):
    """
    Makes a smaller copy ONLY for Cloudflare AI.

    The original uploaded image is still used later
    for the high resolution PNG render.
    """

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    max_side = 1280

    image.thumbnail(
        (max_side, max_side),
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
You are a human social media editor who runs a highly engaging viral Instagram media page.

Look carefully at the supplied image.

Write content for an Instagram carousel.

Return VALID JSON ONLY.

Do not use markdown.
Do not use a code block.
Do not write anything before or after the JSON.

Use exactly this structure:

{
  "headline": "maximum 8 words",
  "suggested_highlight": ["one or two words copied exactly from headline"],
  "paragraph_1": "4 to 6 natural sentences",
  "paragraph_2": "4 to 6 natural sentences",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]
}

HEADLINE RULES:

The headline must be short, bold, interesting, and social media friendly.

Maximum 8 words.

Do not invent facts that cannot reasonably be known from the image.

CAPTION RULES:

Write exactly two paragraphs.

Each paragraph should normally contain around 4 to 6 sentences.

The caption should deliberately over explain the situation.

Make it dramatic, expressive, entertaining, energetic, engaging, and attention grabbing.

Write like a real human running a viral Instagram page.

Talk about the mood, visual details, reaction, atmosphere, situation, and why the subject feels interesting.

It can feel slightly exaggerated.

You may exaggerate emotion, humor, shock, surprise, drama, scale, tension, or intensity when appropriate.

Do not make the caption sound academic.

Do not make it sound like a technical report.

Do not use robotic phrases.

Do not start with phrases such as "This image shows".

Do not say "It is important to note".

VERY IMPORTANT:

Never use hyphens in paragraph_1 or paragraph_2.

Never use the character "-".

Never use em dashes.

Never use en dashes.

Never use bullet points.

Never write dash separated phrases.

Use normal punctuation such as commas, periods, exclamation marks, and question marks.

Do not invent precise statistics.

Do not invent dates.

Do not invent names.

Do not invent locations.

Do not invent world records.

Do not claim something is historically true unless that information is known.

Do not invent factual details that cannot reasonably be determined from the image.

HASHTAG RULES:

Return exactly 5 hashtags.

Each hashtag must begin with #.

Every hashtag should be relevant.

Do not put hyphens inside hashtags.

Do not return more than 5 hashtags.
"""

    if custom_headline.strip():

        prompt += f"""

The user already provided this exact headline:

{custom_headline.strip()}

Use it EXACTLY as the value of "headline".

Do not rewrite it.

Every word in "suggested_highlight" must appear in that exact headline.
"""

    return prompt


# =========================================================
# CALL CLOUDFLARE
# =========================================================

def ask_cloudflare(
    original_image_bytes,
    custom_headline=""
):

    ai_image_bytes = prepare_ai_image(
        original_image_bytes
    )

    image_data_url = image_to_data_url(
        ai_image_bytes
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

        "image": image_data_url,

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
            "Cloudflare Workers AI returned an error:\n\n"
            + response.text
        )

    data = response.json()

    if not data.get("success", False):
        raise RuntimeError(
            "Cloudflare request was not successful:\n\n"
            + json.dumps(
                data,
                indent=2
            )
        )

    result = data.get(
        "result",
        {}
    )

    if isinstance(result, dict):

        text = (
            result.get("response")
            or result.get("text")
            or result.get("output")
            or ""
        )

    else:
        text = str(result)

    if not text:
        raise RuntimeError(
            "Cloudflare returned no generated text."
        )

    return text


# =========================================================
# PARSE AI RESPONSE
# =========================================================

def strip_code_fence(text):

    text = text.strip()

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


def parse_ai_response(text):

    cleaned = strip_code_fence(
        text
    )

    # Try to extract the first JSON object.

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:

        json_text = cleaned[
            start:end + 1
        ]

        try:

            data = json.loads(
                json_text
            )

            headline = str(
                data.get(
                    "headline",
                    ""
                )
            ).strip()

            highlights = data.get(
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

            paragraph_1 = str(
                data.get(
                    "paragraph_1",
                    ""
                )
            ).strip()

            paragraph_2 = str(
                data.get(
                    "paragraph_2",
                    ""
                )
            ).strip()

            hashtags = data.get(
                "hashtags",
                []
            )

            if isinstance(
                hashtags,
                str
            ):
                hashtags = hashtags.split()

            return (
                headline,
                highlights,
                paragraph_1,
                paragraph_2,
                hashtags
            )

        except Exception:
            pass


    # Fallback parser if the model ignored JSON.

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

    text = text.replace(
        "•",
        ""
    )

    text = text.replace(
        "●",
        ""
    )

    text = text.replace(
        "▪",
        ""
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_hashtag(tag):

    tag = str(
        tag
    ).strip()

    tag = tag.replace(
        "-",
        ""
    )

    tag = tag.replace(
        "—",
        ""
    )

    tag = tag.replace(
        "–",
        ""
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

    final_tags = []

    for tag in hashtags:

        cleaned = clean_hashtag(
            tag
        )

        if (
            cleaned
            and cleaned
            not in final_tags
        ):

            final_tags.append(
                cleaned
            )

        if len(
            final_tags
        ) == 5:
            break


    # If AI returned fewer than 5,
    # create additional tags from headline words.

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
        "FOR"
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
        "#Instagram",
        "#Explore",
        "#Photo"
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

    return (
        p1
        + "\n\n"
        + p2
        + "\n\n"
        + " ".join(tags)
    ).strip()


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

    words = headline.upper().split()

    lines = []

    current = []

    for word in words:

        candidate = " ".join(
            current
            + [word]
        )

        width = text_width(
            draw,
            candidate,
            font
        )

        if width <= max_width:

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
# DRAW COLORED HEADLINE
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
            )
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
            + space_width
        )


# =========================================================
# HIGH QUALITY IMAGE FIT
# =========================================================

def high_quality_fit(
    source,
    size,
    centering=(
        0.5,
        0.5
    )
):

    return ImageOps.fit(
        source.convert(
            "RGB"
        ),
        size,
        method=(
            Image.Resampling.LANCZOS
        ),
        centering=centering
    )


def make_full_slide(
    source
):

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


    # BLACK FADE

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

        max_logo_width = (
            px(760)
        )

        max_logo_height = (
            px(760)
        )

        logo.thumbnail(
            (
                max_logo_width,
                max_logo_height
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

    font_size = (
        px(78)
    )

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
        and font_size
        > px(52)
    ):

        font_size -= (
            px(3)
        )

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

    for (
        line_number,
        words
    ) in enumerate(
        lines
    ):

        draw_highlighted_line(
            draw=draw,
            words=words,
            highlight_words=(
                highlight_words
            ),
            font=font,
            center_x=(
                WIDTH // 2
            ),
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

def image_to_png_bytes(
    image
):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=False
    )

    return (
        buffer.getvalue()
    )


# =========================================================
# STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title=(
        "Instagram Carousel Machine"
    ),
    layout="wide"
)

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

    "ai_ready":
        False,

    "headline":
        "",

    "suggested_highlight":
        [],

    "caption":
        "",

    "ai_response":
        "",

    "rendered_slides":
        []

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
# ORIGINAL IMAGE PREVIEW
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
    # GENERATE AI TEXT
    # =====================================================

    if st.button(
        "GENERATE AI TEXT",
        type="primary"
    ):

        with st.spinner(
            "Cloud AI is analyzing the first image..."
        ):

            first_bytes = (
                uploaded_files[
                    0
                ].getvalue()
            )

            try:

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

                if (
                    custom_headline
                    .strip()
                ):

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

            except Exception as error:

                st.error(
                    str(error)
                )


# =========================================================
# CUSTOMIZE
# =========================================================

if (
    uploaded_files
    and
    st.session_state.ai_ready
):

    st.divider()

    st.header(
        "Customize Cover"
    )


    edited_headline = st.text_input(
        "Headline",

        value=(
            st.session_state
            .headline
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

        for word
        in st.session_state[
            "suggested_highlight"
        ]
    }


    default_highlights = [

        word

        for word
        in word_options

        if word
        in suggested_words
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
            st.session_state
            .caption
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

            slide = make_full_slide(
                source
            )

            generated_slides.append(
                image_to_png_bytes(
                    slide
                )
            )


        st.session_state[
            "rendered_slides"
        ] = generated_slides

        st.success(
            "High quality carousel generated!"
        )


# =========================================================
# RESULT
# =========================================================

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

        preview_image = Image.open(
            io.BytesIO(
                png_data
            )
        )

        st.image(
            preview_image,
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

        value=edited_caption,

        height=450,

        key="final_caption"
    )


    with st.expander(
        "Raw AI Response"
    ):

        st.code(
            st.session_state[
                "ai_response"
            ]
        )