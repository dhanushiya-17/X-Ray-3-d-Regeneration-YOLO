from PIL import Image, ImageDraw

subjects = [
    "Pat001",
    "Pat025",
    "Pat050",
    "Pat075",
    "Pat101",
]

angles = [0, 45, 90, 135, 180, 225, 270, 315]

CELL_WIDTH = 256
CELL_HEIGHT = 160

output = Image.new(
    "RGB",
    (CELL_WIDTH * 8, CELL_HEIGHT * len(subjects)),
    "white",
)

draw = ImageDraw.Draw(output)

for row, subject in enumerate(subjects):

    for col, angle in enumerate(angles):

        path = (
            f"data/drr_dataset/{subject}/"
            f"view_{angle:03d}.png"
        )

        image = Image.open(path).convert("RGB")
        image = image.resize((CELL_WIDTH, 128))

        x = col * CELL_WIDTH
        y = row * CELL_HEIGHT

        output.paste(image, (x, y + 25))

        draw.text(
            (x + 5, y + 5),
            f"{subject} {angle}°",
            fill="red",
        )

output.save("drr/dataset_qc.png")

print("Saved: drr/dataset_qc.png")

