import zipfile
import xml.etree.ElementTree as ET

zip_path = "data/grazpedwri/grazpedwri-dx.zip"

wrist_images = 0
wrist_fractures = 0

with zipfile.ZipFile(zip_path, "r") as z:

    for xml_file in z.namelist():

        if (
            xml_file.startswith("folder_structure/pascalvoc/")
            and xml_file.endswith(".xml")
            and "_WRI-" in xml_file
        ):

            wrist_images += 1

            root = ET.fromstring(z.read(xml_file))

            for obj in root.findall("object"):

                name = obj.findtext("name", "").strip()

                if name == "fracture":
                    wrist_fractures += 1
                    break

print("Wrist annotation files:", wrist_images)
print("Wrist images with fracture:", wrist_fractures)

