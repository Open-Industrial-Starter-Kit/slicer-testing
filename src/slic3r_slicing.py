import argparse
import logging
import os
import re
import subprocess
import tempfile
import warnings
import xml.etree.ElementTree as ET
import subprocess
import tempfile
import cairosvg
from image_to_gcode import Image2gcode
from constants import Constants as constants
from svg2gcode.svg_to_gcode.svg_parser import parse_file
from svg2gcode.svg_to_gcode.compiler import Compiler, interfaces


def convert_slic3r_svg_to_images_data(root, layer_num=None):
    # Namespace
    ns = {
        'svg': 'http://www.w3.org/2000/svg',
        'slic3r': 'http://slic3r.org/namespaces/slic3r'
    }

    all_group_elements = root.findall(".//svg:g", ns)

    images_data = []

    # get a specific layer image data
    if len(layer_num) == 1:
        layer_group_elem = all_group_elements[layer_num[0]]
        svg_content = generate_svg_element(layer_group_elem,
                                           ns,
                                           root.get("width"),
                                           root.get("height"))

        # convert svg to image
        raster_image_content = cairosvg.svg2png(
            bytestring=svg_content)

        # each layer converted to svg element and raster image
        images_data.append({
            "svg": svg_content,
            "raster": raster_image_content
        })
    else:
        # Loop through groups and polygons
        for group_elem in all_group_elements[layer_num[0]: layer_num[1]]:
            svg_content = generate_svg_element(group_elem,
                                               ns,
                                               root.get("width"),
                                               root.get("height"))

            # convert svg to image
            raster_image_content = cairosvg.svg2png(
                bytestring=svg_content)

            # each layer converted to svg element and raster image
            images_data.append({
                "svg": svg_content,
                "raster": raster_image_content
            })

    return images_data


def generate_svg_element(group_elem, ns, width, height):
    polygon_elem = group_elem.findall("svg:polygon", ns)[0]
    path_elem = polygon_to_path(polygon_elem)
    # Minimal SVG with path
    svg_content = f'''
            <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
                {ET.tostring(path_elem, encoding='unicode')}
            </svg>
            '''
    return svg_content.encode('utf-8')


def polygon_to_path(polygon_elem):
    # Get the points attribute from the polygon
    points = polygon_elem.attrib.get("points", "").strip()
    if not points:
        return None  # or raise an error

    # Parse the points into a list of (x, y) tuples
    coords = [tuple(map(float, point.split(','))) for point in points.split()]

    # Construct the path data string
    path_data = f"M {coords[0][0]},{coords[0][1]}"
    for x, y in coords[1:]:
        path_data += f" L {x},{y}"
    path_data += " Z"  # Close the path

    # Create the new path element and copy attributes
    path_elem = ET.Element("path")
    path_elem.set("d", path_data)

    # Copy other attributes (like fill, stroke, etc.)
    for attrib in polygon_elem.attrib:
        if attrib != "points":
            path_elem.set(attrib, polygon_elem.attrib[attrib])

    path_elem.set("stroke", "black")

    return path_elem


def generate_gcode_for_svg(svg_file_content, svg_settings):
    svg_tmp_file_path = save_spooled_temp_file_to_disk(
        svg_file_content)
    gcode_tmp_file_path = generate_temp_gcode_file()
    try:
        # disable logging for the compiler function
        logging.disable()
        warnings.simplefilter("ignore")

        compiler = Compiler(interfaces.Gcode, params={
            'maximum_image_laser_power': int(svg_settings.get('power')),
            'image_movement_speed': int(svg_settings.get('speed')),
            'rapid_move': int(svg_settings.get('speedmoves')),
            'color_coded': 'black',
            'laser_mode': "constant",
            'pixel_size': float(100),
            'nofill': True
        })
        compiler.compile_to_file(
            gcode_tmp_file_path, svg_tmp_file_path, parse_file(svg_tmp_file_path, delta_origin=svg_settings.get('offset')))

        gcode_content = open(gcode_tmp_file_path, 'r').read()

        # remove an necessary commands
        gcode_content = gcode_content.replace(constants.INITIAL_COMMANDS, '')
        gcode_content = gcode_content.replace(constants.GRBL_END_PROGRAM, '')
        gcode_content = gcode_content.replace(constants.GRBL_COOLANT_OFF, '')

        return gcode_content

    except subprocess.CalledProcessError as e:
        print("Error:", e)
        return None

    finally:
        # Remove temp files
        os.remove(svg_tmp_file_path)
        os.remove(gcode_tmp_file_path)


def save_spooled_temp_file_to_disk(spooled_temp_file_content):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.svg') as temp_file:
        temp_file.write(spooled_temp_file_content)
    return temp_file.name


def generate_temp_gcode_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.gcode') as temp_file:
        return temp_file.name


def remove_comments(gcode_content):
    pattern = r';.*?\n'
    # Replace all occurrences of the pattern with an empty string
    gcode_content = re.sub(pattern, '', gcode_content)

    return gcode_content


##################################################################
if __name__ == "__main__":
    # defaults
    cfg = {
        "layer_height_default": 1,
        "layer_speed_default": 1000,
        "scale_default": 1,
        "rotate_default": 0,
        "speedmoves_default": 1000,
        "infill_pixelsize_default": (1, 1),
        "infill_offset_default": (0, 0),
        "infill_speed_default": 3000,
        "infill_power_default": 50,
        "infill_noise_default": 5,
        "contour_speed_default": 3000,
        "contour_power_default": 50,
    }

    parser = argparse.ArgumentParser(
        description="A tool to test slic3r and generate 2D gcode based on 3D model from raster images")
    parser.add_argument("--model-path", type=str,
                        required=True, help="3D model file path")
    parser.add_argument("--layer-height", type=float, metavar="<default:" + str(cfg["layer_height_default"])+">", default=cfg["layer_height_default"],
                        help="Layer height that to slice the 3D model")
    parser.add_argument("--scale", type=float,  metavar="<default:" + str(cfg["scale_default"])+">", default=cfg["scale_default"],
                        help="Factor for scaling input object", )
    parser.add_argument("--rotate", type=float,  metavar="<default:" + str(cfg["rotate_default"])+">", default=cfg["rotate_default"],
                        help="Rotation angle in degrees(0-360)")
    parser.add_argument("--gcode-output-path", type=str, required=True,
                        help="Generated gcode file path")
    parser.add_argument('--contour-speed', default=cfg["contour_speed_default"], metavar="<default:" + str(cfg["contour_speed_default"])+">",
                        type=int, help='contouring speed in mm/min')
    parser.add_argument('--contour-power', default=cfg["contour_power_default"], metavar="<default:" + str(cfg["contour_power_default"]) + ">",
                        type=int, help="sets laser power of line (path) contouring")
    parser.add_argument('--infill-image-pixel-size', default=cfg["infill_pixelsize_default"], nargs=2, metavar="<default:" + str(cfg["infill_pixelsize_default"])+">",
                        type=float, help="pixel size in mm (XY-axis): each image pixel is drawn this size")
    parser.add_argument('--infill-speed', default=cfg["infill_speed_default"], metavar="<default:" + str(cfg["infill_speed_default"])+">",
                        type=int, help='draw speed in mm/min')
    parser.add_argument('--infill-power', default=cfg["infill_power_default"], metavar="<default:" + str(cfg["infill_power_default"]) + ">",
                        type=int, help="maximum laser power while drawing (as a rule of thumb set to 1/3 of the maximum of a machine having a 5W laser)")
    parser.add_argument('--infill-image-size', default=None, nargs=2, metavar=('gcode-width', 'gcode-height'),
                        type=float, help="target gcode width and height in mm (default: not set and determined by pixelsize and image source resolution)")
    parser.add_argument('--infill-noise', default=cfg["infill_noise_default"], metavar="<default:" + str(cfg["infill_noise_default"]) + ">",
                        type=int, help="noise power level, do not burn pixels below this power level")
    parser.add_argument('--speed-moves', default=cfg["speedmoves_default"], metavar="<default:" + str(cfg["speedmoves_default"])+">",
                        type=float, help="length of zero burn zones in mm (0 sets no speedmoves): issue speed (G0) moves when skipping space of given length (or more)")
    parser.add_argument('--layer-num', nargs='*', default=None,
                        type=int, help="Generate gcode for specific layer(s) (default: generate gcode for all layers) (layer number) or (range start layer end layer)")
    parser.add_argument('--layer-speed', nargs='*', default=cfg["layer_speed_default"], metavar="<default:" + str(cfg["layer_speed_default"])+">",
                        type=int, help="Platform speed to move to the next layer")

    args = parser.parse_args()

    # generate svg file
    with tempfile.TemporaryFile(delete=False, suffix=".svg") as svg_tmp_file:
        subprocess.run([".\Slic3r\Slic3r-console.exe",
                        "--export-svg", args.model_path,
                        "--layer-height",  str(args.layer_height),
                        "--scale", str(args.scale),
                        "--rotate", str(args.rotate),
                        "--output", svg_tmp_file.name
                        ],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        tree = ET.parse(svg_tmp_file)
        root = tree.getroot()
        images_data = convert_slic3r_svg_to_images_data(
            root, args.layer_num)

        # # generate gcode from images
        if images_data:
            with open(args.gcode_output_path, "w") as gcode_file:
                z_value = 0
                image_width, image_height = float(root.get(
                    'width')), float(root.get('height'))
                user_image_size = args.infill_image_size if args.infill_image_size else (
                    image_width, image_height)

                svg_settings = {
                    'power': args.contour_power,
                    'speed': args.contour_speed,
                    'speedmoves': args.speed_moves,
                }

                raster_settings = {
                    'power': args.infill_power,
                    'speed': args.infill_speed,
                    'speedmoves': args.speed_moves,
                    'pixel_size': args.infill_image_pixel_size,
                    'size': args.infill_image_size,
                    'offset': ((image_width - int(image_width)) + (int(image_width) - user_image_size[0]) // 2,
                               (image_height - int(image_height)) + (int(image_height) - user_image_size[1]) // 2),
                    'noise': args.infill_noise
                }

                # add initial command as header for the file
                gcode_file.write(constants.INITIAL_COMMANDS)

                for i, image in enumerate(images_data):
                    print(
                        f"Processing layer: {i + 1} out of {len(images_data)} => layer height: {z_value} mm")

                    # move z axis based on the layer
                    gcode_content = f'\nG1 Z-{z_value}F{args.layer_speed}\n'

                    convertor = Image2gcode()
                    gcode_content += convertor.image_to_gcode(
                        image.get("raster"), raster_settings)

                    gcode_content += generate_gcode_for_svg(
                        image.get("svg"), svg_settings)

                    # remove comments
                    gcode_content = remove_comments(gcode_content)
                    # write data to file
                    gcode_file.write(gcode_content)
                    # calculate the z value for the next layer
                    z_value += args.layer_height

                gcode_file.write(constants.GRBL_END_PROGRAM)

    os.remove(svg_tmp_file.name)
