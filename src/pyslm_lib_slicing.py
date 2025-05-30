import argparse
import io
import logging
import os
import re
import subprocess
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
import pyslm
from pyslm import hatching as hatching
from image_to_gcode import Image2gcode
from constants import Constants as constants
from svg2gcode.svg_to_gcode.svg_parser import parse_file
from svg2gcode.svg_to_gcode.compiler import Compiler, interfaces
import logging


def setup_hatching_pattern(args):
    infill = None
    if args.infill_type == "Basic":
        infill = hatching.Hatcher()
    elif args.infill_type == "Strip":
        infill = hatching.StripeHatcher()
        infill.stripeWidth = args.infill_strip_width
    elif args.infill_type == "Island":
        infill = hatching.IslandHatcher()
        infill.islandWidth = args.infill_island_width
        infill.islandOverlap = args.infill_island_overlap
        infill.islandOffset = args.infill_island_offset

    infill.hatchAngle = args.infill_angle
    infill.volumeOffsetHatch = args.infill_volume_offset
    infill.spotCompensation = args.infill_spot_compensation
    infill.numInnerContours = args.infill_inner_contour
    infill.numOuterContours = args.infill_outer_contour

    return infill


def slice_layer(z, args):
    geomSliceVector = solidPart.getVectorSlice(z)
    layer = infill.hatch(geomSliceVector)

    hatchGeoms, hatchContours, min_x, min_y, max_x, max_y, image_size = extract_layer_geometry_info(
        layer)

    # Calculate the offset of the layer
    offset_x = min_x
    offset_y = min_y

    if args.infill_image_size:
        image_size = args.infill_image_size

    img_infill = generate_infill_raster_image(
        hatchGeoms, min_x, min_y, max_x, max_y, image_size, args.infill_resolution)

    svg_contour = generate_contour_svg(
        hatchContours, min_x, min_y, max_x, max_y, image_size)

    svg_settings = {
        'power': args.contour_power,
        'speed': args.contour_speed,
        'speedmoves': args.speed_moves,
        'offset': (offset_x, offset_y)
    }

    image_settings = {
        'power': args.infill_power,
        'speed': args.infill_speed,
        'speedmoves': args.speed_moves,
        'pixel_size': args.infill_image_pixel_size,
        'size': image_size,
        'offset': (offset_x, offset_y),
        'noise': args.infill_noise
    }

    return img_infill, svg_contour, image_settings, svg_settings


def extract_layer_geometry_info(layer):
    hatchGeoms = layer.getHatchGeometry()
    hatchContours = layer.getContourGeometry()
    min_x, min_y, max_x, max_y = get_global_bounds(hatchGeoms, hatchContours)
    image_size = compute_image_size_from_bounds(min_x, min_y, max_x, max_y)

    return hatchGeoms, hatchContours, min_x, min_y, max_x, max_y, image_size


def generate_infill_raster_image(hatchGeoms, min_x, min_y, max_x, max_y, image_size, resolution):
    # Apply resolution scaling
    scaled_size = (min((image_size[0] * resolution), 5000),
                   min((image_size[1] * resolution), 5000))

    # Recalculate scale factors for the scaled image
    scale_x, scale_y = calculate_scale_factor(
        min_x, min_y, max_x, max_y, scaled_size)

    # Create a high-resolution image
    img = Image.new("RGB", scaled_size, "black")
    draw = ImageDraw.Draw(img)

    for line_points in modified_hatches_lines(hatchGeoms, min_x, min_y, scale_x, scale_y):
        draw.line(line_points, fill='white', width=1)

    img_bytes_io = io.BytesIO()
    img.save(img_bytes_io, format='PNG')
    img_bytes = img_bytes_io.getvalue()
    return img_bytes


def generate_contour_svg(hatchContours, min_x, min_y, max_x, max_y, image_size):
    scale_x, scale_y = calculate_scale_factor(
        min_x, min_y, max_x, max_y, image_size)   # Create root <svg> element
    svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg",
                     width=str(image_size[0]), height=str(image_size[1]))

    # Process hatch lines
    for contour_line_points in modified_contour_line(hatchContours, min_x, min_y, scale_x, scale_y):
        # Add each hatch line to the SVG
        path = create_path_from_coords(contour_line_points)
        # Create <path> element
        ET.SubElement(svg, 'path', d=path, fill="none",
                      stroke='black', stroke_width=str(0.1))

    return ET.tostring(svg, encoding='utf-8')


def compute_image_size_from_bounds(min_x, min_y, max_x, max_y):
    width_mm = max_x - min_x
    height_mm = max_y - min_y
    return (int(width_mm), int(height_mm))


def create_path_from_coords(coords):
    # Convert coords to SVG path string
    path_data = "M " + " L ".join([f"{x:.2f},{y:.2f}" for x, y in coords])
    return path_data


def calculate_scale_factor(min_x, min_y, max_x, max_y, image_size):
    scale_x = image_size[0] / (max_x - min_x)
    scale_y = image_size[1] / (max_y - min_y)
    return scale_x, scale_y


def get_global_bounds(hatchGeoms, contourGeoms):
    all_coords = []

    for hatch in hatchGeoms:
        all_coords.append(hatch.coords)
    for contour in contourGeoms:
        all_coords.append(contour.coords)

    all_coords = np.vstack(all_coords)
    min_x, min_y = np.min(all_coords, axis=0)
    max_x, max_y = np.max(all_coords, axis=0)
    return min_x, min_y, max_x, max_y


def modified_contour_line(contours, min_x, min_y, scale_x, scale_y):
    all_contours = []
    for contour in contours:
        coords = np.array(contour.coords)
        coords = (coords - [min_x, min_y]) * [scale_x, scale_y]
        all_contours.append([tuple(p) for p in coords])
    return all_contours


def modified_hatches_lines(hatchGeoms, min_x, min_y, scale_x, scale_y):
    hatches = np.vstack([hatchGeom.coords.reshape(-1, 2, 2)
                        for hatchGeom in hatchGeoms])
    modified_lines = []

    for line_segment in hatches:
        modified_points = []
        for wx, wy in line_segment:
            sx = (wx - min_x) * scale_x
            sy = (wy - min_y) * scale_y
            modified_points.append((sx, sy))

        modified_lines.append(modified_points)

    return modified_lines


def generate_gcode_for_layer(gcode_file, img_infill, svg_contour, svg_settings, raster_settings, z, layer_speed):

    # move z axis based on the layer
    gcode_content = f'G1 Z{z}F{layer_speed}\n'

    gcode_content += generate_gcode_for_svg(
        svg_contour, svg_settings)

    convertor = Image2gcode()
    gcode_content += convertor.image_to_gcode(
        img_infill, raster_settings)

    # remove comments
    gcode_content = remove_comments(gcode_content)
    # write data to file
    gcode_file.write(gcode_content)


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
        "rotate_default": (0, 0, 0),
        "infill_default": 'Basic',
        "infill_resolution_default": 50,
        "infill_angle_default": 0,
        "infill_volume_offset_default": 0.1,
        "infill_spot_compensation_default": 0.1,
        "infill_island_offset_default": 0.5,
        "infill_island_width_default": 5,
        "infill_island_overlap_default": 0.1,
        "infill_inner_contour_num_default": 0,
        "infill_outer_contour_num_default": 0,
        "infill_strip_width_default": 0.1,
        # "infill_sort_default": 'Alternate',
        "infill_image_size_default": (500, 500),
        "speedmoves_default": 1000,
        "infill_pixelsize_default": (1, 1),
        "infill_offset_default": (0, 0),
        "infill_speed_default": 3000,
        "infill_power_default": 50,
        "infill_noise_default": 5,
        "contour_speed_default": 3000,
        "contour_power_default": 50,
    }

    infill_choices = ['Basic', 'Strip', 'Island']
    # infill_sort_choices = ['Base', 'Unidirectional', 'Chain', 'HatchDirectional',
    #                        'Flip', 'Alternate', 'Linear', 'Greedy']

    parser = argparse.ArgumentParser(
        description="A tool to test pyslm library in python and generate 2D gcode based on 3D model from raster images")
    parser.add_argument("--model-path", type=str,
                        required=True, help="3D model file path")
    parser.add_argument("--layer-height", type=float, metavar="<default:" + str(cfg["layer_height_default"])+">", default=cfg["layer_height_default"],
                        help="Layer height that to slice the 3D model")
    parser.add_argument("--scale", type=float,  metavar="<default:" + str(cfg["scale_default"])+">", default=cfg["scale_default"],
                        help="Factor for scaling input object", )
    parser.add_argument("--rotate", type=float, nargs=3, metavar="<default:" + str(cfg["rotate_default"])+">", default=cfg["rotate_default"],
                        help="Rotation angle in degrees(0-360). Three axis (X-Y-Z)")
    parser.add_argument("--infill-type", type=str, metavar="<default:" + str(cfg["infill_default"])+">", default=cfg["infill_default"], choices=infill_choices,
                        help="The type of hatching (infill) you want to use (different patterns of infill)")
    parser.add_argument("--infill-angle", type=float, metavar="<default:" + str(cfg["infill_angle_default"])+">", default=cfg["infill_angle_default"],
                        help="Hatching angle (unit: °)")
    parser.add_argument("--infill-volume-offset", type=float, metavar="<default:" + str(cfg["infill_volume_offset_default"])+">", default=cfg["infill_volume_offset_default"],
                        help="An additional offset may be added(positive or negative) between the contour and the internal hatching (unit: mm)")
    parser.add_argument("--infill-spot-compensation", type=float, metavar="<default:" + str(cfg["infill_spot_compensation_default"])+">", default=cfg["infill_spot_compensation_default"],
                        help="The spot(laser point) compensation factor is the distance to offset the outer-boundary and other internal hatch features in order to factor in the exposure radius of the laser.  (unit: mm)")
    parser.add_argument("--infill-island-width", type=float, metavar="<default:" + str(cfg["infill_island_width_default"])+">", default=cfg["infill_island_width_default"],
                        help="Island Size - the length of each size of the square island (unit: mm)")
    parser.add_argument("--infill-island-overlap", type=float, metavar="<default:" + str(cfg["infill_island_overlap_default"])+">", default=cfg["infill_island_overlap_default"],
                        help="the overlap between adjacent islands (unit: mm)")
    parser.add_argument("--infill-island-offset", type=float, metavar="<default:" + str(cfg["infill_island_offset_default"])+">", default=cfg["infill_island_offset_default"],
                        help="The island offset is the relative distance(hatch spacing) to move the scan vectors between adjacent checkers. (unit: mm)")
    # parser.add_argument("--infill-sort", type=str, metavar="<default:" + str(cfg["infill_sort_default"])+">", default=cfg["infill_sort_default"], choices=infill_sort_choices,
    #                     help="Sorts the scan vectors in a particular order")
    parser.add_argument("--infill-inner-contour", type=int, metavar="<default:" + str(cfg["infill_inner_contour_num_default"])+">", default=cfg["infill_inner_contour_num_default"],
                        help="Number of inner contour")
    parser.add_argument("--infill-outer-contour", type=int, metavar="<default:" + str(cfg["infill_outer_contour_num_default"])+">", default=cfg["infill_outer_contour_num_default"],
                        help="Number of outer contour")
    parser.add_argument("--infill-strip-width", type=float, metavar="<default:" + str(cfg["infill_strip_width_default"])+">", default=cfg["infill_strip_width_default"],
                        help="Used in the strip infill type to define the strip width vertically (unit: mm)")
    parser.add_argument("--gcode-output-path", type=str, required=True,
                        help="Generated gcode file path")
    parser.add_argument('--contour-speed', default=cfg["contour_speed_default"], metavar="<default:" + str(cfg["contour_speed_default"])+">",
                        type=int, help='contourting speed in mm/min')
    parser.add_argument('--contour-power', default=cfg["contour_power_default"], metavar="<default:" + str(cfg["contour_power_default"]) + ">",
                        type=int, help="sets laser power of line (path) contourting")
    parser.add_argument('--infill-image-pixel-size', default=cfg["infill_pixelsize_default"], nargs=2, metavar="<default:" + str(cfg["infill_pixelsize_default"])+">",
                        type=float, help="pixel size in mm (XY-axis): each image pixel is drawn this size")
    parser.add_argument('--infill-speed', default=cfg["infill_speed_default"], metavar="<default:" + str(cfg["infill_speed_default"])+">",
                        type=int, help='draw speed in mm/min')
    parser.add_argument('--infill-power', default=cfg["infill_power_default"], metavar="<default:" + str(cfg["infill_power_default"]) + ">",
                        type=int, help="maximum laser power while drawing (as a rule of thumb set to 1/3 of the maximum of a machine having a 5W laser)")
    parser.add_argument('--infill-image-size', default=None, nargs=2, metavar="<default:" + str(cfg["infill_image_size_default"]) + ">",
                        type=int, help="target gcode width and height in mm (default: not set and determined by pixelsize and image source resolution)")
    parser.add_argument('--infill-offset', default=cfg["infill_offset_default"], nargs=2, metavar=('X-off', 'Y-off'),
                        type=float, help="laser drawing starts at offset in mm (default not set, --center cannot be set at the same time)")
    parser.add_argument('--infill-noise', default=cfg["infill_noise_default"], metavar="<default:" + str(cfg["infill_noise_default"]) + ">",
                        type=int, help="noise power level, do not burn pixels below this power level")
    parser.add_argument('--infill-resolution', default=cfg["infill_resolution_default"], metavar="<default:" + str(cfg["infill_resolution_default"])+">",
                        type=int, help="The quality of the generated infill images")
    parser.add_argument('--speed-moves', default=cfg["speedmoves_default"], metavar="<default:" + str(cfg["speedmoves_default"])+">",
                        type=float, help="length of zero burn zones in mm (0 sets no speedmoves): issue speed (G0) moves when skipping space of given length (or more)")
    parser.add_argument('--layer-num', nargs='*', default=None,
                        type=int, help="Generate gcode for specific layer(s) (default: generate gcode for all layers) (layer number) or (range start layer end layer)")
    parser.add_argument('--layer-speed', nargs='*', default=cfg["layer_speed_default"], metavar="<default:" + str(cfg["layer_speed_default"])+">",
                        type=int, help="Platform speed to move to the next layer")

    args = parser.parse_args()


if __name__ == "__main__":
    # Imports the part and sets the geometry to an STL file
    solidPart = pyslm.Part('Model')
    solidPart.setGeometry(args.model_path, fixGeometry=False)
    solidPart.rotation = [args.rotate[0], args.rotate[1], args.rotate[2]]
    solidPart.scaleFactor = args.scale
    solidPart.dropToPlatform()

    # Set the base hatching parameters which are generated within infill
    infill = setup_hatching_pattern(args)

    # Slice the object at Z height
    with open(args.gcode_output_path, "w") as gcode_file:
        # add initial command as header for the file
        gcode_file.write(constants.INITIAL_COMMANDS)

        # to keep all the layers on the same center
        global_center = (
            (solidPart.boundingBox[1] - solidPart.boundingBox[0]) / 2,
            (solidPart.boundingBox[3] - solidPart.boundingBox[2]) / 2)

        if args.layer_num:
            if args.layer_num and len(args.layer_num) == 1:
                # Slice the object at Z and get the boundaries
                z = args.layer_num[0] * args.layer_height
                # get layer data
                image_infill, svg_contour, image_settings, svg_settings = slice_layer(
                    z, args)

                # prevent this function from printing to the stdout
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")

                generate_gcode_for_layer(
                    gcode_file, image_infill, svg_contour, svg_settings, image_settings, z, args.layer_speed)

            else:
                for i in np.arange(args.layer_num[0], args.layer_num[1] + 1):
                    z = i * args.layer_height
                    print(
                        f"Processing layer: {int(i)} out of {args.layer_num[1]} => layer height: {z} mm")

                    # get layer data
                    image_infill, svg_contour, image_settings, svg_settings = slice_layer(
                        z, args)

                    generate_gcode_for_layer(
                        gcode_file, image_infill, svg_contour, svg_settings, image_settings, z, args.layer_speed)
        else:
            for z in np.arange(args.layer_height, solidPart.boundingBox[5], args.layer_height):
                print(
                    f"Processing layer: {int(z / args.layer_height)} out of {int(solidPart.boundingBox[5]/args.layer_height)} => layer height: {z} mm")

                # get layer data
                image_infill, svg_contour, image_settings, svg_settings = slice_layer(
                    z, args)

                generate_gcode_for_layer(
                    gcode_file, image_infill, svg_contour, svg_settings, image_settings, z, args.layer_speed)

        # end of gcode file
        gcode_file.write(constants.GRBL_END_PROGRAM)
