import argparse
import io
import logging
import os
import re
import subprocess
import tempfile
import warnings
import subprocess
import tempfile
import zipfile
from image_to_gcode import Image2gcode
from constants import Constants as constants
from svg2gcode.svg_to_gcode.svg_parser import parse_file
from svg2gcode.svg_to_gcode.compiler import Compiler, interfaces
from PIL import Image
import os
from PIL import Image
import numpy as np
from autotrace import Bitmap, Color, VectorFormat
import xml.etree.ElementTree as ET


def generate_gcode_for_sliced_layers(zip_path, args):
    # Slice the object at Z height
    with open(args.gcode_output_path, "w") as gcode_file:
        # Add initial command as header for the file
        gcode_file.write(constants.INITIAL_COMMANDS)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all .png files in the zip archive
            png_files = [file for file in zip_ref.namelist()
                         if file.endswith('.png')]

            # Determine which files to process
            if args.layer_num:
                # Handle single or multiple layers based on input range
                if len(args.layer_num) == 1:
                    layers_to_process = [args.layer_num[0]]
                else:
                    layers_to_process = range(
                        args.layer_num[0] - 1, args.layer_num[1] + 1)
            else:
                # If no specific layers are provided, process all layers
                layers_to_process = range(len(png_files))

            # Process each layer
            for i, layer in enumerate(layers_to_process):
                z = layer * args.layer_height  # Calculate the Z position for the layer
                print(
                    f"Processing layer: {i + 1} out of {len(layers_to_process)} => layer height: {z} mm")

                file = png_files[layer]
                with zip_ref.open(file, 'r') as png_file:
                    generate_gcode_for_layer(
                        gcode_file, png_file, z, args)


def generate_gcode_for_layer(gcode_file, png_file, z, args):
    # move z axis based on the layer
    gcode_content = f'G1 Z{z}F{args.layer_speed}\n'

    img_infill = png_file.read()

    # generate a contour for the infill image
    svg_contour = generate_svg_contour(img_infill)

    svg_settings, raster_settings = get_image_settings(args)

    convertor = Image2gcode()
    gcode_content += convertor.image_to_gcode(
        img_infill, raster_settings)

    gcode_content += generate_gcode_for_svg(
        svg_contour, svg_settings)

    # remove comments
    gcode_content = remove_comments(gcode_content)
    # write data to file
    gcode_file.write(gcode_content)


def generate_svg_contour(img_infill):
    img = Image.open(io.BytesIO(img_infill))
    bitmap = Bitmap(np.asarray(img.convert('RGB')))

    # Trace the bitmap.
    vector1 = bitmap.trace(
        despeckle_level=20,
        despeckle_tightness=8,
        error_threshold=0.1,
        line_threshold=0,
        corner_threshold=10,
        filter_iterations=0,
        remove_adjacent_corners=True,
        tangent_surround=0,
        noise_removal=1,
        color_count=2,
        background_color=Color(0, 0, 0))

    # modify the generated svg
    root = ET.fromstring(vector1.encode(VectorFormat.SVG))
    ns = {
        'svg': 'http://www.w3.org/2000/svg',
    }

    path_element = root.find(".//svg:path", ns)
    if path_element is not None:
        del path_element.attrib['style']
        path_element.set('stroke', 'black')
        path_element.set('fill', 'none')

    return ET.tostring(root)


def get_image_settings(args):
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
        'offset': (0, 0),
        'noise': args.infill_noise
    }

    return svg_settings, raster_settings


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
        "infill_image_size_default": (200, 200),
        "contour_speed_default": 3000,
        "contour_power_default": 50,
        "bed_size_default": (100, 100),
        "hollowing_closing_distance_default": 2,
        "hollowing_min_thickness_default": 3,
        "hollowing_quality_default": 0.5
    }

    parser = argparse.ArgumentParser(
        description="A tool to test prusa slicer and generate 2D gcode based on 3D model from raster images")
    parser.add_argument("--model-path", required=True,
                        type=str, help="3D model file path")
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
    parser.add_argument('--infill-image-size', default=cfg["infill_image_size_default"], nargs=2, metavar=('gcode-width', 'gcode-height'),
                        type=float, help="target gcode width and height in mm (default: not set and determined by pixelsize and image source resolution)")
    parser.add_argument('--infill-noise', default=cfg["infill_noise_default"], metavar="<default:" + str(cfg["infill_noise_default"]) + ">",
                        type=int, help="noise power level, do not burn pixels below this power level")
    parser.add_argument('--speed-moves', default=cfg["speedmoves_default"], metavar="<default:" + str(cfg["speedmoves_default"])+">",
                        type=float, help="length of zero burn zones in mm (0 sets no speedmoves): issue speed (G0) moves when skipping space of given length (or more)")
    parser.add_argument('--layer-num', nargs='*', default=None,
                        type=int, help="Generate gcode for specific layer(s) (default: generate gcode for all layers) (layer number) or (range start layer end layer)")
    parser.add_argument('--layer-speed', nargs='*', default=cfg["layer_speed_default"], metavar="<default:" + str(cfg["layer_speed_default"])+">",
                        type=int, help="Platform speed to move to the next layer")
    parser.add_argument('--bed-size', nargs=2, default=cfg["bed_size_default"], metavar="<default:" + str(cfg["bed_size_default"])+">",
                        type=int, help="Platform size that the model positioned on (unit: pixels)")
    parser.add_argument("--hollowing-enable", action="store_true",
                        help="Hollow out a model to have an empty interior")
    parser.add_argument("--hollowing-closing-distance", type=float, default=cfg["hollowing_closing_distance_default"],
                        metavar="<default:" +
                        str(cfg["hollowing_closing_distance_default"]) + ">",
                        help="Interior rounding control. Higher = rounder. (mm)")
    parser.add_argument("--hollowing-min-thickness", type=float, default=cfg["hollowing_min_thickness_default"],
                        metavar="<default:" +
                        str(cfg["hollowing_min_thickness_default"]) + ">",
                        help="Minimum wall thickness of the hollowed model. (mm)")
    parser.add_argument("--hollowing-quality", type=float, default=cfg["hollowing_quality_default"],
                        metavar="<default:" +
                        str(cfg["hollowing_quality_default"]) + ">",
                        help="Accuracy vs performance tradeoff (0.0 - 1.0)")
    parser.add_argument("--supports-enable", action="store_true",
                        help="Generate supports for the models")
    parser.add_argument("--support-base-diameter", type=float, default=4,
                        help="Diameter in mm of the pillar base (default: 4)")
    parser.add_argument("--support-base-height", type=float, default=1,
                        help="The height of the pillar base cone (default: 1)")
    parser.add_argument("--support-base-safety-distance", type=float, default=1,
                        help="Minimum distance of the pillar base from the model (default: 1)")
    parser.add_argument("--support-buildplate-only", action="store_true",
                        help="Only create support if it lies on a build plate")
    parser.add_argument("--support-critical-angle", type=float, default=45,
                        help="Default angle for connecting support sticks and junctions (default: 45°)")
    parser.add_argument("--support-enforcers-only", action="store_true",
                        help="Only create support if it lies in a support enforcer")
    parser.add_argument("--support-head-front-diameter", type=float, default=0.4,
                        help="Diameter of the pointing side of the head (default: 0.4)")
    parser.add_argument("--support-head-penetration", type=float, default=0.2,
                        help="Pinhead penetration depth (default: 0.2)")
    parser.add_argument("--support-head-width", type=float, default=1,
                        help="Width from back to front sphere center (default: 1)")
    parser.add_argument("--support-max-bridge-length", type=float, default=15,
                        help="Max length of a bridge (default: 15)")
    parser.add_argument("--support-max-pillar-link-distance", type=float, default=10,
                        help="Max distance for pillars to link (default: 10)")
    parser.add_argument("--support-max-weight-on-model", type=float, default=10,
                        help="Max support tree weight on model (default: 10)")
    parser.add_argument("--support-object-elevation", type=float, default=5,
                        help="Support elevation (default: 5)")
    parser.add_argument("--support-pillar-diameter", type=float, default=1,
                        help="Support pillar diameter (default: 1)")
    parser.add_argument("--support-pillar-widening-factor", type=float, default=0.5,
                        help="Factor for pillar widening (default: 0.5)")
    parser.add_argument("--support-points-density-relative", type=float, default=100,
                        help="Relative density of support points in percentage (default: 100 %%)")
    parser.add_argument("--support-points-minimal-distance", type=float, default=1,
                        help="Minimum distance between support points (default: 1)")
    parser.add_argument("--support-small-pillar-diameter-percent", type=float, default=50,
                        help="Percentage for small pillars (default: 50 %%)")

    args = parser.parse_args()

# generate svg file
    with tempfile.TemporaryFile(delete=False, suffix=".zip") as zip_tmp_file:
        # Build slicer command
        slicer_cmd = [
            ".\\PrusaSlicer\\prusa-slicer-console.exe",
            "--export-sla", args.model_path,
            "--layer-height", str(args.layer_height),
            "--scale", str(args.scale),
            "--rotate", str(args.rotate),
            "--sla-archive-format", "SL1",
            "--center", f"{args.bed_size[0] / 2}, {args.bed_size[1] / 2}",
            "--display-pixels-x", str(args.infill_image_size[0]),
            "--display-pixels-y", str(args.infill_image_size[1]),
            "--display-width", str(args.bed_size[0]),
            "--display-height", str(args.bed_size[1]),
            "--output", zip_tmp_file.name
        ]

        # Add hollowing options if enabled
        if args.hollowing_enable:
            slicer_cmd += [
                "--hollowing-enable",
                "--hollowing-closing-distance", str(
                    args.hollowing_closing_distance),
                "--hollowing-min-thickness", str(args.hollowing_min_thickness),
                "--hollowing-quality", str(args.hollowing_quality)
            ]

        if args.supports_enable:
            slicer_cmd += [
                "--supports-enable",
                "--support-base-diameter", str(args.support_base_diameter),
                "--support-base-height", str(args.support_base_height),
                "--support-base-safety-distance", str(
                    args.support_base_safety_distance),
                "--support-critical-angle", str(args.support_critical_angle),
                "--support-head-front-diameter", str(
                    args.support_head_front_diameter),
                "--support-head-penetration", str(
                    args.support_head_penetration),
                "--support-head-width", str(args.support_head_width),
                "--support-max-bridge-length", str(
                    args.support_max_bridge_length),
                "--support-max-pillar-link-distance", str(
                    args.support_max_pillar_link_distance),
                "--support-max-weight-on-model", str(
                    args.support_max_weight_on_model),
                "--support-object-elevation", str(
                    args.support_object_elevation),
                "--support-pillar-diameter", str(args.support_pillar_diameter),
                "--support-pillar-widening-factor", str(
                    args.support_pillar_widening_factor),
                "--support-points-density-relative", str(
                    args.support_points_density_relative),
                "--support-points-minimal-distance", str(
                    args.support_points_minimal_distance),
                "--support-small-pillar-diameter-percent", str(
                    args.support_small_pillar_diameter_percent)
            ]

            if args.support_buildplate_only:
                slicer_cmd.append("--support-buildplate-only")
            if args.support_enforcers_only:
                slicer_cmd.append("--support-enforcers-only")

        subprocess.run(
            slicer_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        generate_gcode_for_sliced_layers(zip_tmp_file, args)
