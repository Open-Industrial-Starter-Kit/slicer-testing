import io
import math
import numpy as np
from PIL import Image, ImageOps


class Image2gcode:
    def __init__(self, power_func=None, transformation_func=None):
        self._power = power_func if power_func is not None else self.linear_power
        self._transformation = transformation_func

    def _load_image(self, img_bytes, pixel_size_tuple, image_size=None):
        img = Image.open(io.BytesIO(img_bytes))
        # Invert: black (0) becomes white (255), white (255) becomes black (0)
        img = ImageOps.invert(img.convert('RGB'))
        # For engraving, black parts of original image should get power.
        # After invert, original black (strong) is 255, original white (weak) is 0.
        # The self._power function will treat smaller values as higher power.
        img = img.convert('RGBA')

        img_background = Image.new(
            mode="RGBA", size=img.size, color=(255, 255, 255, 255))
        img = Image.alpha_composite(img_background, img)

        # Calculate new dimensions for resizing
        # The image is resized so that each pixel in the new image corresponds to one G-code step
        if image_size:
            new_width = int(round(image_size[0] / pixel_size_tuple[0]))
            new_height = int(round(image_size[1] / pixel_size_tuple[1]))
        else:
            new_width = int(round(img.size[0] / pixel_size_tuple[0]))
            new_height = int(round(img.size[1] / pixel_size_tuple[1]))

        new_width = max(1, new_width)
        new_height = max(1, new_height)

        img_resized = img.resize(
            (new_width, new_height), Image.Resampling.LANCZOS).convert("L")
        img_resized.show()
        img_arr = np.array(img_resized)
        # Flip for G-code coordinate system
        img_arr_flipped = np.flipud(img_arr)
        return img_arr_flipped

    # Calculates the number of decimal places for X and Y G-code coordinates.
    def _calculate_precision(self, pixel_size_tuple):
        x_str = str(pixel_size_tuple[0])
        y_str = str(pixel_size_tuple[1])
        X_prec = len(x_str.split('.')[1]) if '.' in x_str else 0
        Y_prec = len(y_str.split('.')[1]) if '.' in y_str else 0
        return X_prec, Y_prec

    # Gets the rounded initial logical X, Y G-code coordinates from offset.
    def _get_initial_logical_coordinates(self, offset_tuple, X_prec, Y_prec):
        X = round(offset_tuple[0], X_prec)
        Y = round(offset_tuple[1], Y_prec)
        return X, Y

    # (X,Y) coordinates and rounds the result to the specified precision.
    def _apply_transformation(self, XY_tuple, X_prec, Y_prec):
        if self._transformation is None:
            # No transformation, just ensure rounding to precision
            return round(XY_tuple[0], X_prec), round(XY_tuple[1], Y_prec)

        Xd, Yd = self._transformation(XY_tuple)
        return round(Xd, X_prec), round(Yd, Y_prec)

    @staticmethod
    # This revised formula maps high pixel values (255, from original dark areas) to high power.
    def linear_power(pixel_value, max_power, power_offset=0):
        return power_offset + round((1.0 - float(pixel_value/255)) * (max_power - power_offset))

    @staticmethod
    # Calculates Euclidean distance between two points A=(x1,y1) and B=(x2,y2).
    def distance(A_tuple, B_tuple):
        return math.sqrt((A_tuple[0] - B_tuple[0])**2 + (A_tuple[1] - B_tuple[1])**2)

    # Adds initial G-code commands: G0 move to transformed start, and G1 speed setting.
    def _add_initial_gcode_setup(self, gcode_commands_list, X_logical_start, Y_logical_start, args, X_prec, Y_prec):
        Xd, Yd = self._apply_transformation(
            (X_logical_start, Y_logical_start), X_prec, Y_prec)
        # gcode_commands_list.append(f"G0 X{Xd} Y{Yd}")
        gcode_commands_list.append(f"G1 F{args['speed']}")

    def _ensure_head_at_position(self, target_logical_coord_tuple, current_head_logical_coord_tuple, args, X_prec, Y_prec):
        # Round logical coordinates for comparison to avoid floating point issues
        if round(target_logical_coord_tuple[0], X_prec) == round(current_head_logical_coord_tuple[0], X_prec) and \
           round(target_logical_coord_tuple[1], Y_prec) == round(current_head_logical_coord_tuple[1], Y_prec):
            return ""  # Head is already at the target logical position

        # Apply transformation for G-code output and distance calculation
        target_transformed_X, target_transformed_Y = self._apply_transformation(
            target_logical_coord_tuple, X_prec, Y_prec)
        current_head_transformed_X, current_head_transformed_Y = self._apply_transformation(
            current_head_logical_coord_tuple, X_prec, Y_prec)

        move_gcode = ""
        speedmoves_threshold = args.get("speedmoves")

        # Check if a fast move (G0) is warranted
        if speedmoves_threshold is not None and \
           self.distance((current_head_transformed_X, current_head_transformed_Y),
                         (target_transformed_X, target_transformed_Y)) > speedmoves_threshold:
            # G1 to ensure next move is controlled speed
            move_gcode = f"G0 X{target_transformed_X} Y{target_transformed_Y}\nG1"
        else:
            # Standard speed move with laser off (S0)
            move_gcode = f"G1 X{target_transformed_X} Y{target_transformed_Y} S0"

        return move_gcode

    # Processes a single scan line of pixels (including a terminator) and generates G-code segments.
    def _process_scan_line_pixels(self,
                                  line_pixel_data_terminated,
                                  line_processing_start_X_logical,
                                  current_Y_logical_for_line,
                                  initial_head_logical_on_line_entry,
                                  is_left_to_right_scan,
                                  args, X_prec, Y_prec):
        gcode_segments_for_this_line = []
        # current_gcode_X_logical tracks the G-code X coordinate for the *start* of the current pixel being evaluated
        current_gcode_X_logical = line_processing_start_X_logical
        # Tracks head's logical position
        head_logical_tracker = initial_head_logical_on_line_entry

        pixel_width_logical = args["pixel_size"][0]
        # Directional step for iterating through logical X G-code coordinates along the scan path
        x_step_logical = pixel_width_logical if is_left_to_right_scan else -pixel_width_logical

        # Initialize based on the first actual pixel of the (potentially flipped) line data
        first_pixel_val = line_pixel_data_terminated[0]

        power_of_previous_segment = self._power(
            first_pixel_val, args["power"])

        # The logical X G-code coordinate where the current continuous power segment started.
        current_segment_start_X_logical = current_gcode_X_logical

        # Iterate through each pixel *position* and its *value*, plus one for the terminator
        for i in range(len(line_pixel_data_terminated)):
            current_pixel_val = line_pixel_data_terminated[i]
            power_for_current_pixel = self._power(
                current_pixel_val, args["power"])

            # A segment ends if power changes OR we are at the terminator pixel (which finalizes the last segment)
            if power_for_current_pixel != power_of_previous_segment or i == len(line_pixel_data_terminated) - 1:
                # `current_gcode_X_logical` is the coordinate of the *start* of the pixel whose processing
                # has just revealed a power change or is the terminator.
                # The segment with `power_of_previous_segment` extended *up to* this `current_gcode_X_logical`.
                segment_draw_target_X_logical = current_gcode_X_logical

                # Only generate G-code if power is above noise threshold
                if power_of_previous_segment > args["noise"]:
                    # 1. Ensure head is positioned at the start of this "active" (non-noise) segment.
                    gcode_to_reach_segment_start = self._ensure_head_at_position(
                        (current_segment_start_X_logical,
                         current_Y_logical_for_line),
                        head_logical_tracker, args, X_prec, Y_prec
                    )
                    if gcode_to_reach_segment_start:
                        gcode_segments_for_this_line.append(
                            gcode_to_reach_segment_start)
                    head_logical_tracker = (
                        current_segment_start_X_logical, current_Y_logical_for_line)

                    # 2. Generate the G1 command to draw the segment.
                    # The move is TO segment_draw_target_X_logical with power_of_previous_segment.
                    target_Xd, target_Yd = self._apply_transformation(
                        (segment_draw_target_X_logical,
                         current_Y_logical_for_line), X_prec, Y_prec
                    )

                    gcode_cmd_parts = [f"G1 X{target_Xd}"]
                    if self._transformation is not None:  # Add Y if transforming, as it might change from the last Y
                        gcode_cmd_parts.append(f"Y{target_Yd}")
                    gcode_cmd_parts.append(f"S{power_of_previous_segment}")
                    gcode_segments_for_this_line.append(
                        " ".join(gcode_cmd_parts))

                    head_logical_tracker = (
                        segment_draw_target_X_logical, current_Y_logical_for_line)

                # Prepare for the next segment
                power_of_previous_segment = power_for_current_pixel
                # New segment starts where old one ended
                current_segment_start_X_logical = segment_draw_target_X_logical

            # Advance the logical X G-code coordinate for the *next* pixel's start position,
            # unless we just processed the terminator pixel.
            if i < len(line_pixel_data_terminated) - 1:
                current_gcode_X_logical = round(
                    current_gcode_X_logical + x_step_logical, X_prec)

        return head_logical_tracker, gcode_segments_for_this_line

    # Converts an image to G-code commands for laser engraving or similar processes.
    def image_to_gcode(self, img_bytes, args):
        # Load and preprocess the image based on pixel size
        image_array = self._load_image(
            img_bytes, args["pixel_size"], args["size"])
        X_prec, Y_prec = self._calculate_precision(args["pixel_size"])

        # Get initial logical G-code coordinates from offset
        initial_logical_X, initial_logical_Y = self._get_initial_logical_coordinates(
            args["offset"], X_prec, Y_prec)

        all_gcode_commands = []
        # Add G0 move to the (transformed) starting position and set G1 speed
        self._add_initial_gcode_setup(
            all_gcode_commands, initial_logical_X, initial_logical_Y, args, X_prec, Y_prec)

        # Head's logical G-code position, starts at the defined offset.
        current_head_logical = (initial_logical_X, initial_logical_Y)

        is_scan_left_to_right = True  # Start scanning from Left to Right
        num_pixel_columns_in_image = image_array.shape[1]
        pixel_width_gcode_units = args["pixel_size"][0]
        pixel_height_gcode_units = args["pixel_size"][1]

        # Iterate over each row of pixels in the (vertically flipped) image array
        for line_index, unprocessed_pixel_row in enumerate(image_array):
            # Calculate the current line's Y G-code coordinate
            current_line_Y_logical = round(
                initial_logical_Y + line_index * pixel_height_gcode_units, Y_prec)

            if is_scan_left_to_right:
                # Scanning L->R: The first pixel processed is at the leftmost X G-code coordinate for this line.
                line_scan_start_X_logical = initial_logical_X
                pixel_row_for_processing = unprocessed_pixel_row  # Use the row as is
            else:  # Scanning R->L
                # The first pixel processed is the rightmost image pixel for this line.
                # Its logical X G-code coordinate (left edge for drawing) is: offset_X + (num_columns - 1) * pixel_width
                line_scan_start_X_logical = round(initial_logical_X +
                                                  (num_pixel_columns_in_image - 1) * pixel_width_gcode_units, X_prec)
                # Flip the row for R->L processing
                pixel_row_for_processing = np.flip(unprocessed_pixel_row)

            # Add a terminator sentinel value (0) to the end of the pixel row data.
            # This helps finalize the last segment of pixels in _process_scan_line_pixels.
            pixel_row_terminated = np.append(pixel_row_for_processing, 0)

            # Process the current line of pixels to generate G-code segments
            head_after_line_proc, gcode_for_this_line = self._process_scan_line_pixels(
                pixel_row_terminated,
                line_scan_start_X_logical,
                current_line_Y_logical,
                current_head_logical,
                is_scan_left_to_right,
                args, X_prec, Y_prec
            )

            # Add generated G-code for the line
            all_gcode_commands.extend(gcode_for_this_line)
            # Update head's logical G-code position
            current_head_logical = head_after_line_proc

            # Alternate scan direction for the next line
            is_scan_left_to_right = not is_scan_left_to_right

        # Join all G-code commands, filtering out any potential empty strings from moves not taken
        return '\n'.join(filter(None, all_gcode_commands))
