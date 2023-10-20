# Import required libraries
import cv2
import mediapipe as mp
import json


def get_position():
    """
    Get the pose landmarks from a single frame of the webcam or video.

    :return: A list of detected pose landmarks as (x, y, z) coordinates, or an error message if detection fails.
    """

    # Read frame from webcam or video
    ret, frame = cap.read()
    # 'ret' is a boolean variable indicating if the frame was successfully read.
    # 'frame' is a NumPy array representing the image.

    if not ret:  # Return if the frame was not successfully read
        return "Frame was not successfully read!"

    # Convert image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # The Mediapipe Pose model requires RGB images, but OpenCV captures images in BGR format.
    # This line converts the image from BGR to RGB.

    # Detect pose landmarks
    results = pose.process(frame_rgb)
    # Use the Mediapipe Pose model to detect the pose landmarks in the image.

    if isTestMode:
        # Display the frame with pose landmarks in a window
        show_window(frame, results)

    if results.pose_landmarks:  # Check if pose landmarks were successfully detected
        # Calculating global positions
        global_positions = dict()
        global_positions['hip'] = calculate_average(indexes=[23, 24], results=results)
        global_positions['neck'] = calculate_average(indexes=[9, 10, 11, 12], results=results,
                                                     ratio=[0.5 * 4 / 3, 0.5 * 4 / 3, 1 * 4 / 3, 1 * 4 / 3])
        global_positions['head'] = calculate_average(indexes=[0], results=results)
        global_positions['upperarm_l'] = calculate_average(indexes=[11], results=results)
        global_positions['lowerarm_l'] = calculate_average(indexes=[13], results=results)
        global_positions['hand_l'] = calculate_average(indexes=[10], results=results)
        global_positions['upperarm_r'] = calculate_average(indexes=[12], results=results)
        global_positions['lowerarm_r'] = calculate_average(indexes=[14], results=results)
        global_positions['hand_r'] = calculate_average(indexes=[16], results=results)
        global_positions['upperleg_twist_l'] = calculate_average(indexes=[23], results=results)
        global_positions['lowerleg_twist_l'] = calculate_average(indexes=[25], results=results)
        global_positions['foot_l'] = calculate_average(indexes=[27], results=results)
        global_positions['ball_l'] = calculate_average(indexes=[31], results=results)
        global_positions['upperleg_twist_r'] = calculate_average(indexes=[24], results=results)
        global_positions['lowerleg_twist_r'] = calculate_average(indexes=[26], results=results)
        global_positions['foot_r'] = calculate_average(indexes=[28], results=results)
        global_positions['ball_r'] = calculate_average(indexes=[30], results=results)

        # Testing
        # print("hip :", global_positions['hip'])
        # print("neck :", global_positions['neck'])
        # print("neck - hip (x):",
        #       global_positions['neck']['positions']['x'] - global_positions['hip']['positions']['x'])

        # Converting global positions to relative positions
        relative_positions = dict()
        relative_positions['hip'] = global_positions['hip']
        relative_positions['neck'] = position_minus(global_positions['hip'], global_positions['neck'])
        relative_positions['head'] = position_minus(global_positions['neck'], global_positions['head'])
        relative_positions['upperarm_l'] = position_minus(global_positions['neck'], global_positions['upperarm_l'])
        relative_positions['lowerarm_l'] = position_minus(global_positions['upperarm_l'],
                                                          global_positions['lowerarm_l'])
        relative_positions['hand_l'] = position_minus(global_positions['lowerarm_l'], global_positions['hand_l'])
        relative_positions['upperarm_r'] = position_minus(global_positions['neck'], global_positions['upperarm_r'])
        relative_positions['lowerarm_r'] = position_minus(global_positions['upperarm_r'],
                                                          global_positions['lowerarm_r'])
        relative_positions['hand_r'] = position_minus(global_positions['lowerarm_r'], global_positions['hand_r'])
        relative_positions['upperleg_twist_l'] = position_minus(global_positions['hip'],
                                                                global_positions['upperleg_twist_l'])
        relative_positions['lowerleg_twist_l'] = position_minus(global_positions['upperleg_twist_l'],
                                                                global_positions['lowerleg_twist_l'])
        relative_positions['foot_l'] = position_minus(global_positions['lowerleg_twist_l'], global_positions['foot_l'])
        relative_positions['ball_l'] = position_minus(global_positions['foot_l'], global_positions['ball_l'])
        relative_positions['upperleg_twist_r'] = position_minus(global_positions['hip'],
                                                                global_positions['upperleg_twist_r'])
        relative_positions['lowerleg_twist_r'] = position_minus(global_positions['upperleg_twist_r'],
                                                                global_positions['lowerleg_twist_r'])
        relative_positions['foot_r'] = position_minus(global_positions['lowerleg_twist_r'], global_positions['foot_r'])
        relative_positions['ball_r'] = position_minus(global_positions['foot_r'], global_positions['ball_r'])

        return relative_positions

    else:
        return "Pose landmarks were not successfully detected!"


def position_minus(parent_node, child_node):
    return {
        'positions': {
            'x': child_node['positions']['x'] - parent_node['positions']['x'],
            'y': child_node['positions']['y'] - parent_node['positions']['y'],
            'z': child_node['positions']['z'] - parent_node['positions']['z']
        },
        'visibility': child_node['visibility']
    }


def calculate_average(indexes, results, ratio=None):
    if ratio is None:
        return {
            'positions': {
                'x': sum(results.pose_landmarks.landmark[i].x for i in indexes) / len(indexes),
                'y': sum(results.pose_landmarks.landmark[i].y for i in indexes) / len(indexes),
                'z': sum(results.pose_landmarks.landmark[i].z for i in indexes) / len(indexes)
            },
            'visibility': sum(results.pose_landmarks.landmark[i].visibility for i in indexes) / len(indexes)
        }
    else:
        return {
            'positions': {
                'x': sum(results.pose_landmarks.landmark[i].x * ratio[indexes.index(i)] for i in indexes) / len(
                    indexes),
                'y': sum(results.pose_landmarks.landmark[i].y * ratio[indexes.index(i)] for i in indexes) / len(
                    indexes),
                'z': sum(results.pose_landmarks.landmark[i].z * ratio[indexes.index(i)] for i in indexes) / len(indexes)
            },
            'visibility': sum(
                results.pose_landmarks.landmark[i].visibility * ratio[indexes.index(i)] for i in indexes) / len(indexes)
        }


def show_window(frame, results):
    """
    Draw the pose landmarks and connections on the input frame using the Mediapipe drawing utilities.

    :param frame: The frame (image) to draw on.
    :param results: The results containing the pose landmarks.
    :return: None
    """
    # Use the Mediapipe drawing utilities to draw the pose landmarks on the image
    mp_draw = mp.solutions.drawing_utils
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # Draw the landmarks and connections on the frame using the Mediapipe drawing utilities.

    # Show image
    cv2.imshow("Pose Detection", frame)
    # Display the frame in a window with the title "Pose Detection".

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        exit()


# Load Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Set up webcam or video
cap = cv2.VideoCapture(0)
# 0 indicates the default webcam. Change it to a video file path if you want to process a video.

isTestMode = True

if isTestMode:
    while True:
        positions = get_position()
        print(json.dumps(positions, indent=4))
        print('-' * 50)
else:
    frame_count = 5
    positions = dict()
    for i in range(frame_count):
        positions[f"frame{i}"] = get_position()

    print(json.dumps(positions, indent=4))
    #
    # print('-' * 50)
    # print(json.dumps(positions['frame0'], indent=4))
    # print(json.dumps(positions['frame0']['right shoulder']['position']['x'], indent=4))
    #

# Release video capture and destroy windows
cap.release()  # Release the video capture device (webcam or video file).
cv2.destroyAllWindows()  # Destroy all windows created by OpenCV.
