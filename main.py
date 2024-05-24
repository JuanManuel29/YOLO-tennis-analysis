from utils.Video_Utils import read_video, save_video
from trackers.Player_Tracker import PlayerTracker
from trackers.Ball_Tracker import BallTracker
from court_line_detection.court_line_detector import CourtLineDetector

def main():
    # Read the input video
    input_video_path = "input/rg_point.mp4"
    video_frames = read_video(input_video_path)

    # Player and Ball detection
    player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
    ball_tracker = BallTracker(model_path='models/Best_BallTracker_Model.pt')

    player_detections = player_tracker.detect_frames(frames = video_frames,
                                                     read_from_stub = False,
                                                     stub_path = "tracker_stubs/player_detections.pkl")
    
    ball_detections = ball_tracker.detect_frames(frames = video_frames,
                                                   read_from_stub = False,
                                                   stub_path = "tracker_stubs/ball_detections.pkl")

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Draw court keypoints
    court_kps_model_path = "models/KeyPointsModelResNet101.pt"
    court_kps_detector = CourtLineDetector(court_kps_model_path)
    court_kps = court_kps_detector.predict(video_frames[0])


    # Draw player bboxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    # Draw ball bboxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Draw court keypoints
    output_video_frames = court_kps_detector.draw_keypoints_on_video(output_video_frames, court_kps)

    # Save the output video
    output_video_path = "output/rg_point_output.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()