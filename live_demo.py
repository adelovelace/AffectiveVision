import argparse
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from models.custom_cnn_rnn import CustomCNN_RNN
from models.inceptionv3_rnn import InceptionV3_RNN
from models.mobilenetv2_rnn import MobileNetV2_RNN


FER2013_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CKPLUS_CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Sadness", "Surprise"]


def parse_args():
    parser = argparse.ArgumentParser(description="AffectiveVision live demo for image, video, or webcam input.")

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--image", type=str, help="Path to a single image.")
    source_group.add_argument("--video", type=str, help="Path to a video file.")
    source_group.add_argument("--camera", action="store_true", help="Use the webcam. This is the default if no source is given.")

    parser.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/custom/fer2013_best.pth")
    parser.add_argument("--model", type=str, default="custom", choices=["custom", "mobilenet", "inception"])
    parser.add_argument("--dataset", type=str, default="fer2013", choices=["fer2013", "ckplus"])
    parser.add_argument("--class-names", type=str, default="", help="Comma-separated labels. Overrides --dataset.")
    parser.add_argument("--img-size", type=int, default=None, help="Override input size. Defaults: custom=48, mobilenet=224, inception=299.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps", "auto"])

    parser.add_argument("--no-face-crop", action="store_true", help="Run inference on the full frame/image.")
    parser.add_argument("--margin", type=float, default=0.20, help="Face crop margin as a fraction of face size.")
    parser.add_argument("--frame-step", type=int, default=5, help="For videos, process every Nth frame.")
    parser.add_argument("--smooth-window", type=int, default=7, help="Rolling window for stable webcam/video labels.")
    parser.add_argument("--min-face-size", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit for video/camera frames. 0 means no limit.")

    parser.add_argument("--gradcam", action="store_true", help="Add Grad-CAM overlay for single-image demos.")
    parser.add_argument("--tts", action="store_true", help="Speak emotion changes during webcam demos.")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV display window.")
    parser.add_argument("--output", type=str, default="", help="Output image/video path. If omitted, a sensible default is used.")
    parser.add_argument("--save-frames", action="store_true", help="For video/camera, write annotated frames to --output.")

    return parser.parse_args()


def get_device(choice):
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_class_names(args):
    if args.class_names:
        return [name.strip() for name in args.class_names.split(",") if name.strip()]
    if args.dataset == "ckplus":
        return CKPLUS_CLASSES
    return FER2013_CLASSES


def get_image_size(model_name, override):
    if override is not None:
        return override
    if model_name == "mobilenet":
        return 224
    if model_name == "inception":
        return 299
    return 48


def build_model(model_name, num_classes):
    if model_name == "custom":
        return CustomCNN_RNN(num_classes=num_classes)
    if model_name == "mobilenet":
        return MobileNetV2_RNN(num_classes=num_classes)
    if model_name == "inception":
        return InceptionV3_RNN(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    cleaned = {}
    for key, value in checkpoint.items():
        cleaned[key.replace("module.", "")] = value

    model.load_state_dict(cleaned, strict=True)
    return model


def make_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def create_face_detector():
    require_cv2()
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade face detector.")
    return detector


def require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV is required for live_demo.py. Install it with: pip install opencv-python")


def detect_largest_face(frame_bgr, detector, min_face_size):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def crop_with_margin(frame_bgr, box, margin_ratio):
    if box is None:
        return frame_bgr, None

    x, y, w, h = box
    margin = int(max(w, h) * margin_ratio)
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, frame_bgr.shape[1])
    y2 = min(y + h + margin, frame_bgr.shape[0])
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def predict_frame(model, frame_bgr, transform, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probabilities))
    return pred_idx, probabilities


def draw_probability_panel(frame, probabilities, class_names, origin=(10, 34), width=240):
    x0, y0 = origin
    row_h = 24
    max_bar_w = width - 105
    panel_h = row_h * len(class_names) + 18

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 24), (x0 + width, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        y = y0 + i * row_h
        color = (0, 180, 255) if i == int(np.argmax(probabilities)) else (100, 170, 230)
        cv2.putText(frame, name, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0 + 82, y - 12), (x0 + 82 + int(max_bar_w * prob), y + 3), color, -1)
        cv2.putText(frame, f"{prob:.2f}", (x0 + width - 45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)


def annotate_frame(frame, box, label, confidence, probabilities, class_names):
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
        text_y = max(y1 - 10, 24)
    else:
        text_y = 34

    label_text = f"{label}: {confidence:.1%}"
    cv2.putText(frame, label_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 80), 2, cv2.LINE_AA)
    draw_probability_panel(frame, probabilities, class_names, origin=(10, max(text_y + 36, 64)))
    return frame


def find_last_conv_layer(module):
    last_name = None
    last_layer = None
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d):
            last_name = name
            last_layer = child
    return last_name, last_layer


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self.layer_name, self.target_layer = find_last_conv_layer(model)
        if self.target_layer is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM.")

        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor, class_idx):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward()

        weights = self.gradients[0].mean(dim=(1, 2), keepdim=True)
        heatmap = torch.sum(weights * self.activations[0], dim=0)
        heatmap = torch.relu(heatmap)
        heatmap -= heatmap.min()

        max_value = heatmap.max()
        if max_value > 0:
            heatmap /= max_value
        return heatmap.detach().cpu().numpy()


def make_gradcam_overlay(model, face_bgr, transform, device, pred_idx):
    frame_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    gradcam = GradCAM(model)
    try:
        heatmap = gradcam(input_tensor, pred_idx)
    finally:
        gradcam.remove_hooks()

    heatmap = cv2.resize(heatmap, (face_bgr.shape[1], face_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(face_bgr, 0.55, heatmap_color, 0.45, 0)


class EmotionSpeaker:
    def __init__(self, enabled):
        self.enabled = enabled
        self.engine = None
        self.is_speaking = False
        self.last_spoken = None

        if self.enabled:
            try:
                import pyttsx3

                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 150)
            except Exception as exc:
                print(f"[!] TTS disabled because pyttsx3 could not initialize: {exc}")
                self.enabled = False

    def maybe_speak(self, label):
        if not self.enabled or self.is_speaking or label == self.last_spoken:
            return
        self.last_spoken = label
        threading.Thread(target=self._speak, args=(label,), daemon=True).start()

    def _speak(self, label):
        self.is_speaking = True
        try:
            self.engine.say(f"You look {label.lower()}")
            self.engine.runAndWait()
        finally:
            self.is_speaking = False


def prepare_runtime(args):
    device = get_device(args.device)
    class_names = get_class_names(args)
    img_size = get_image_size(args.model, args.img_size)
    transform = make_transform(img_size)

    print(f"[!] Device: {device}")
    print(f"[!] Model: {args.model}")
    print(f"[!] Checkpoint: {args.checkpoint}")
    print(f"[!] Classes: {class_names}")
    print(f"[!] Input size: {img_size}x{img_size}")

    model = build_model(args.model, num_classes=len(class_names))
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    detector = None if args.no_face_crop else create_face_detector()
    return model, transform, detector, class_names, device


def run_image_demo(args, model, transform, detector, class_names, device):
    require_cv2()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original = cv2.imread(str(image_path))
    if original is None:
        raise RuntimeError(f"OpenCV could not read image: {image_path}")

    box = None if detector is None else detect_largest_face(original, detector, args.min_face_size)
    face_crop, expanded_box = crop_with_margin(original, box, args.margin)
    if box is None:
        print("[!] No face detected; using the full image.")

    pred_idx, probabilities = predict_frame(model, face_crop, transform, device)
    label = class_names[pred_idx]
    confidence = float(probabilities[pred_idx])

    annotated = annotate_frame(original.copy(), expanded_box, label, confidence, probabilities, class_names)

    if args.gradcam:
        try:
            gradcam_overlay = make_gradcam_overlay(model, face_crop, transform, device, pred_idx)
            gradcam_overlay = cv2.resize(gradcam_overlay, (original.shape[1] // 2, original.shape[0] // 2))
            annotated[-gradcam_overlay.shape[0]:, -gradcam_overlay.shape[1]:] = gradcam_overlay
            print("[!] Grad-CAM overlay added in the bottom-right corner.")
        except Exception as exc:
            print(f"[!] Grad-CAM skipped: {exc}")

    output = Path(args.output or "outputs/demo_image_result.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)

    print("\nPrediction probabilities:")
    for name, prob in sorted(zip(class_names, probabilities), key=lambda item: item[1], reverse=True):
        print(f"  {name:>10s}: {prob:.4f}")
    print(f"\n[!] Predicted emotion: {label} ({confidence:.1%})")
    print(f"[!] Saved annotated image to: {output.resolve()}")

    if not args.no_display:
        cv2.imshow("AffectiveVision image demo", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def open_video_source(args):
    require_cv2()
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return cv2.VideoCapture(str(video_path)), str(video_path)
    return cv2.VideoCapture(0), "webcam"


def stream_should_quit(window_name):
    key = cv2.waitKey(10) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return True

    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def close_stream_windows():
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)


def run_stream_demo(args, model, transform, detector, class_names, device):
    cap, source_name = open_video_source(args)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source_name}")

    speaker = EmotionSpeaker(enabled=args.tts and source_name == "webcam")
    label_window = deque(maxlen=max(args.smooth_window, 1))
    frame_idx = 0
    processed_predictions = []
    last_probabilities = np.ones(len(class_names), dtype=np.float32) / len(class_names)
    last_box = None

    writer = None
    if args.save_frames:
        output = Path(args.output or ("outputs/demo_webcam.mp4" if source_name == "webcam" else "outputs/demo_video.mp4"))
        output.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        print(f"[!] Writing annotated video to: {output.resolve()}")

    window_name = "AffectiveVision live demo"

    print("[!] Starting stream. Click the video window, then press 'q' or Esc to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_idx >= args.max_frames:
                break

            should_process = source_name == "webcam" or frame_idx % max(args.frame_step, 1) == 0
            frame_idx += 1

            if should_process:
                box = None if detector is None else detect_largest_face(frame, detector, args.min_face_size)
                face_crop, expanded_box = crop_with_margin(frame, box, args.margin)
                last_box = expanded_box

                if box is not None or args.no_face_crop:
                    pred_idx, probabilities = predict_frame(model, face_crop, transform, device)
                    last_probabilities = probabilities
                    label_window.append(class_names[pred_idx])
                    processed_predictions.append(class_names[pred_idx])

            if label_window:
                label = Counter(label_window).most_common(1)[0][0]
                label_idx = class_names.index(label)
                confidence = float(last_probabilities[label_idx])
                speaker.maybe_speak(label)
                annotate_frame(frame, last_box, label, confidence, last_probabilities, class_names)
            else:
                cv2.putText(frame, "No face detected", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow(window_name, frame)
                if stream_should_quit(window_name):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            close_stream_windows()

    if processed_predictions:
        final_label, count = Counter(processed_predictions).most_common(1)[0]
        print(f"[!] Final majority-vote emotion: {final_label} ({count}/{len(processed_predictions)} processed frames)")
    else:
        print("[!] No faces were confidently detected in the stream.")


def main():
    args = parse_args()
    if not args.image and not args.video:
        args.camera = True

    started = time.time()
    model, transform, detector, class_names, device = prepare_runtime(args)

    if args.image:
        run_image_demo(args, model, transform, detector, class_names, device)
    else:
        run_stream_demo(args, model, transform, detector, class_names, device)

    print(f"[!] Demo finished in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
