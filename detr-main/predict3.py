import cv2
import torch
from PIL import Image
import torchvision.transforms as T
from hubconf import detr_resnet50
from util.misc import nested_tensor_from_tensor_list

torch.set_grad_enabled(False)

# COCO classes
CLASSES = [
    'Primary tearing',
    'Primary tearing',
    'Secondary tearing'
]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.00001

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def predict(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    anImg = transform(im)
    data = nested_tensor_from_tensor_list([anImg])

    # propagate through the model
    outputs = model(data)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]

    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def draw_boxes(image, prob, boxes):
    img = image.copy()
    colors = [(0, 0, 255), (255, 0, 0)]  # BGR format
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), c, 2)
        cv2.putText(img, text, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return img

def process_video(input_path, output_path, model, transform):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scores, boxes = predict(im, model, transform)
        result_frame = draw_boxes(frame, scores, boxes)
        out.write(result_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = detr_resnet50(False, 3)
    state_dict = torch.load("outputs4/checkpoint.pth", map_location='cpu')
    model.load_state_dict(state_dict["model"])
    model.eval()

    input_video_path = 'img/test2.MP4'
    output_video_path = 'img_out'

    process_video(input_video_path, output_video_path, model, transform)
