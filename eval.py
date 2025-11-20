from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as T

from utils import model

project_root = Path(__file__).resolve().parent


def read_img(imgPath):  # read image & data pre-process
    data = torch.randn(1, 3, 112, 112)
    transform = T.Compose(
        [
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img = Image.open(imgPath).convert("RGB")
    data[0, :, :, :] = transform(img)
    return data


def network(eval_model, device):
    net = model.IR50([112, 112], use_type="Qua").to(device)
    net_dict = net.state_dict()
    data_dict = {
        key.replace("module.", ""): value
        for key, value in torch.load(
            eval_model, map_location=device, weights_only=True
        ).items()
    }
    net_dict.update(data_dict)
    net.load_state_dict(net_dict)
    net.eval()
    return net


if __name__ == "__main__":
    img_dir = project_root / "assets/demo_imgs"
    device = "cpu"  # 'cpu' or 'cuda:x'
    eval_model = str(
        project_root / "checkpoints/quality_model/SDD-PIQA_quality_model_best.pth"
    )  # checkpoint
    net = network(eval_model, device)

    if img_dir.exists():
        extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        images = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in extensions]
        )

        for imgpath in images:
            input_data = read_img(imgpath).to(device)
            pred_score = net(input_data).data.cpu().numpy().squeeze()
            print(f"Image: {imgpath.name}, Quality score = {pred_score}")
    else:
        print(f"Directory not found: {img_dir}")
