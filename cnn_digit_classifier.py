# cnn_digit_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import os

class DigitCNN(nn.Module):
    """
    Lightweight CNN for single/double digit recognition.
    Based on a simple LeNet-style architecture — fast and effective
    for small isolated digit images like scorecard cells.
    """
    def __init__(self, num_classes=10):
        super(DigitCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNNDigitEngine:
    """
    Wrapper around DigitCNN that handles preprocessing,
    MNIST pretraining, and fine-tuning on scorecard cells.
    """

    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DigitCNN(num_classes=10).to(self.device)

        # Transform for inference — matches MNIST preprocessing
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
        ])

        if model_path and os.path.exists(model_path):
            print(f"Loading saved model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            print("No saved model found — run pretrain_on_mnist() first")

        self.model.eval()

    # ── Preprocessing ────────────────────────────────────────────────────────

    def preprocess_cell(self, image: Image.Image) -> Image.Image:
        """
        Preprocessing specifically tuned for digit recognition.
        More aggressive than TrOCR preprocessing because CNN
        needs clean, high-contrast digit shapes.
        """
        # Grayscale
        image = image.convert("L")

        # Remove grid line bleed from edges
        img = np.array(image)
        h, w = img.shape
        border = 4
        if h > border * 3 and w > border * 3:
            img = img[border:h-border, border:w-border]
        image = Image.fromarray(img)

        # Aggressive contrast boost — helps faint pencil marks
        image = ImageOps.autocontrast(image, cutoff=5)

        # Sharpen to recover detail lost in blurry crops
        image = image.filter(ImageFilter.SHARPEN)
        image = image.filter(ImageFilter.SHARPEN)

        # Remove annotation circles/marks using morphological opening
        # This erodes thin rings while preserving thicker digit strokes
        img = np.array(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        image = Image.fromarray(img)

        return image

    def is_empty_cell(self, image: Image.Image, threshold=0.95) -> bool:
        gray = np.array(image.convert("L"))
        light_ratio = np.sum(gray > 180) / gray.size
        return light_ratio > threshold

    # ── Training ─────────────────────────────────────────────────────────────

    def pretrain_on_mnist(self, epochs=5, save_path="models/cnn_mnist.pth"):
        """
        Pretrain on MNIST dataset. This gives the model a strong
        starting point for digit recognition before fine-tuning
        on actual scorecard cells.
        Downloads MNIST automatically on first run (~11MB).
        """
        os.makedirs("models", exist_ok=True)
        os.makedirs("data/mnist", exist_ok=True)

        print("Loading MNIST dataset...")
        train_data = datasets.MNIST(
            root="data/mnist",
            train=True,
            download=True,
            transform=self.transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=64, shuffle=True
        )

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)

                if batch_idx % 200 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} "
                          f"[{batch_idx * len(data)}/{len(train_data)}] "
                          f"Loss: {loss.item():.4f}")

            acc = 100. * correct / total
            print(f"Epoch {epoch+1} complete — "
                  f"Avg Loss: {total_loss/len(train_loader):.4f}, "
                  f"Accuracy: {acc:.1f}%")
            scheduler.step()

        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        self.model.eval()

    def finetune_on_cells(self, cells_dir="data/cells",
                          epochs=20, save_path="models/cnn_finetuned.pth"):
        """
        Fine-tune on your labeled scorecard cells from the crop tool.
        Only uses cells with pure digit labels (skips labels, empty, names etc.)
        Even a small number of cells (20-30) helps significantly.
        """
        from pathlib import Path

        print(f"Loading scorecard digit cells from {cells_dir}...")

        images, labels = [], []
        for img_path in Path(cells_dir).glob("*.png"):
            true_label = img_path.stem.rsplit("_", 1)[0]

            # Only use cells where the label is a pure digit 0-9
            if true_label.isdigit() and int(true_label) <= 9:
                image = Image.open(img_path)
                image = self.preprocess_cell(image)
                tensor = self.transform(image)
                images.append(tensor)
                labels.append(int(true_label))

        if len(images) == 0:
            print("No single-digit cells found for fine-tuning.")
            print("Crop more single-digit cells labeled as '4', '3', '5' etc.")
            return

        print(f"Found {len(images)} digit cells for fine-tuning")

        dataset = torch.utils.data.TensorDataset(
            torch.stack(images),
            torch.tensor(labels)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=True
        )

        # Lower learning rate for fine-tuning to avoid overwriting MNIST weights
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

            acc = 100. * correct / len(images)
            print(f"Epoch {epoch+1}/{epochs} — "
                  f"Loss: {total_loss/len(loader):.4f}, "
                  f"Accuracy: {acc:.1f}%")

        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Fine-tuned model saved to {save_path}")
        self.model.eval()

    # ── Inference ────────────────────────────────────────────────────────────

    def read_digit(self, image_path: str) -> tuple[int, float] | tuple[None, None]:
        """
        Predict a single digit from a cell image.
        Returns (digit, confidence) or (None, None) if empty.
        Confidence is useful for deciding whether to fall back to TrOCR.
        """
        image = Image.open(image_path).convert("RGB")

        if self.is_empty_cell(image):
            return None, None

        image = self.preprocess_cell(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = probs.max(1)

        return predicted.item(), confidence.item()

    def read_score_cell(self, image_path: str,
                        confidence_threshold=0.6) -> int | None:
        """
        Read a score cell. Returns the predicted digit if confidence
        is above threshold, otherwise returns None to signal that
        TrOCR fallback should be tried.
        """
        digit, confidence = self.read_digit(image_path)

        if digit is None:
            return None  # empty cell

        if confidence < confidence_threshold:
            print(f"  [LOW CONFIDENCE] CNN predicted {digit} "
                  f"with {confidence:.2f} — consider TrOCR fallback")
            return None

        return digit


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = CNNDigitEngine()

    # Step 1: pretrain on MNIST
    engine.pretrain_on_mnist(epochs=5, save_path="models/cnn_mnist.pth")

    # Step 2: fine-tune on your scorecard cells
    engine.finetune_on_cells(
        cells_dir="data/cells",
        epochs=20,
        save_path="models/cnn_finetuned.pth"
    )

    # Step 3: test on your digit cell
    digit, conf = engine.read_digit("data/test_digit1.png")
    print(f"\nCNN result: digit={digit}, confidence={conf:.2f}")