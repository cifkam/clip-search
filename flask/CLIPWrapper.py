import torch
import clip
import pprint


class CLIPWrapper:
    def __init__(self, model_name="ViT-B/32", prefer_cuda=False) -> None:
        self.device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
        self.log(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.log(f"Model {model_name} loaded.")

    def Create(*, prefer_cuda=False, **kwargs):
        # If prefer_cuda == True, try to load model on GPU
        # If false or loading failed, load model on CPU
        try:
            return CLIPWrapper(prefer_cuda=prefer_cuda, **kwargs)
        except Exception as e:
            if prefer_cuda:
                print(e)
                return CLIPWrapper(prefer_cuda=False, **kwargs)
            else:
                raise e

    @staticmethod
    def available_models():
        return clip.available_models()

    def log(self, *args):
        print("CLIPWrapper:", *args)

    def img2vec(self, img):
        with torch.no_grad():
            img = self.preprocess(img).unsqueeze(0).to(self.device)
            return self.model.encode_image(img)
        # image_features = self.model.encode_image(img)
        # return image_features / image_features.norm(dim=-1, keepdim=True)

    def text2vec(self, text):
        with torch.no_grad():
            text = clip.tokenize(text).to(self.device)
            return self.model.encode_text(text)
        # text_features = self.model.encode_text(text)
        # return text_features / text_features.norm(dim=-1, keepdim=True)

    def classify(self, img, labels):
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        text = clip.tokenize(labels).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        result = dict(
            zip(
                labels,
                map(lambda x: str(x / 100), (100**2 * probs).astype(int)[0].tolist()),
            )
        )
        self.log("\n" + pprint.pformat(result))
        return result
