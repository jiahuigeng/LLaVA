import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
from torchvision import transforms

torch.manual_seed(50)

import requests
from PIL import Image
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"predict the sentiment of the sentence, positive, negative, or neutral?"
target_prompt = "translate the following sentence to English:"
empty_id = 29871
image_shape = (1, 3, 336, 336)
image_token_len = 576

tp = transforms.ToPILImage()

# Define a simple 3-layer MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        logits = self.layer3(x)
        return logits

# # Initialize the model
# input_size = 784  # For example, this could be an image of size 28x28
# hidden_size = 128
# output_size = 100  # Assuming 10 classes for classification
# model = SimpleMLP(input_size, hidden_size, output_size)

def main(args):
    # Generate a random input
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           device=args.device)  # load_4bit=True,

    input_ids = tokenizer_image_token(target_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
        model.device)
    input_embeds = model.get_model().embed_tokens(input_ids).to(model.device)
    empty_embed = model.get_model().embed_tokens(torch.tensor([[empty_id]]).to(model.device))
    empty_embeds = empty_embed.repeat(1, image_token_len - input_ids.shape[1], 1)
    padded_input_embeds = torch.cat((input_embeds, empty_embeds), dim=1).to(model.device)

    image_tensor = torch.randn(image_shape).to(device).requires_grad_(True)

    optimizer_name = 'Adam'

    if optimizer_name == 'LBFGS':
        optimizer = optim.LBFGS([image_tensor], lr=1, max_iter=20, line_search_fn='strong_wolfe')
        # optimizer = optim.LBFGS([image_tensor])
        def closure():
            optimizer.zero_grad()
            image_embeds = model.encode_images(image_tensor)
            diff = image_embeds[:, -1, :] - input_embeds[:, -1, :]
            loss = (diff**2).mean()
            loss.backward(retain_graph=True)
            return loss

        for step in range(20):
            optimizer.step(closure)
            with torch.no_grad():
                image_embeds = model.encode_images(image_tensor).requires_grad_(True)
                diff = image_embeds[:, -1, :] - input_embeds[:, -1, :]
                loss = (diff**2).mean()
                print(f'Step {step}, Loss: {loss.item()}')

            if loss < 1e-4:
                break

    elif optimizer_name == 'Adam': #(1, 576, 1024)

        optimizer = optim.Adam([image_tensor], lr=0.1)
        # image_embeds = torch.randn(model.encode_images(image_tensor).shape).to(device).requires_grad_(True)
        # optimizer = optim.Adam([image_embeds], lr=0.1)
        model.train()
        for param in model.parameters():
            param.requires_grad = False
        for step in range(2):  # May need more iterations for Adam
            optimizer.zero_grad()
            image_embeds = model.encode_images(image_tensor)
            # diff = image_embeds[:, -9:, :] - padded_input_embeds[:, -9:, :]
            diff = image_embeds - padded_input_embeds
            loss = (diff ** 2).mean()
            loss.backward(retain_graph=True)
            # loss.backward()
            optimizer.step()

            if step % 1 == 0:  # Print loss every 10 steps
                print(f'Step {step}, Loss: {loss.item()}')

            if loss < 1e-4:  # A threshold for convergence
                break

    image = tp(image_tensor[0].detach().cpu())
    image.save('prompt.png')



if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        image_example = "https://llava-vl.github.io/static/images/view.jpg"
        parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
        parser.add_argument("--model-base", type=str, default=None)
        parser.add_argument("--image-file", type=str, default="")
        parser.add_argument("--device", type=str, default="cuda")
        # parser.add_argument("--conv-mode", type=str, default=None)
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--max-new-tokens", type=int, default=512)
        # parser.add_argument("--load-8bit", action="store_true")
        # parser.add_argument("--load-4bit", action="store_true")
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()
        main(args)
