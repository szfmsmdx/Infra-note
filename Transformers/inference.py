import torch
from Config import T5Config
from Transformer import Model
from Tokenizer.BPE import BPE_Tokenizer

def run_inference(checkpoint_path, text_prompt):
    tokenizer = BPE_Tokenizer()
    tokenizer.load("./Tokenizer/tokenizer.pt")
    sentinels = [f"<extra_id_{i}>" for i in range(100)]
    tokenizer.add_special_token(sentinels)
    tokenizer.add_special_token(["<pad>", "<eos>", "<unk>"])
    config = T5Config(tokenizer)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(config)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(text_prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], device=device)

    print(f"\nInput Text: {text_prompt}")
    print("Generating...")
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_tensor, 
            max_new_token=20
        )

    result_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
    print(f"Model Output: {result_text}")

if __name__ == "__main__":
    latest_checkpoint = "./checkpoints/t5_epoch_8.pt"
    
    test_text_1 = "北京是中国的<extra_id_0>。"
    run_inference(latest_checkpoint, test_text_1)
    
    test_text_2 = "程序员最喜欢的语言是<extra_id_0>。"
    run_inference(latest_checkpoint, test_text_2)