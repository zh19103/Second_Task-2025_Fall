import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# 使用你在 tests/adapters.py 中的实现
from tests.adapters import run_transformer_lm, run_magnitude_pruning

# --- 配置 ---
CONFIG = {
    "vocab_size": 20,      # 0..19，其中 18=SEP, 19=EOS
    "context_length": 22,  # 输入10 + SEP1 + 输出10 + EOS1
    "d_model": 64,
    "num_layers": 2,
    "num_heads": 2,
    "d_ff": 128,
    "rope_theta": 10000.0,
    "batch_size": 64,
    "lr": 1e-3,
    "steps": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# --- 数据生成器 ---
def get_batch(batch_size, vocab_size, seq_len, device):
    """
    生成数据：Input: [乱序数字, SEP] -> Target: [排序数字, EOS]
    """
    L = (seq_len - 2) // 2
    assert 2*L + 2 == seq_len
    x_raw = torch.randint(0, vocab_size - 2, (batch_size, L), device=device)
    y_raw = torch.sort(x_raw, dim=1).values
    SEP = vocab_size - 2
    EOS = vocab_size - 1
    batch_seq = torch.cat(
        [x_raw, torch.full((batch_size, 1), SEP, device=device),
         y_raw, torch.full((batch_size, 1), EOS, device=device)],
        dim=1,
    )  # 形状 [B, 2L+2]
    return batch_seq

# --- 每层参数模块 ---
class LayerParams(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Attention
        self.attn_q_proj_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.attn_k_proj_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.attn_v_proj_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.attn_output_proj_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        # FFN（与 tests/adapters.run_swiglu 对齐）
        self.ffn_w1_weight = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        self.ffn_w2_weight = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
        self.ffn_w3_weight = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        # Norm
        self.ln1_weight = nn.Parameter(torch.ones(d_model))
        self.ln2_weight = nn.Parameter(torch.ones(d_model))

# --- 模型包装器 ---
class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config["d_model"]
        d_ff = config["d_ff"]
        num_layers = config["num_layers"]
        vocab_size = config["vocab_size"]

        # token embeddings 与 lm_head 权重共享（weight tying）
        self.token_embeddings = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.ln_final_weight = nn.Parameter(torch.ones(d_model))
        self.layers = nn.ModuleList([LayerParams(d_model, d_ff) for _ in range(num_layers)])

    def _build_weights_mapping(self):
        # 构造与 adapters.run_transformer_lm 兼容的权重字典（带 "." 的键名）
        weights = {
            "token_embeddings.weight": self.token_embeddings,
            "ln_final.weight": self.ln_final_weight,
            "lm_head.weight": self.token_embeddings,  # tying
        }
        for idx, layer in enumerate(self.layers):
            pfx = f"layers.{idx}."
            weights[pfx + "attn.q_proj.weight"] = layer.attn_q_proj_weight
            weights[pfx + "attn.k_proj.weight"] = layer.attn_k_proj_weight
            weights[pfx + "attn.v_proj.weight"] = layer.attn_v_proj_weight
            weights[pfx + "attn.output_proj.weight"] = layer.attn_output_proj_weight
            weights[pfx + "ffn.w1.weight"] = layer.ffn_w1_weight
            weights[pfx + "ffn.w2.weight"] = layer.ffn_w2_weight
            weights[pfx + "ffn.w3.weight"] = layer.ffn_w3_weight
            weights[pfx + "ln1.weight"] = layer.ln1_weight
            weights[pfx + "ln2.weight"] = layer.ln2_weight
        return weights

    def forward(self, idx):
        weights = self._build_weights_mapping()
        return run_transformer_lm(
            vocab_size=self.config["vocab_size"],
            context_length=self.config["context_length"],
            d_model=self.config["d_model"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
            d_ff=self.config["d_ff"],
            rope_theta=self.config["rope_theta"],
            weights=weights,
            in_indices=idx,
        )

# --- 训练（只在 SEP 之后监督） ---
def train():
    print(f"🚀 开始在 {CONFIG['device']} 上训练 Sort-Transformer...")
    set_seed(CONFIG["seed"])
    device = CONFIG["device"]
    model = TinyTransformer(CONFIG).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

    L = (CONFIG["context_length"] - 2) // 2
    losses = []
    pbar = tqdm(range(CONFIG["steps"]))
    model.train()
    for _ in pbar:
        batch = get_batch(CONFIG["batch_size"], CONFIG["vocab_size"], CONFIG["context_length"], device)
        # teacher forcing：输入去掉最后一位；目标是右移一位
        inputs = batch[:, :-1]   # [B, 2L+1]，最后一位是 EOS 之前的 token
        targets = batch[:, 1:]   # [B, 2L+1]，第0位是原第1位

        # 只监督 SEP 之后：targets 的起点是索引 L（对应原序列中的 SEP 后第一位 y1）
        start = L
        end = 2 * L  # 包含 yL，下一位是 EOS（targets 中的 end 位置仍包含）
        # 对应长度为 L+1（y1..yL, EOS）
        logits = model(inputs)                             # [B, 2L+1, V]
        logits_slice = logits[:, start:end+1+1, :]         # +1 包含 yL，+1 再包含 EOS => total L+1
        targets_slice = targets[:, start:end+1+1]          # 同样长度 L+1

        loss = criterion(
            logits_slice.reshape(-1, CONFIG["vocab_size"]),
            targets_slice.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pbar.set_description(f"Loss: {loss.item():.4f}")

    print("✅ 训练完成！")
    return model, losses

# --- 评估（贪婪生成，禁止生成 SEP） ---
def evaluate(model):
    print("\n🧐 验证模型能力...")
    model.eval()
    device = CONFIG["device"]
    SEP = CONFIG["vocab_size"] - 2
    EOS = CONFIG["vocab_size"] - 1

    test_seq = [15, 3, 9, 1, 8]
    L = len(test_seq)
    input_ids = torch.tensor([test_seq + [SEP]], device=device)

    print(f"输入序列: {test_seq}")
    print("模型生成: ", end="")
    generated = []
    with torch.no_grad():
        for _ in range(L + 1):  # 生成排序序列 + EOS
            logits = model(input_ids)  # [1, T, V]
            # 禁止生成 SEP（避免无意义回到分隔符）
            logits[:, -1, SEP] = float("-inf")
            next_token = int(logits[0, -1].argmax().item())
            print(f"{next_token} ", end="")
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == EOS:
                break
    print("\n")

    expected = sorted(test_seq)
    ok = (generated[:len(expected)] == expected) and (generated[len(expected):len(expected)+1] == [EOS])
    if ok:
        print("✨ 完美排序！Success!")
    else:
        print("❌ 失败，还需要继续训练或调参。")
    return generated

# --- 剪枝实验 ---
def prune_and_test(model):
    print("\n✂️ 正在进行模型大剪枝 (50% Sparsity)...")
    try:
        run_magnitude_pruning(model, sparsity_level=0.5)
        print("剪枝完成。再次测试模型...")
        evaluate(model)
        print("💡 观察：如果模型在剪枝后依然能部分工作，说明它学习到了鲁棒的回路。")
    except NotImplementedError:
        print("你还没有实现 run_magnitude_pruning，跳过此步骤。")

if __name__ == "__main__":
    trained_model, loss_history = train()

    # 绘制 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss (Sorting Task)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("sorting_loss.png")
    print("📊 Loss 曲线已保存为 sorting_loss.png")

    # 验证
    evaluate(trained_model)

    # 剪枝实验
    prune_and_test(trained_model)
