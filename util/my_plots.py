
import os
import re
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

def _sanitize_name(name: str) -> str:
    import re
    return re.sub(r"[^\w\.-]+", "_", name).strip("_")


def _load_pil_image(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image path not found: {img}")
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    raise TypeError(f"Unsupported image type for visualization: {type(img)}")


def _reduce_attention(attention, head_idx=None):
    if isinstance(attention, torch.Tensor):
        a = attention
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        return a.mean(dim=0) if head_idx is None else a[head_idx]
    a = np.asarray(attention)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    return a.mean(axis=0) if head_idx is None else a[head_idx]


def _restore_heatmap(v, H_eff, W_eff, t=1, reduce_mode="mean"):
    n = v.shape[0]
    hw = H_eff * W_eff
    if n == hw and t == 1:
        return v.reshape(H_eff, W_eff)
    if n != t * hw:
        raise ValueError(
            f"len(v)={n} does not match t*H_eff*W_eff={t*hw}"
        )
    v3 = v.reshape(t, H_eff, W_eff)
    if reduce_mode == "mean":
        return v3.mean(axis=0)
    elif reduce_mode == "max":
        return v3.max(axis=0)
    return v3.sum(axis=0)


def _visualize_attention_on_images(
    images,
    attn_i,
    label_pos_list,
    spans,
    patch_grids,
    save_dir,
    layer_idx,
    direction="label_to_image",
):
    if isinstance(attn_i, torch.Tensor):
        attn_i = attn_i.detach().cpu().numpy()
    os.makedirs(save_dir, exist_ok=True)
    count = min(len(images), len(label_pos_list), len(spans))
    for i in range(count):
        img = images[i]
        b, e = spans[i]
        label_pos = label_pos_list[i]
        if direction == "label_to_image":
            vec = attn_i[label_pos, b:e]
        elif direction == "image_to_label":
            vec = attn_i[b:e, label_pos]
        else:
            raise ValueError("direction must be 'label_to_image' or 'image_to_label'")
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size == 0:
            continue
        vmax, vmin = float(v.max()), float(v.min())
        if np.isclose(vmax, vmin):
            v = np.zeros_like(v, dtype=np.float32)
        else:
            v = (v - vmin) / (vmax - vmin + 1e-8)
        if patch_grids is not None and i < len(patch_grids) and patch_grids[i] is not None:
            H, W = patch_grids[i]
            n = e - b
            if n != H * W and n % (H * W) != 0:
                raise ValueError(
                    f"span_len={n} does not match H*W={H*W} for image {i}."
                )
            heat = _restore_heatmap(v, H, W, reduce_mode="mean")
        else:
            side = int(np.sqrt(v.size))
            if side * side != v.size:
                raise ValueError(
                    "Unable to restore image heatmap: missing patch grid info and non-square token count."
                )
            heat = v.reshape(side, side)
        heat_rgba = plt.get_cmap("jet")(heat)[:, :, :3]
        heat_rgb = (heat_rgba * 255).astype(np.uint8)
        heat_img_color = Image.fromarray(heat_rgb).resize(img.size, resample=Image.BILINEAR)
        overlay = Image.blend(img.convert("RGB"), heat_img_color, alpha=0.5)
        concat = Image.new("RGB", (img.width * 2, img.height))
        concat.paste(img.convert("RGB"), (0, 0))
        concat.paste(overlay, (img.width, 0))
        out_path = os.path.join(save_dir, f"layer{layer_idx}_img{i}_{direction}.png")
        concat.save(out_path)
        print(f"[Saved] {out_path}")


def plot_attention_on_images(
    model,
    processor,
    item,
    atts,
    spans,
    patch_grids,
    demos_label_token_idx,
    save_dir,
    head_idx=None,
    layer_idx=None,
    direction="label_to_image",
):
    os.makedirs(save_dir, exist_ok=True)
    images = [_load_pil_image(img) for img in item.get("image", [])]
    if not images:
        raise ValueError("No images found in item for attention visualization.")
    layer_indices = [layer_idx] if layer_idx is not None else list(range(len(atts)))
    for layer in layer_indices:
        attention = atts[layer]
        attn_i = _reduce_attention(attention, head_idx=head_idx)
        sample_save_dir = os.path.join(save_dir, f"layer_{layer}")
        _visualize_attention_on_images(
            images=images,
            attn_i=attn_i,
            label_pos_list=demos_label_token_idx,
            spans=spans,
            patch_grids=patch_grids,
            save_dir=sample_save_dir,
            layer_idx=layer,
            direction=direction,
        )


def plot_img_text_attn(all_img_curves, all_txt_curves, title=None, save_path=None):
    """
    all_img_curves: list[np.ndarray(L,)]  每个样本一条 curve
    all_txt_curves: list[np.ndarray(L,)]
    """
    plt.rcParams.update({
        "font.size": 16,        # 坐标轴、刻度、legend 基础字体
        "axes.titlesize": 18,   # 标题
        "axes.labelsize": 16,   # x/y label
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
    })

    img_arr = np.stack(all_img_curves, axis=0)  # (N, L)
    txt_arr = np.stack(all_txt_curves, axis=0)  # (N, L)

    L = img_arr.shape[1]
    layers = np.arange(L)

    img_mean = img_arr.mean(axis=0)        # (L,)
    img_std  = img_arr.std(axis=0)

    txt_mean = txt_arr.mean(axis=0)
    txt_std  = txt_arr.std(axis=0)

    plt.figure(figsize=(5, 4))

    # text curve 画成蓝色圆点
    plt.errorbar(
        layers, txt_mean, yerr=txt_std,
        fmt="o-", markersize=5, capsize=3, linewidth=1.5, label="Text"
    )
    # image curve 画成红色方块
    plt.errorbar(
        layers, img_mean, yerr=img_std,
        fmt="s-", markersize=5, capsize=3, linewidth=1.5, label="Image"
    )

    # 整体平均的虚线
    plt.axhline(txt_mean.mean(), linestyle="--", alpha=0.4)
    plt.axhline(img_mean.mean(), linestyle="--", alpha=0.4)

    plt.xlabel("# Layer Index")
    plt.ylabel("Attention Value")
    if title is not None:
        plt.title(title)
    plt.xlim(-0.5, L - 0.5)
    plt.ylim(0, 1)          # 看你自己数据范围调
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=800)
    else:
        plt.show()


def _ensure_size(node, h, w):
    """把 node['sum']/['cnt'] 扩展到至少 (h,w)。"""
    H0, W0 = node["sum"].shape
    if h <= H0 and w <= W0:
        return
    Hnew, Wnew = max(h, H0), max(w, W0)
    # 扩展 sum（float）和 cnt（uint16）
    sum_new = np.full((Hnew, Wnew), 0.0, dtype=np.float64)
    cnt_new = np.zeros((Hnew, Wnew), dtype=np.uint32)
    sum_new[:H0, :W0] = node["sum"]
    cnt_new[:H0, :W0] = node["cnt"]
    node["sum"], node["cnt"] = sum_new, cnt_new

def _resize_curve_1d(curve: np.ndarray, out_len: int) -> np.ndarray:
    """把 1D 曲线线性插值到指定长度 out_len。"""
    in_len = curve.shape[0]
    if in_len == out_len:
        return curve

    x_old = np.linspace(0.0, 1.0, in_len, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, out_len, dtype=np.float64)
    return np.interp(x_new, x_old, curve)

def _acc_heat(acc, di, H_2d):
    H = np.asarray(H_2d, dtype=np.float64)
    assert H.ndim == 2
    h, w = H.shape

    if acc[di]["sum"] is None:
        acc[di]["sum"] = np.zeros((h, w), dtype=np.float64)
        acc[di]["cnt"] = np.zeros((h, w), dtype=np.uint32)
    else:
        _ensure_size(acc[di], h, w)
    acc[di]["sum"][:h, :w] += H
    acc[di]["cnt"][:h, :w] += 1


def _acc_curve(acc, di, curve_1d):
    c = np.asarray(curve_1d, dtype=np.float64).reshape(-1)
    L = c.size

    if acc[di]["sum"] is None:
        # 第一次见到该 demo，记录下长度
        acc[di]["sum"] = c.copy()
        acc[di]["L"]   = L
        acc[di]["cnt"] = 1
    else:
        L0 = acc[di]["L"]
        if L != L0:
            # ⬅ 不再跳过，而是插值到 L0 再累加
            c = _resize_curve_1d(c, L0)
        acc[di]["sum"] += c
        acc[di]["cnt"] += 1



def _to_avg_list(acc_dict):
    """转为稠密 list：索引从 0..max_di，缺失的用 None。"""
    if not acc_dict:
        return []
    max_di = max(acc_dict.keys())
    out = []
    for di in range(max_di + 1):
        node = acc_dict.get(di)
        if node and node["cnt"] > 0:
            out.append((node["sum"] / node["cnt"]).astype(np.float32))
        else:
            out.append(None)
    return out


def plot_avg_demo_heatmaps(avg_img_H_list, avg_txt_H_list, model_name, save_dir, dpi=160):
    os.makedirs(save_dir, exist_ok=True)
    safe = _sanitize_name(model_name)
    for i, H in enumerate(avg_img_H_list):
        if H is None: continue
        plt.figure(figsize=(8,5), dpi=dpi)
        plt.imshow(H, aspect="auto", origin="lower")
        plt.colorbar(label="Attention")
        plt.xlabel("Image token (normalized)")
        plt.ylabel("Layer")
        plt.title(f"Label → Demo{i+1} Image Tokens (Average) — {model_name}")
        plt.tight_layout()
        out = os.path.join(save_dir, f"{safe}_AVERAGE_demo{i+1}_image_heatmap.png")
        plt.savefig(out); plt.close()
        print(f"[Saved] {out}")

    for i, H in enumerate(avg_txt_H_list):
        if H is None: continue
        plt.figure(figsize=(8,5), dpi=dpi)
        plt.imshow(H, aspect="auto", origin="lower")
        plt.colorbar(label="Attention")
        plt.xlabel("Text token (normalized)")
        plt.ylabel("Layer")
        plt.title(f"Label → Demo{i+1} Text Tokens (Average) — {model_name}")
        plt.tight_layout()
        out = os.path.join(save_dir, f"{safe}_AVERAGE_demo{i+1}_text_heatmap.png")
        plt.savefig(out); plt.close()
        print(f"[Saved] {out}")

def _to_avg_list_heat(acc_dict): # _to_avg_list_heat(acc_dict) —— 作用于「累加器 → 列表」
    out = []
    if not acc_dict:
        return out
    max_di = max(acc_dict.keys())
    for di in range(max_di + 1):
        node = acc_dict.get(di)
        if not node or node["sum"] is None:
            out.append(None)
            continue
        sumH, cntH = node["sum"], node["cnt"]
        avg = np.divide(sumH, np.maximum(cntH, 1), where=(cntH > 0))
        avg[cntH == 0] = np.nan  # 纯可视化时可以留 NaN
        out.append(avg.astype(np.float32))
    return out



def plot_avg_demo_curves(avg_img_curves, avg_txt_curves, model_name, save_dir, dpi=160):
    os.makedirs(save_dir, exist_ok=True)
    safe = _sanitize_name(model_name)

    # 统一长度 L
    L = 0
    for arr in avg_img_curves + avg_txt_curves:
        if arr is not None:
            L = max(L, len(arr))
    x = np.arange(L)

    # 为 demo ID 分配固定颜色（colormap）
    k = max(len(avg_img_curves), len(avg_txt_curves))
    # cmap = cm.get_cmap("tab10", k)  # 例如 tab10，前 10 个颜色 cmap(i) for i in range(k)
    colors = ['red','green','blue','purple']

    plt.figure(figsize=(9, 5), dpi=dpi)
    for i in range(k):
        color = colors[i]
        if i < len(avg_img_curves) and avg_img_curves[i] is not None:
            plt.plot(
                x,
                avg_img_curves[i],
                color=color,
                linewidth=2,
                linestyle='-',
                label=f"demo{i+1} image (sum)"
            )
        if i < len(avg_txt_curves) and avg_txt_curves[i] is not None:
            plt.plot(
                x,
                avg_txt_curves[i],
                color=color,
                linewidth=2,
                linestyle='--',
                label=f"demo{i+1} text (sum)"
            )

    plt.xlabel("Layer")
    plt.ylabel("Attention")
    plt.title(f"Label → Demos (Averaged Curves) — {model_name}")
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()

    out = os.path.join(save_dir, f"{safe}_AVERAGE_demo_curves.png")
    plt.savefig(out)
    plt.close()
    print(f"[Saved] {out}")


# _avg_heat_from_list(heat_list) —— 作用于「列表 → 单张平均图」
def _avg_heat_from_list(heat_list):
    """将若干 (L_i, W_i) 的热力图按左上角对齐，用 NaN padding 后逐点平均。
       返回 shape=(L_max, W_max) 的平均图；若全 None 返回 None。
    """
    mats = [H for H in heat_list if H is not None]
    if not mats:
        return None

    L_max = max(H.shape[0] for H in mats)
    W_max = max(H.shape[1] for H in mats)

    stack = []
    for H in mats:
        L, W = H.shape
        pad = np.full((L_max, W_max), np.nan, dtype=np.float64)
        pad[:L, :W] = H
        stack.append(pad)

    stack = np.stack(stack, axis=0)  # (N, L_max, W_max)
    avg = np.nanmean(stack, axis=0)  # 逐点忽略 NaN
    return avg.astype(np.float32)


def plot_text_true_tokens(avg_txt_H_list, real_txt_tokens, model_name, save_dir, dpi=160, share_scale=False):
    """
    逐个 demo 出图（与原函数同名同参），但横轴使用真实 token 个数，不做归一化。
    可选 share_scale=False 时：
        - 对于 image 模态：不同 demo 共享相同 vmin/vmax（来自所有 image 热图的 nanmin/nanmax）
        - 对于 text  模态：不同 demo 共享相同 vmin/vmax（来自所有 text  热图的 nanmin/nanmax）
    """
    os.makedirs(save_dir, exist_ok=True)
    safe = _sanitize_name(model_name) if "_sanitize_name" in globals() else model_name.replace("/", "_")
    txt_vmin = 0
    txt_vmax = 0.05


    # ===== 逐 demo：Text =====

    for i, H in enumerate(avg_txt_H_list[:-1]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 text token 个数

        fig, ax = plt.subplots(figsize=(max(6, W/12), 4), dpi=dpi)

        # 颜色范围
        vmin = txt_vmin if share_scale else np.nanmin(H)
        vmax = txt_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = vmax = None

        im = ax.imshow(
            H, origin="lower", interpolation="nearest", 
            aspect="equal", extent=[0, W, 0, L],
            vmin=vmin, vmax=vmax
        )

        # --- x 轴标签：去重 + 下采样 ---
        tokens_vis = list(real_txt_tokens[i])

        # 1) 连续重复的 token 置空（只保留第一次出现，减噪）
        for j in range(1, len(tokens_vis)):
            if tokens_vis[j] == tokens_vis[j-1]:
                tokens_vis[j] = ""  # 折叠连续相同 token
        # 如果是 BPE/Spm，可进一步把 "Ġ"/"▁" 合并成词后再显示第一个

        # 2) 对齐宽度
        tokens_vis = tokens_vis[:W] + [""] * max(0, W - len(tokens_vis))


        max_labels = 80
        step = int(np.ceil(W / max_labels)) if W > max_labels else 1
        tick_pos = np.arange(0, W, step) + 0.5
        tick_lab = [tokens_vis[j] for j in range(0, W, step)]

        ax.set_xlim(0, W)
        ax.set_ylim(0, L)
        ax.xaxis.set_major_locator(FixedLocator(tick_pos))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_lab))
        ax.tick_params(axis="x", labelrotation=90, labelsize=6, pad=1)

        ax.set_xlabel("Text token")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Demo{i+1} Text Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_demo{i+1}_text_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")

    for i, H in enumerate(avg_txt_H_list[-1:]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 text token 个数

        fig, ax = plt.subplots(figsize=(max(6, W/12), 4), dpi=dpi)

        # 颜色范围
        vmin = txt_vmin if share_scale else np.nanmin(H)
        vmax = txt_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = vmax = None

        im = ax.imshow(
            H, origin="lower", interpolation="nearest", 
            aspect="equal", extent=[0, W, 0, L],
            vmin=vmin, vmax=vmax
        )

        # --- x 轴标签：去重 + 下采样 ---
        tokens_vis = list(real_txt_tokens[-1])

        # 1) 连续重复的 token 置空（只保留第一次出现，减噪）
        for j in range(1, len(tokens_vis)):
            if tokens_vis[j] == tokens_vis[j-1]:
                tokens_vis[j] = ""  # 折叠连续相同 token
        # 如果是 BPE/Spm，可进一步把 "Ġ"/"▁" 合并成词后再显示第一个

        # 2) 对齐宽度
        tokens_vis = tokens_vis[:W] + [""] * max(0, W - len(tokens_vis))


        max_labels = 80
        step = int(np.ceil(W / max_labels)) if W > max_labels else 1
        tick_pos = np.arange(0, W, step) + 0.5
        tick_lab = [tokens_vis[j] for j in range(0, W, step)]

        ax.set_xlim(0, W)
        ax.set_ylim(0, L)
        ax.xaxis.set_major_locator(FixedLocator(tick_pos))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_lab))
        ax.tick_params(axis="x", labelrotation=90, labelsize=6, pad=1)

        ax.set_xlabel("Text token")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Demo{i+1} Text Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")


        out = os.path.join(save_dir, f"{safe}_AVERAGE_Query_text_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")


def plot_pair_true_tokens(avg_img_H_list, avg_txt_H_list,real_txt_tokens, model_name, save_dir, dpi=160, share_scale=False):
    """
    逐个 demo 出图（与原函数同名同参），但横轴使用真实 token 个数，不做归一化。
    可选 share_scale=False 时：
        - 对于 image 模态：不同 demo 共享相同 vmin/vmax（来自所有 image 热图的 nanmin/nanmax）
        - 对于 text  模态：不同 demo 共享相同 vmin/vmax（来自所有 text  热图的 nanmin/nanmax）
    """
    os.makedirs(save_dir, exist_ok=True)
    safe = _sanitize_name(model_name) if "_sanitize_name" in globals() else model_name.replace("/", "_")

    # ===== 计算共享色阶（可选） =====
    img_vmin = 0
    img_vmax = 0.02
    txt_vmin = 0
    txt_vmax = 0.05

    # ===== 逐 demo：Image =====
    for i, H in enumerate(avg_img_H_list[:-1]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 image token 个数

        # 宽度随列数自适应；常数 12 可按你的喜好调（越小图越宽）
        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        # 每张图若未设置 share_scale，就用自身的极值
        vmin = img_vmin if share_scale else np.nanmin(H)
        vmax = img_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None  # 退回自动色阶

        im = ax.imshow(
            H, origin="lower", interpolation="nearest", 
            aspect="equal",  # 每列等宽，列多时每列更细
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)

        ax.set_xlabel("Image token index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Demo{i+1} Image Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_demo{i+1}_image_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")


    for i, H in enumerate(avg_img_H_list[-1:]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 image token 个数

        # 宽度随列数自适应；常数 12 可按你的喜好调（越小图越宽）
        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        # 每张图若未设置 share_scale，就用自身的极值
        vmin = img_vmin if share_scale else np.nanmin(H)
        vmax = img_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None  # 退回自动色阶

        im = ax.imshow(
            H, origin="lower", interpolation="nearest", 
            aspect="equal",  # 每列等宽，列多时每列更细
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)
        ax.set_xlabel("Image token index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Query Image Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_Query_image_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")

    for i, H in enumerate(avg_txt_H_list[:-1]):
        if H is None:
            continue
        L, W = H.shape
        fig, ax = plt.subplots(figsize=(max(6, W/12), 4), dpi=dpi)

        # 颜色范围
        vmin = txt_vmin if share_scale else np.nanmin(H)
        vmax = txt_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = vmax = None

        im = ax.imshow(
            H, origin="lower", interpolation="nearest",
            aspect="equal", extent=[0, W, 0, L],
            vmin=vmin, vmax=vmax
        )

        # --- x 轴标签：去重 + 下采样 ---
        tokens_vis = list(real_txt_tokens[i])

        # 1) 连续重复的 token 置空（只保留第一次出现，减噪）
        for j in range(1, len(tokens_vis)):
            if tokens_vis[j] == tokens_vis[j-1]:
                tokens_vis[j] = ""  # 折叠连续相同 token
        # 如果是 BPE/Spm，可进一步把 "Ġ"/"▁" 合并成词后再显示第一个

        # 2) 对齐宽度
        tokens_vis = tokens_vis[:W] + [""] * max(0, W - len(tokens_vis))


        max_labels = 80
        step = int(np.ceil(W / max_labels)) if W > max_labels else 1
        tick_pos = np.arange(0, W, step) + 0.5
        tick_lab = [tokens_vis[j] for j in range(0, W, step)]

        ax.set_xlim(0, W)
        ax.set_ylim(0, L)
        ax.xaxis.set_major_locator(FixedLocator(tick_pos))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_lab))
        ax.tick_params(axis="x", labelrotation=90, labelsize=6, pad=1)

        ax.set_xlabel("Text token")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Demo{i+1} Text Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_demo{i+1}_text_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")

    for i, H in enumerate(avg_txt_H_list[-1:]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 text token 个数
        # print(H.shape, "H.shape") # (28, 24) H.shape
        fig, ax = plt.subplots(figsize=(max(6, W/12), 4), dpi=dpi)

        # 颜色范围
        vmin = txt_vmin if share_scale else np.nanmin(H)
        vmax = txt_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = vmax = None

        im = ax.imshow(
            H, origin="lower", interpolation="nearest", 
            aspect="equal", extent=[0, W, 0, L],
            vmin=vmin, vmax=vmax
        )
        tokens_vis = list(real_txt_tokens[-1])

        # 2) 对齐宽度
        tokens_vis = tokens_vis[:W] + [""] * max(0, W - len(tokens_vis))

        max_labels = 80
        step = int(np.ceil(W / max_labels)) if W > max_labels else 1
        tick_pos = np.arange(0, W, step) + 0.5
        tick_lab = [tokens_vis[j] for j in range(0, W, step)]

        ax.set_xlim(0, W)
        ax.set_ylim(0, L)
        ax.xaxis.set_major_locator(FixedLocator(tick_pos))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_lab))
        ax.tick_params(axis="x", labelrotation=90, labelsize=6, pad=1)

        ax.set_xlabel("Text token")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Demo{i+1} Text Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_Query_text_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")


def plot_text_true_tokens_wo_idx( avg_txt_H_list, model_name, save_dir, dpi=160, share_scale=False):
    """
    逐个 demo 出图（与原函数同名同参），但横轴使用真实 token 个数，不做归一化。
    可选 share_scale=True 时：
        - 对于 image 模态：不同 demo 共享相同 vmin/vmax（来自所有 image 热图的 nanmin/nanmax）
        - 对于 text  模态：不同 demo 共享相同 vmin/vmax（来自所有 text  热图的 nanmin/nanmax）
    """
    os.makedirs(save_dir, exist_ok=True)
    safe = _sanitize_name(model_name) if "_sanitize_name" in globals() else model_name.replace("/", "_")


    txt_vmin = txt_vmax = None
    if share_scale:
        txt_vals = [H for H in avg_txt_H_list if H is not None and np.isfinite(H).any()]
        txt_vmin = np.nanmin([np.nanmin(H) for H in txt_vals])
        txt_vmax = np.nanmax([np.nanmax(H) for H in txt_vals])
        if not np.isfinite(txt_vmin): txt_vmin = None
        if not np.isfinite(txt_vmax): txt_vmax = None

    # ===== 逐 demo：Text =====
    for i, H in enumerate(avg_txt_H_list[:-1]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 text token 个数

        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        vmin = txt_vmin if share_scale else np.nanmin(H)
        vmax = txt_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None

        im = ax.imshow(
            H, origin="lower", interpolation="nearest", 
            aspect="equal",
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)
        ax.set_xlabel("Text token index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Demo{i+1} Text Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_demo{i+1}_text_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")

    # ===== query：Text =====
    for i, H in enumerate(avg_txt_H_list[-1:]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 text token 个数

        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        vmin = txt_vmin if share_scale else np.nanmin(H)
        vmax = txt_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None

        im = ax.imshow(
            H, origin="lower", interpolation="nearest",
            aspect="equal",
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)
        ax.set_xlabel("Text token index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Query Text Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_Query_text_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")



def plot_pair_true_tokens_wo_idx(avg_img_H_list, avg_txt_H_list, model_name, save_dir, dpi=160, share_scale=False):
    """
    逐个 demo 出图（与原函数同名同参），但横轴使用真实 token 个数，不做归一化。
    可选 share_scale=True 时：
        - 对于 image 模态：不同 demo 共享相同 vmin/vmax（来自所有 image 热图的 nanmin/nanmax）
        - 对于 text  模态：不同 demo 共享相同 vmin/vmax（来自所有 text  热图的 nanmin/nanmax）
    """
    os.makedirs(save_dir, exist_ok=True)
    safe = _sanitize_name(model_name) if "_sanitize_name" in globals() else model_name.replace("/", "_")

    # ===== 计算共享色阶（可选） =====
    img_vmin = img_vmax = None
    txt_vmin = txt_vmax = None
    if share_scale:
        img_vals = [H for H in avg_img_H_list if H is not None and np.isfinite(H).any()]
        txt_vals = [H for H in avg_txt_H_list if H is not None and np.isfinite(H).any()]
        if img_vals:
            img_vmin = np.nanmin([np.nanmin(H) for H in img_vals])
            img_vmax = np.nanmax([np.nanmax(H) for H in img_vals])
            if not np.isfinite(img_vmin): img_vmin = None
            if not np.isfinite(img_vmax): img_vmax = None
        if txt_vals:
            txt_vmin = np.nanmin([np.nanmin(H) for H in txt_vals])
            txt_vmax = np.nanmax([np.nanmax(H) for H in txt_vals])
            if not np.isfinite(txt_vmin): txt_vmin = None
            if not np.isfinite(txt_vmax): txt_vmax = None

    # ===== 逐 demo：Image =====
    for i, H in enumerate(avg_img_H_list[:-1]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 image token 个数

        # 宽度随列数自适应；常数 12 可按你的喜好调（越小图越宽）
        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        # 每张图若未设置 share_scale，就用自身的极值
        vmin = img_vmin if share_scale else np.nanmin(H)
        vmax = img_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None  # 退回自动色阶

        im = ax.imshow(
            H, origin="lower", interpolation="nearest",
            aspect="equal",  # 每列等宽，列多时每列更细
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)
        ax.set_xlabel("Image token index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Demo{i+1} Image Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_demo{i+1}_image_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")


    for i, H in enumerate(avg_img_H_list[-1:]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 image token 个数

        # 宽度随列数自适应；常数 12 可按你的喜好调（越小图越宽）
        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        # 每张图若未设置 share_scale，就用自身的极值
        vmin = img_vmin if share_scale else np.nanmin(H)
        vmax = img_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None  # 退回自动色阶

        im = ax.imshow(
            H, origin="lower", interpolation="nearest",
            aspect="equal",  # 每列等宽，列多时每列更细
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)
        ax.set_xlabel("Image token index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Query Image Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_Query_image_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")


    # ===== 逐 demo：Text =====
    for i, H in enumerate(avg_txt_H_list[:-1]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 text token 个数

        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        vmin = txt_vmin if share_scale else np.nanmin(H)
        vmax = txt_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None

        im = ax.imshow(
            H, origin="lower", interpolation="nearest",
            aspect="equal",
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)
        ax.set_xlabel("Text token index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Demo{i+1} Text Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_demo{i+1}_text_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")

        # ===== query：Text =====
    for i, H in enumerate(avg_txt_H_list[-1:]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 text token 个数

        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        vmin = txt_vmin if share_scale else np.nanmin(H)
        vmax = txt_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None

        im = ax.imshow(
            H, origin="lower", interpolation="nearest",
            aspect="equal",
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)
        ax.set_xlabel("Text token index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Query Text Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_Query_text_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")



def plot_text_heads_wo_idx( avg_txt_H_list, model_name, save_dir, dpi=160, share_scale=False):
    """
    逐个 demo 出图（与原函数同名同参），但横轴使用真实 token 个数，不做归一化。
    可选 share_scale=True 时：
        - 对于 image 模态：不同 demo 共享相同 vmin/vmax（来自所有 image 热图的 nanmin/nanmax）
        - 对于 text  模态：不同 demo 共享相同 vmin/vmax（来自所有 text  热图的 nanmin/nanmax）
    """
    os.makedirs(save_dir, exist_ok=True)
    safe = _sanitize_name(model_name) if "_sanitize_name" in globals() else model_name.replace("/", "_")


    txt_vmin = txt_vmax = None
    if share_scale:
        txt_vals = [H for H in avg_txt_H_list if H is not None and np.isfinite(H).any()]
        txt_vmin = np.nanmin([np.nanmin(H) for H in txt_vals])
        txt_vmax = np.nanmax([np.nanmax(H) for H in txt_vals])
        if not np.isfinite(txt_vmin): txt_vmin = None
        if not np.isfinite(txt_vmax): txt_vmax = None

    # ===== 逐 demo：Text =====
    for i, H in enumerate(avg_txt_H_list[:-1]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 text token 个数

        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        vmin = txt_vmin if share_scale else np.nanmin(H)
        vmax = txt_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None

        im = ax.imshow(
            H, origin="lower", interpolation="nearest", 
            aspect="equal",
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)
        ax.set_xlabel("Attention Heads index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Demo{i+1} Text Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_demo{i+1}_text_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")

    # ===== query：Text =====
    for i, H in enumerate(avg_txt_H_list[-1:]):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 text token 个数

        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        vmin = txt_vmin if share_scale else np.nanmin(H)
        vmax = txt_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None

        im = ax.imshow(
            H, origin="lower", interpolation="nearest",
            aspect="equal",
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)
        ax.set_xlabel("Attention Heads index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Query Text Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_Query_text_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")



def plot_pair_heads_wo_idx(avg_img_H_list, avg_txt_H_list, model_name, save_dir, dpi=160, share_scale=False):
    """
    逐个 demo 出图（与原函数同名同参），但横轴使用真实 token 个数，不做归一化。
    可选 share_scale=True 时：
        - 对于 image 模态：不同 demo 共享相同 vmin/vmax（来自所有 image 热图的 nanmin/nanmax）
        - 对于 text  模态：不同 demo 共享相同 vmin/vmax（来自所有 text  热图的 nanmin/nanmax）
    """
    os.makedirs(save_dir, exist_ok=True)
    safe = _sanitize_name(model_name) if "_sanitize_name" in globals() else model_name.replace("/", "_")

    # ===== 计算共享色阶（可选） =====
    img_vmin = img_vmax = None
    txt_vmin = txt_vmax = None
    if share_scale:
        img_vals = [H for H in avg_img_H_list if H is not None and np.isfinite(H).any()]
        txt_vals = [H for H in avg_txt_H_list if H is not None and np.isfinite(H).any()]
        if img_vals:
            img_vmin = np.nanmin([np.nanmin(H) for H in img_vals])
            img_vmax = np.nanmax([np.nanmax(H) for H in img_vals])
            if not np.isfinite(img_vmin): img_vmin = None
            if not np.isfinite(img_vmax): img_vmax = None
        if txt_vals:
            txt_vmin = np.nanmin([np.nanmin(H) for H in txt_vals])
            txt_vmax = np.nanmax([np.nanmax(H) for H in txt_vals])
            if not np.isfinite(txt_vmin): txt_vmin = None
            if not np.isfinite(txt_vmax): txt_vmax = None

    # ===== 逐 demo：Image =====
    for i, H in enumerate(avg_img_H_list):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 image token 个数

        # 宽度随列数自适应；常数 12 可按你的喜好调（越小图越宽）
        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        # 每张图若未设置 share_scale，就用自身的极值
        vmin = img_vmin if share_scale else np.nanmin(H)
        vmax = img_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None  # 退回自动色阶

        im = ax.imshow(
            H, origin="lower", interpolation="nearest",
            aspect="equal",  # 每列等宽，列多时每列更细
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)
        ax.set_xlabel("Attention Heads index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Demo's' Image Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_demos_image_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")



    # ===== 逐 demo：Text =====
    for i, H in enumerate(avg_txt_H_list):
        if H is None:
            continue
        L, W = H.shape  # L=层数, W=真实 text token 个数

        fig = plt.figure(figsize=(max(6, W / 12), 4), dpi=dpi)
        ax = fig.add_subplot(111)

        vmin = txt_vmin if share_scale else np.nanmin(H)
        vmax = txt_vmax if share_scale else np.nanmax(H)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = None; vmax = None

        im = ax.imshow(
            H, origin="lower", interpolation="nearest",
            aspect="equal",
            extent=[0, W, 0, L], vmin=vmin, vmax=vmax
        )
        ax.set_xlim(0, W); ax.set_ylim(0, L)
        ax.set_xlabel("Attention Heads index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Label → Demo's Text Tokens (Average) — {model_name}")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention")

        out = os.path.join(save_dir, f"{safe}_AVERAGE_demes_text_heatmap.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out}")




