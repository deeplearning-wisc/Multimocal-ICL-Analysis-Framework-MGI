import numpy as np

def _to_arr(x):
    # x: List[np.ndarray] or List[list[float]] -> np.ndarray
    # 目标：shape = [P, D]
    return np.asarray(x, dtype=np.float32)

def _cosine_distance(A, B, eps=1e-8):
    # A,B: [P, D]
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    # 距离 = 1 - 相似度
    sim = np.sum(A_n * B_n, axis=1)
    return 1.0 - sim  # [P]

def _l2_distance(A, B):
    # A,B: [P, D]
    return np.linalg.norm(A - B, axis=1)  # [P]

def _center(K):
    # K: [P, P]
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def _linear_cka(A, B, eps=1e-8):
    """
    线性 CKA（越大越相似，取1-CKA作为距离）
    A,B: [P, D]
    """
    KA = A @ A.T
    KB = B @ B.T
    KAc = _center(KA)
    KBc = _center(KB)
    hsic = np.sum(KAc * KBc)
    normA = np.sqrt(np.sum(KAc * KAc) + eps)
    normB = np.sqrt(np.sum(KBc * KBc) + eps)
    cka = hsic / (normA * normB + eps)
    return 1.0 - cka  # 标量“距离”

def layerwise_distance(H_with, H_without, metric="cosine"):
    """
    H_with:  [L][P][D]  -> ICL_hidden_states（加了演示的）
    H_without:[L][P][D] -> query_hidden_states（Query-only）
    返回: 
      - 若 metric in {"cosine","l2"}: List[float] (每层“样本距离均值”)
      - 若 metric == "cka":           List[float] (每层 1-CKA)
    """
    assert len(H_with) == len(H_without), "层数不一致"
    L = len(H_with)
    ret = []
    for l in range(L):
        A = _to_arr(H_with[l])    # [P, D]
        B = _to_arr(H_without[l]) # [P, D]
        P = min(len(A), len(B))
        A = A[:P]
        B = B[:P]

        if metric == "cosine":
            d = _cosine_distance(A, B)     # [P]
            ret.append(float(np.mean(d)))
        elif metric == "l2":
            d = _l2_distance(A, B)         # [P]
            ret.append(float(np.mean(d)))
        elif metric == "cka":
            d = _linear_cka(A, B)          # 标量(1-CKA)
            ret.append(float(d))
        else:
            raise ValueError(f"unknown metric: {metric}")
    return ret

# # ================= 使用示例 =================
# # ICL_hidden_states, query_hidden_states, pesudo_query_hidden_states
# # 都来自你已有的 ICL_inference_to_hidden_states(...)
# avg_cos_layer = layerwise_distance(ICL_hidden_states, query_hidden_states, metric="cosine")
# avg_l2_layer  = layerwise_distance(ICL_hidden_states, query_hidden_states, metric="l2")
# avg_cka_layer = layerwise_distance(ICL_hidden_states, query_hidden_states, metric="cka")

# # 作为对照实验（用“伪查询”）
# avg_cos_layer_ref = layerwise_distance(ICL_hidden_states, pesudo_query_hidden_states, metric="cosine")
