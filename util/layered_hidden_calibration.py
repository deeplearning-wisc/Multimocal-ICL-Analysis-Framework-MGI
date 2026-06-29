# Official implementation of paper "Revisiting In-context Learning Inference Circuit in Large Language Models"
# Author: Hakaze Cho, yfzhao@jaist.ac.jp

from StaICC.util import stable_random, functional
from tqdm import tqdm as tqdm

# proto_calib.py
import torch
import torch.nn.functional as F
from typing import List, Iterable, Union

# proto_calib_fixed.py
import torch
import torch.nn.functional as F
from typing import List, Iterable, Union

ArrayLike = Union[torch.Tensor, "np.ndarray", List[float], List[List[float]]]

def _to_tensor_batch(lst: Iterable[ArrayLike],
                     device: torch.device,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for x in lst:
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        t = t.detach().to(device=device, dtype=dtype)
        if t.dim() == 1:
            t = t.unsqueeze(0)            # [D] -> [1,D]
        elif t.dim() > 2:
            t = t.view(t.size(0), -1)     # [N,*] -> [N,D]
        tensors.append(t)
    if not tensors:
        raise ValueError("Empty class.")
    return torch.cat(tensors, dim=0)       # [N,D]

class ProtoCalib:
    """
    多原型原型分类器（全局中心化 + 可选 L2 归一化）。
    metric: "cos" | "l2" | "maha"
    兼容旧接口: inference(hidden_state) -> list[float]
    """
    def __init__(self,
                 n_protos_per_class: int = 1,
                 metric: str = "cos",
                 device: Union[str, torch.device] = "cpu",
                 center: bool = True,          # 使用【全局】均值中心化
                 l2_normalize: bool = True):   # 余弦建议 True
        self.np = int(max(1, n_protos_per_class))
        self.metric = metric
        self.device = torch.device(device)
        self.center = center
        self.l2_normalize = l2_normalize

        self.centroids: torch.Tensor | None = None   # [C,P,D]
        self.means: torch.Tensor | None = None       # [C,D] (maha)
        self.covs_inv: torch.Tensor | None = None    # [C,D,D] (maha)
        self.global_mean: torch.Tensor | None = None # [D]
        self.T = torch.tensor(1.0, device=self.device)
        self._D: int | None = None

    @torch.no_grad()
    def train(self, hidden_states_with_labels: List[Iterable[ArrayLike]]) -> None:
        # 1) 整理成每类张量
        class_feats = [_to_tensor_batch(lst, self.device) for lst in hidden_states_with_labels]
        C = len(class_feats)
        if C == 0:
            raise ValueError("No classes.")
        D = class_feats[0].shape[1]
        self._D = D

        # 2) 用【全体样本】的全局均值做中心化，避免原型塌到零
        if self.center:
            all_feats = torch.vstack(class_feats)           # [N_total, D]
            self.global_mean = all_feats.mean(0, keepdim=True)  # [1,D]
        else:
            self.global_mean = torch.zeros(1, D, device=self.device)

        feats = []
        for f in class_feats:
            g = f - self.global_mean                        # 全局中心化
            if self.l2_normalize:
                g = F.normalize(g, dim=1)
            feats.append(g)                                  # 每类 [N_k, D]

        # 3) 计算每类的 P 个原型（均匀分块求均值），并再次按需要做 L2
        P = self.np
        cents = []
        for g in feats:
            N, D = g.shape
            if N <= P:
                c = g.mean(0, keepdim=True)                 # [1,D]
                if self.l2_normalize: c = F.normalize(c, dim=1)
                c = c.repeat(P, 1).contiguous()             # [P,D]
            else:
                idx = torch.randperm(N, device=g.device)
                pieces = torch.tensor_split(g[idx], P)
                proto_list = []
                for p in pieces:
                    mu = p.mean(0, keepdim=True)            # [1,D]
                    if self.l2_normalize: mu = F.normalize(mu, dim=1)
                    proto_list.append(mu)
                c = torch.cat(proto_list, dim=0)            # [P,D]
            cents.append(c.to(self.device, dtype=torch.float32))
        # 形状一致后堆叠
        shapes = {tuple(t.shape) for t in cents}
        if len(shapes) != 1:
            raise RuntimeError(f"Prototype shapes mismatch: {shapes}")
        self.centroids = torch.stack(cents, dim=0).contiguous()  # [C,P,D]

        # 4) （可选）马氏距离的均值/协方差（在未中心化的原始特征上估计更稳）
        if self.metric == "maha":
            means, cov_invs = [], []
            I = torch.eye(D, device=self.device)
            for f in class_feats:
                X = f - f.mean(0, keepdim=True)
                cov = (X.T @ X) / max(1, X.size(0)-1)
                lam = 0.1
                cov = (1-lam)*cov + lam*I
                cov_invs.append(torch.inverse(cov))
                means.append(f.mean(0))
            self.means = torch.stack(means, dim=0)
            self.covs_inv = torch.stack(cov_invs, dim=0)

        self.T = torch.tensor(1.0, device=self.device)
        print(f"[ProtoCalib] Trained. C={C}, P={P}, D={D}, metric={self.metric}, center=global")

    @torch.no_grad()
    def _prepare_x(self, x: ArrayLike) -> torch.Tensor:
        X = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        X = X.detach().to(self.device, dtype=torch.float32)
        if X.dim() == 1: X = X.unsqueeze(0)
        elif X.dim() > 2: X = X.view(X.size(0), -1)
        if self._D is not None and X.size(1) != self._D:
            raise ValueError(f"Input dim {X.size(1)} != trained dim {self._D}")
        # 用【训练时的全局均值】中心化
        if self.global_mean is not None:
            X = X - self.global_mean
        if self.l2_normalize:
            X = F.normalize(X, dim=1)
        return X

    @torch.no_grad()
    def _logits(self, x: ArrayLike) -> torch.Tensor:
        X = self._prepare_x(x)                                # [B,D]
        C, P, D = self.centroids.shape                        # type: ignore

        if self.metric == "cos":
            sim = torch.matmul(X, self.centroids.view(C*P, D).T)  # [B,C*P]
            sim = sim.view(X.size(0), C, P).max(dim=2).values     # [B,C]
            return sim / self.T

        if self.metric == "l2":
            diff = X[:, None, None, :] - self.centroids[None, :, :, :]  # [B,C,P,D]
            dist2 = (diff**2).sum(-1).min(2).values                      # [B,C]
            return (-dist2) / self.T

        if self.metric == "maha":
            if self.means is None or self.covs_inv is None:
                raise RuntimeError("Mahalanobis needs means/covs_inv.")
            diff = X[:, None, :] - self.means[None, :, :]                # [B,C,D]
            dist = torch.einsum('bcd,cde,bce->bc', diff, self.covs_inv, diff)
            return (-dist) / self.T

        raise ValueError(f"Unknown metric: {self.metric}")

    @torch.no_grad()
    def predict_proba(self, x: ArrayLike) -> torch.Tensor:
        return F.softmax(self._logits(x), dim=-1)

    @torch.no_grad()
    def predict(self, x: ArrayLike) -> torch.Tensor:
        return self._logits(x).argmax(dim=-1)

    # 兼容你旧接口：返回 list[float]
    @torch.no_grad()
    def inference(self, hidden_state) -> list[float]:
        probs = self.predict_proba(hidden_state)  # [1,C] or [B,C]
        if probs.dim() == 2 and probs.size(0) == 1:
            return probs.squeeze(0).detach().cpu().tolist()
        return probs.detach().cpu().tolist()


class hidden_calibration():
    # https://arxiv.org/abs/2406.16535
    def __init__(self, label_space) -> None:
        n_label = len(label_space)
        self.n_label = n_label
        self.centroid = []

    def train(
        self, 
        hidden_states_with_labels: list[list[float]]
    ):
        for list in hidden_states_with_labels:
            sum = [0] * len(list[0])
            for hidden_state in list:
                for i in range(len(hidden_state)):
                    sum[i] += hidden_state[i]
            self.centroid.append([x / len(list) for x in sum])
        print("Calibration Training Finished.\n")

    def inference(self, hidden_state) -> list[float]:
        L2_dist = [functional.L2_dist(hidden_state.tolist(), self.centroid[i]) for i in range(self.n_label)]
        normlized = [L2_dist[0] - L2_dist[i] for i in range(0, len(L2_dist))]
        return functional.softmax(normlized)
    
class layered_hidden_calibration():
    def __init__(self, label_space, layer_number, prompt_cut = "none", target_label_correction = True) -> None:
        self.label_space = label_space
        self.calibrations = []
        self.n_label = len(label_space)
        for i in range(layer_number):
            self.calibrations.append(hidden_calibration(self.n_label))
        self.prompt_cut = prompt_cut
        self.target_label_correction = target_label_correction
    
    def train(
        self, 
        default_prompt_maker: callable,
        feedforward_with_layered_hidden_state: callable, 
        calibration_set = None,
        calibration_number = 128,
        k = 4
    ):
        hidden_states = [[[] for _ in range(self.n_label)] for _ in range(len(self.calibrations))]
        my_random = stable_random.stable_random()
        demonstration_and_queue_samples = my_random.sample_index_set(calibration_number * (k + 1), len(calibration_set), allow_repetition=True)
        for i in range(calibration_number):
            print("\r", end="")
            print("Process: {}%, {} in {}".format(
                int((i + 1) / calibration_number * 100), 
                (i + 1), 
                calibration_number
            ), ">>" * int((i + 1) / calibration_number * 32), end="")

            '''
                prepare the dataset,
            '''
            demonstration_idx = demonstration_and_queue_samples[i * (k + 1) : (i + 1) * (k + 1) - 1]

            demo_text_lines = []
            demo_image_lines = []
            for index in demonstration_idx:
                label_token = calibration_set.get_label(index)
                demo_text_lines.append((calibration_set.get_input_text(index), label_token))
                demo_image_lines.append((calibration_set.get_input_image(index)))

            query_image_line = []
            query_idx= demonstration_and_queue_samples[(i + 1) * (k + 1) - 1] # query_sample  384
            query_text_line = calibration_set.get_input_text(query_idx)
            query_image_line.append(calibration_set.get_input_image(query_idx))


            query_label_index = calibration_set.find_index_from_label(calibration_set.get_label(query_idx))
            if not self.target_label_correction and self.prompt_cut == "label_words":
                print(not self.target_label_correction)
                query_label_index = (query_label_index + 1) % len(calibration_set._label_space)
            
            prompt = default_prompt_maker.write_prompt_from_dataline(demo_text_lines, demo_image_lines, query_text_line, query_image_line)

            if self.prompt_cut == "none":
                cut_amount = -1
                prompt['text'][-1] = prompt['text'][-1][:cut_amount]
            elif self.prompt_cut == "label_words":
                prompt['text'][-1] = prompt['text'][-1] + calibration_set._label_space[query_label_index] + ' '

            elif self.prompt_cut == "last_sentence_token":
                label_prefix_length = len(default_prompt_maker._label_prefix)
                cut_amount = -label_prefix_length - 1
                prompt['text'][-1] = prompt['text'][-1][:cut_amount]

            elif self.prompt_cut == "last_image_token":
                prompt['text'][-1] = ""
            elif self.prompt_cut == "only_text_last_sentence_token":
                prompt['image'][-1] =""
                
            '''
                obatin the hidden_state embedding
            '''

            hidden_state = feedforward_with_layered_hidden_state(prompts = [prompt])[0]
            for i in range(len(self.calibrations)):
                hidden_states[i][query_label_index].append(hidden_state[i])
        
        for i in range(len(self.calibrations)):
            self.calibrations[i].train(hidden_states[i])
    
    def single_layered_inference(self, layered_hidden_state_for_one_sample: list[list[float]]) -> list[list[float]]: # [layer][hidden_state] -> [layer][label_prob]
        ret = []
        for i in range(len(self.calibrations)):
            ret.append(self.calibrations[i].inference(layered_hidden_state_for_one_sample[i]))
        return ret

    def batched_layered_inference(self, layered_hidden_states: list[list[list[float]]]) -> list[list[list[float]]]: # [layer][sample][hidden_state] -> [layer][sample][label_prob]
        ret = [[] for _ in range(len(self.calibrations))]
        for sample_index in tqdm(range(len(layered_hidden_states[0]))):
            layered_hidden_state = []
            for i in range(len(self.calibrations)):
                layered_hidden_state.append(layered_hidden_states[i][sample_index])
            singleres = self.single_layered_inference(layered_hidden_state)
            for i in range(len(self.calibrations)):
                ret[i].append(singleres[i])
        return ret
