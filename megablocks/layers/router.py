from megablocks.layers import common
from megablocks.layers.arguments import Arguments
import torch
from ipdb import set_trace as bp

# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)
_uniform_expert_assignment = _UniformExpertAssignment.apply


class LearnedRouter(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device)
        args.init_method(self.layer.weight)

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1,keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x, expert_mask = None):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        if self.args.moe_expert_choice:
            # Get probability for each token
            bs, sq, _ = x.shape
            capacity = self.args.moe_top_k # Use top k as the capacity to match regular MoEs
            logits = self.layer(x)
            scores = logits.softmax(dim=-1) # [batch_size, seq_len, num_experts]
            expert_weights, expert_indices = torch.topk(scores.transpose(1,2), (capacity * sq) // self.args.moe_num_experts, dim=-1) # [batch_size, num_experts, k]
        elif self.args.moe_expert_choice_grouped:
            bs, sq, _ = x.shape
            capacity = self.args.moe_top_k # Use top k as the capacity to match regular MoEs
            logits = self.layer(x.view(-1, x.shape[-1])) # [bs & sq, num_experts]
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = torch.topk(scores.transpose(0,1),  (capacity * bs * sq) // self.args.moe_num_experts, dim=-1) # [num_experts, k]
        else:
            bs, sq, _ = x.shape
            expert_mask_shape = 2
            num_exp = self.args.moe_num_experts

            logits = self.layer(x.view(-1, x.shape[-1]))
            # implement here

            scores = logits.softmax(dim=-1)
            
            # swj add mask: 
            '''
            bs, sq, _ = x.shape [4 batch size, 4096 seq length, 64 experts]
            scores [bs x sq, num_experts] = [4 batch size x 4096 seq length, 64 experts]
            expert_mask: [4 batch size, 64 experts] shape
            '''
            if expert_mask is not None: # expert_mask is not None: # expert_mask is not None:
                # bp()
                # mem+:
                scores = scores.reshape(bs, sq, num_exp)
                scores = expert_mask.unsqueeze(1) * scores
                scores = scores.reshape(bs * sq, num_exp)
                # .expand(-1, sq, expert_mask_shape)
                # # merge batch size and seq length
                # expanded_mask = expanded_mask.reshape(-1, expert_mask_shape)
                # assert expanded_mask.shape == scores.shape
                # apply mask
                # scores = scores * expanded_mask
                # print("applied masks")

                # mem-:
                # scores = scores.reshape(bs, sq, num_exp)
                # mask = torch.zeros_like(scores)
                # for i in range(expert_mask.shape[0]):  # For each of the 4 items
                #     start, end = expert_mask[i]
                #     mask[i, :, start:end] = 1
                # scores = scores * mask
                # # bp()
                # scores = scores.reshape(bs * sq, num_exp)
                # del mask

     
            # test mask
            # expert_weights: [8192, 8], 8192 layer, select 8 experts per layer
            # scores_new = scores * expanded_mask
            # expert_weights, expert_indices_new = self._top_k(scores_new)
            expert_weights, expert_indices = self._top_k(scores)

            # swj apply softmax here again over the expert_weights
            # if False:
            expert_weights = expert_weights.softmax(dim=-1)

        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights, p=self.args.moe_normalize_expert_weights,dim=-1, keepdim=True)

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment else expert_indices
        )
        return scores, logits, expert_weights, expert_indices
