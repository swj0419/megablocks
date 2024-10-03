from megablocks.layers import common
from megablocks.layers import mpu
from megablocks.layers import router
from megablocks.layers import mlp
from megablocks.layers import sharedexpert_registry
from megablocks.layers.all_to_all import all_to_all
from megablocks.layers.arguments import Arguments
import megablocks.ops as ops
import numpy as np
import torch
from typing import Optional


_LOAD_BALANCING_LOSS = []


def save_load_balancing_loss(loss):
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.append(loss)


def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS


def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()


def batched_load_balancing_loss(args : Arguments):
    if args.moe_loss_weight == 0:
        return 0.0

    # tokens_per_expert[i].shape = (num_experts)
    # expert_scores[i].shape = (tokens, num_experts)
    # tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())
    tokens_per_expert, expert_scores, expert_logits = zip(*get_load_balancing_loss())
    num_layers_per_pipeline_stage = (
        args.num_layers // args.pipeline_model_parallel_size)
    if args.num_layers_per_virtual_pipeline_stage is not None:
        num_layers_per_pipeline_stage = args.num_layers_per_virtual_pipeline_stage

    if len(tokens_per_expert) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} token_per_experts "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{args.num_layers}\npipeline_model_parallel_size = "
            f"{args.pipeline_model_parallel_size}\n"
            "num_layers_per_virtual_pipeline_stage"
            f" = {args.num_layers_per_virtual_pipeline_stage}")
    if len(expert_scores) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} expert_scores "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{args.num_layers}\npipeline_model_parallel_size = "
            f"{args.pipeline_model_parallel_size}\n"
            "num_layers_per_virtual_pipeline_stage"
            f" = {args.num_layers_per_virtual_pipeline_stage}")

    # Verify the shape of the tokens_per_expert and expert_scores tensors.
    assert all([
        x.ndim == 1 and x.numel() == args.moe_num_experts
        for x in tokens_per_expert
    ])

    tokens = expert_scores[0].shape[0]
    assert all([
        (x.ndim == 2 and x.shape[1] == args.moe_num_experts and
         x.shape[0] == tokens) for x in expert_scores
    ])


    # Concatenate the contributions of each layer and convert to
    # the correct types and formats for the dot product.
    expert_scores = torch.cat(expert_scores, dim=1)
    if args.moe_lbl_in_fp32:
        expert_scores = expert_scores.float()
    if tokens != 0:
        expert_scores = expert_scores.mean(dim=0)
    else:
        expert_scores = expert_scores.sum(dim=0)
    tokens_per_expert = torch.cat(tokens_per_expert).to(expert_scores.dtype)
    expert_logits = torch.cat(expert_logits, dim=0).to(expert_scores.dtype)

    expected_values = num_layers_per_pipeline_stage * args.moe_num_experts
    assert tokens_per_expert.numel() == expected_values
    assert expert_scores.numel() == expected_values

    # Calculate the total scale across all factors.
    #
    # loss_weight * num_experts / (num_layers * tokens * top_k)
    scale_numerator = (
        args.moe_num_experts *
        args.moe_loss_weight
    )
    scale_denominator = (
        args.num_layers *
        tokens *
        args.moe_top_k
    )
    scale = scale_numerator / scale_denominator
    zloss = (torch.log(torch.exp(expert_logits).sum(dim=-1)) ** 2).sum() / scale_denominator
    return scale * torch.dot(tokens_per_expert, expert_scores), args.moe_zloss_weight * zloss    


# NOTE: This class defines MoE expert computation, including expert model parallel
# communication. When using FSDP on top of MegaBlocks this is the module that should
# be wrapped s.t. the weight all-gathers can be scheduled *before* the expert model
# parallel all2all.
class ParallelMLP(torch.nn.Module):

    def __init__(self, args : Arguments):
        super(ParallelMLP, self).__init__()
        self.args = args

        # Calculate the number of experts in total and the number of experts
        # owned by this rank.
        world_size = mpu.get_expert_parallel_world_size(args)
        self.num_experts = args.moe_num_experts
        self.top_k = self.args.moe_top_k

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)

        # Expert MLP.
        self.mlp = mlp.MLP(args)

        if self.args.bias:
            # Note that the output bias is not parallelized with expert
            # model parallelism.
            self.bias = torch.nn.Parameter(torch.empty(
                args.hidden_size,
                device=args.device,
                dtype=common.dtype(args)))
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        # Select the forward function for the operating mode.
        if self.args.moe_expert_choice_grouped:
            self.forward = self.forward_ec_grouped
        if self.args.moe_expert_choice:
            self.forward = self.forward_ec
        elif args.moe_expert_model_parallelism:
            self.forward_fn = self.parallel_forward_once
        else:
            self.forward_fn = self.forward_once

    def expert_capacity(self, tokens):
        world_size = mpu.get_expert_parallel_world_size(self.args)
        tokens_per_expert = (
            self.top_k * tokens * world_size / self.num_experts)
        return int(self.args.moe_capacity_factor * tokens_per_expert)

    def load_balancing_loss(self, tokens_per_expert, expert_scores):
        """Calculate the load balancing loss contribution."""
        assert len(expert_scores.size()) == 2
        tokens, num_experts = expert_scores.size()
        assert num_experts == self.num_experts
        assert len(tokens_per_expert.size()) == 1
        num_experts, = tokens_per_expert.size()
        assert num_experts == self.num_experts
        scale = self.num_experts / (tokens * self.top_k)
        return scale * torch.dot(
            tokens_per_expert.to(expert_scores.dtype),
            expert_scores.mean(dim=0))

    def indices_and_bins(self, top_expert):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        # bin_ids is sorted list of expert ids
        # e.g. if top_expert is [0,1,0] bin_ids is [0,0,1]
        # indices is the expert indices of the bin_ids 
        # e.g. for the above it'd be [0,2,1]
        # Both are of shape [sl * bs * topk]        
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        # tokens_per_expert is the number of tokens assigned to each expert
        # shape [num_experts] e.g. for the above example it'd be [2,1]
        tokens_per_expert = ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        # bins becomes the cumulative sum of tokens_per_expert e.g. for above example it'd be [2,3]
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    def permute_and_compute(
            self,
            x,
            tokens_per_expert, # unused
            indices,
            bin_ids, # unused
            expert_weights,
            bins,
            expert_capacity,
            top_k):
        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1]) # [bs * sl, hs]
        # Permutes the tokens based on the expert indices
        # x shape: [num_experts, exp capacity, hid_dim]
        # exp capacity includes topk i.e. topk * mbs * sl / num_experts
        x = ops.binned_gather(
            x, indices, bins, expert_capacity, top_k)

        # Perform the expert computation. Note that we don't
        # use biases for these linear operations.
        # mlp.w1.shape: [num_experts, hid_dim, ffn_dim]
        x = self.mlp(x)

        # Un-route the data for the MoE output.
        return ops.binned_scatter(
            x, indices, expert_weights, bins, top_k)

    def forward_once(self, x, expert_weights, top_experts):
        # x: [bs, sl, hs]
        # expert_weights: [sl * bs, top-k]
        # top_experts: [sl * bs, top-k]
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_experts))

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            # Usually expert capacity is >0 and tokens are dropped!
            bs, sl, hs = x.size()
            expert_capacity = self.expert_capacity(sl * bs)
            if expert_capacity == 0:
                expert_capacity = torch.max(tokens_per_expert).item()

        x = self.permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            expert_capacity,
            self.top_k)
        return x, tokens_per_expert

    def parallel_forward_once(self, x, expert_weights, top_experts):
        # NOTE: This function implements the same computation as forward_once
        # but with expert model parallelism.
        #
        # 1. Permute the tokens locally so that they are grouped by their
        # expert assignments. This allows us to transfer all of the tokens
        # for a remote device in one communication primitive.
        #
        # 2. Permute the tokens across the expert parallel devices. After
        # this is completed each device has all of the tokens assigned to
        # its set of experts in its local HBM.
        #
        # 3. Permute the tokens locally so that they are grouped by their
        # expert assignement. After the distributed permutation the tokens
        # are grouped by which device they came from. We re-order them
        # locally to allow for efficient computation.
        #
        # After this series of permutations we compute the linear layers
        # and then repeat these three steps in reverse to produce the final
        # output.
        #
        # Compute the mapping of local tokens to experts.
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_experts))

            # If we're sharding the experts along the hidden dimension
            # multiple devices own parts of the same sets of experts.
            # Replicate the token counts so every device gets the counts.
            repeated_tokens_per_expert = ops.repeat(
                tokens_per_expert, (mpu.hidden_sharding_degree(self.args),))

            # Pass token count information to the device on which the
            # target expert resides.
            parallel_tokens_per_expert = torch.empty_like(repeated_tokens_per_expert)
            tpe_handle = torch.distributed.all_to_all_single(
                parallel_tokens_per_expert,
                repeated_tokens_per_expert,
                group=self.args.expert_parallel_group,
                async_op=True)

        # Permute locally and without any padding so that tokens for each
        # parallel device are stored contiguously.
        #
        # This view updates the shape of the tensor from [sl, bs, hs] to
        # [sl * bs, hs] prior to the permutation.
        x = x.view(-1, x.shape[-1])
        x = ops.gather(
            x,
            indices,
            bin_ids,
            bins,
            self.top_k)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()
            experts_per_rank = mpu.experts_per_rank(self.args)

            # Reshape to [world_size, num_experts_per_rank].
            world_size = mpu.get_expert_parallel_world_size(self.args)
            repeated_tokens_per_expert = (
                repeated_tokens_per_expert.view(world_size, experts_per_rank))
            parallel_tokens_per_expert = (
                parallel_tokens_per_expert.view(world_size, experts_per_rank))

            # TODO(tgale): It might be faster to do this on the GPU and
            # then communicate the results back to the host.
            send_counts = repeated_tokens_per_expert.cpu().sum(dim=-1)
            parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
            recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)

            # Convert the send/recv counts to lists.
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            tokens_received = sum(recv_counts)

        # If we're sharding the experts along the hidden dimension
        # multiple devices own parts of the same sets of experts.
        # Replicate the token counts so devices that share experts
        # get all of the tokens assigned to them.
        #
        # TODO(tgale): Fuse this into the prior, local permutation.
        x = ops.repeat(x, (mpu.hidden_sharding_degree(self.args), 1))

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        parallel_x, parallel_x_handle = all_to_all(
            x, recv_counts, send_counts,
            self.args.expert_parallel_group,
            async_op=True)

        with torch.no_grad():
            # After we do the cross-device permutation we have the tokens on the
            # correct device but not yet grouped by expert because we received
            # tokens from each device as contiguous chunks. To group the tokens
            # for expert computation we'll do one more local permutation. The
            # rest of this torch.no_grad() scope sets up the indices and bins
            # for this permutation.
            replicate_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert.flatten(), 0)
            replicate_bins = (
                replicate_bins.view(1)
                if not len(replicate_bins.size())
                else replicate_bins
            )

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(
                    self.num_experts * mpu.hidden_sharding_degree(self.args),
                    dtype=torch.int32,
                    device=indices.device
                ),
                mpu.experts_per_rank(self.args),
            )
            parallel_top_expert = ops.replicate(
                parallel_top_expert.unsqueeze(dim=0),
                replicate_bins, tokens_received).flatten()

            # TODO(tgale): The sort_end_bit here can be reduced.
            parallel_bin_ids, parallel_indices = ops.sort(
                parallel_top_expert, self.sort_end_bit)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(
                dim=0, dtype=torch.int)
            parallel_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert, 0)
            parallel_bins = (
                parallel_bins.view(1)
                if not len(parallel_bins.size())
                else parallel_bins
            )

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, hs = x.size()
            expert_capacity = self.expert_capacity(tokens)
            if expert_capacity == 0:
                expert_capacity = torch.max(
                    parallel_tokens_per_expert).item()

        # Locally permute the tokens and perform the expert computation.
        # Block to make sure that the cross-device permutation is complete.
        if self.args.mlp_impl == 'grouped':
            # GroupedMLP requires counts on CPU. We can use the tensor already
            # moved to CPU for the prior all_to_all, which avoids an extra
            # device synchronization.
            parallel_tokens_per_expert = parallel_tokens_per_expert_cpu.sum(
                dim=0, dtype=torch.int)
        parallel_x_handle.wait()
        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_tokens_per_expert,
            parallel_indices,
            parallel_bin_ids,
            None,  # expert_weights
            parallel_bins,
            expert_capacity,
            top_k=1)

        # Un-permute the tokens across the devices.
        x, _ = all_to_all(
            parallel_x, send_counts, recv_counts,
            self.args.expert_parallel_group)

        # Reduce along the hidden sharding to get the final outputs.
        #
        # TODO(tgale): Fuse this into the following local permutation.
        shape = (
            mpu.hidden_sharding_degree(self.args),
            -1,
            self.args.hidden_size
        )
        x = ops.sum(x.view(shape), dim=0)

        # Un-permute locally to setup for the next series of operations.
        x = ops.scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            self.top_k)
        return x, tokens_per_expert.flatten()

    def forward(self, x, scores, logits, expert_weights, top_experts):
        in_shape = x.size()

        # Compute the experts.
        x, tokens_per_expert = self.forward_fn(
            x, expert_weights, top_experts)
        if self.training and self.args.moe_loss_weight > 0:
            save_load_balancing_loss((tokens_per_expert, scores, logits))
        x = x.view(in_shape)
        if self.bias is not None:
            if self.args.return_bias:
                return x, self.bias
            return x + self.bias
        return x

    def forward_ec_grouped(self, x, scores, expert_weights, top_experts):
        bs, sl, hs = x.shape
        num_experts, k = expert_weights.shape

        x = x.flatten(start_dim=0, end_dim=1)
        x = torch.index_select(x, dim=0, index=top_experts.flatten())
        x = x.reshape((num_experts, k, hs))
        x = self.mlp(x)
        x = torch.einsum("ekd,ek->ekd", x, expert_weights)
        x = x.flatten(start_dim=0, end_dim=1)
        z = torch.zeros((bs * sl, hs)).type(x.type()).to(x.device)
        z.index_add_(dim=0, index=top_experts.flatten().to(int), source=x)
        x = z.reshape((bs, sl, hs))
        return x

    def forward_ec(self, x, scores, expert_weights, top_experts):
        """
        Expert choice forward func
        sl = sequence length
        bs = batch size
        hs = hidden size
        k = expert capacity
        Refs:
        - https://arxiv.org/pdf/2202.09368
        - https://github.com/google/flaxformer/blob/main/flaxformer/architectures/moe/routing.py#L647-L717
        - https://github.com/google/flaxformer/blob/399ea3a85e9807ada653fd0de1a9de627eb0acde/flaxformer/architectures/moe/moe_layers.py#L361
        - https://github.com/microsoft/DeepSpeed/issues/2517
        """
        bs, sl, hs = x.shape
        _, num_experts, k = expert_weights.shape
        # [bs, num_experts, k, sl]
        expert_gather_indices = torch.nn.functional.one_hot(top_experts, num_classes=sl).to(x.dtype)
        # [bs, sl, num_experts, k]
        expert_gather_indices = torch.moveaxis(expert_gather_indices, 3, 1)
        x_in = torch.einsum('bs...,bsek->bek...', x, expert_gather_indices)
        x_in = x_in.permute(1, 0, 2, 3).reshape(num_experts, bs * k, hs)
        x_e = self.mlp(x_in) # [num_experts, bs*k, d]
        combine_array = torch.einsum('...ek,...sek->...sek', expert_weights, expert_gather_indices)
        x_e = x_e.reshape(num_experts, bs, k, hs).permute(1, 0, 2, 3)
        x_out = torch.einsum('bek...,bsek->bs...', x_e, combine_array)
        return x_out                

class MoE(torch.nn.Module):

    def __init__(self, args : Arguments):
        super(MoE, self).__init__()

        # Token router.
        self.router = router.LearnedRouter(args)

        # Expert computation helper.
        self.experts = self._init_experts_mlp(args)

        self.shared_expert = None
        if args.shared_expert:
            # SharedExpert computation helper.
            self.shared_expert = sharedexpert_registry.get(args)

    def _init_experts_mlp(self, args: Arguments):
        return ParallelMLP(args)

    def forward(self, x, expert_mask: Optional[torch.Tensor] = None):
        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth.
        x = common.cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments.
        # swj: routing
        scores, logits, expert_weights, top_experts = self.router(x, expert_mask=expert_mask)

        # Compute the experts.
        out = self.experts(x, scores, logits, expert_weights, top_experts)
        if self.shared_expert is not None:
            shared_expert_out = self.shared_expert(x)
            out = self.shared_expert.add_experts_sharedexpert(shared_expert_out, out)
        return out