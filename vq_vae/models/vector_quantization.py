# ---encoding:utf-8---
# @Time    : 2025/3/7 20:22
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : SingleCellExperiment
# @Software: PyCharm
import torch
from torch.autograd import Function

#  向量量化
class VectorQuantization(Function):

    @staticmethod
    def forward(ctx, inputs, codebook):
        # 停止梯度传播
        with torch.no_grad():
            embedding_size = codebook.size(1) # 码本中每个向量的维度
            inputs_size = inputs.size() # 输入的形状
            inputs_flatten = inputs.view(-1, embedding_size) # 将输入展平，形成一维向量

            codebook_sqr = torch.sum(codebook ** 2, dim=1) # 计算码本中每个向量的平方和
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True) # 计算输入向量的平方和
            # 计算每个输入向量与码本中所有码字的距离
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
            # 找到与输入向量距离最近的码字
            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1]) # 将结果恢复为原输入的形状
            ctx.mark_non_differentiable(indices) # 标记indices不需要计算梯度 mark_non_differentiable 将输出标记为不可微

            return indices

# STE 直通估计器
class VectorQuantizationStraightThrough(Function):

    @staticmethod
    def forward(ctx, inputs, codebook):
        #  向量标准化
        indices = vq(inputs, codebook) # 获取最近向量索引
        indices_flatten = indices.view(-1) # 将索引展平
        ctx.save_for_backward(indices_flatten, codebook) # 保存索引和码本用于反向传播
        ctx.mark_non_differentiable(indices_flatten) # 标记索引为不需要计算梯度
        # 根据选中的索引从码本中选择对应的向量
        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs) # 将选择的向量恢复为与输入相同的形状

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]: # 如果需要计算输入的梯度
            # 直通估计器：直接通过梯度
            grad_inputs = grad_output.clone() # 直接将输出的梯度传递给输入
        if ctx.needs_input_grad[1]: # 如果需要计算码本的梯度
            # 计算码本的梯度
            indices, codebook = ctx.saved_tensors # 获取保存的索引和码本
            embedding_size = codebook.size(1) # 获取码本中向量的维度
            grad_output_flatten = (grad_output.contiguous().view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)  # 初始化梯度
            # 使用选中的索引更新码本的梯度
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]
