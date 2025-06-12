# ---encoding:utf-8---
# @Time    : 2025/6/4 15:12
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce-plus
# @Software: PyCharm
import os

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from gseapy.parser import read_gmt

def create_pathway_file(dir_paths, output_gmt_path, groupby_key='celltype', annotation_key="celltype", top_n_genes=30):
    all_pathways = {}

    for data_dir in tqdm(dir_paths, desc="Processing directories"):
        h5ad_files = [f for f in os.listdir(data_dir) if f.endswith(".h5ad")]
        h5ad_file = h5ad_files[0]
        h5ad_path = os.path.join(data_dir, h5ad_file)
        sample_name = os.path.splitext(h5ad_file)[0]

        annot_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        annot_file = annot_files[0]
        annot_path = os.path.join(data_dir, annot_file)

        # 读取数据
        adata = sc.read(h5ad_path)
        adata.var_names_make_unique()
        annot = pd.read_csv(annot_path, index_col=0)

        # 细胞ID匹配
        common_cells = adata.obs_names.intersection(annot.index)

        adata = adata[common_cells]
        annot = annot.loc[common_cells]
        # 添加注释列
        adata.obs[groupby_key] = annot[annotation_key]

        sc.pp.normalize_total(adata, target_sum=1e4) # 归一化
        sc.pp.log1p(adata)  # 对数转换

        # 对指定 groupby_key 使用 Wilcoxon 排序检验计算每个群体相对于其他群体的差异表达基因
        sc.tl.rank_genes_groups(adata, groupby=groupby_key, method="wilcoxon")
        # 取得差异表达的结果数据
        result = adata.uns["rank_genes_groups"]
        # 获取差异表达基因针对的每个细胞群体标签
        groups = result["names"].dtype.names

        for group in tqdm(groups, desc=f"Extracting {sample_name} markers", leave=False):
            gene_names = result["names"][group][:top_n_genes].tolist()
            module_name = f"{sample_name}_{groupby_key}_{group}"
            all_pathways[module_name] = gene_names

    with open(os.path.join(output_gmt_path, 'my_pathways.gmt'), "w") as f:
        for pathway, genes in all_pathways.items():
            line = "\t".join([pathway, "-"] + genes)
            f.write(line + "\n")

    print(f"通路文件已保存至{output_gmt_path}")

def pathway_guided(h5ad_paths, gmt_file, output_prefix):
    adatas = [sc.read(path) for path in h5ad_paths]
    # 提取公共基因集
    common_genes = set(adatas[0].var_names)
    for adata in adatas[1:]:
        common_genes &= set(adata.var_names)
    common_genes = list(common_genes)

    # 从自定义 gmt 文件读取 pathway 基因集 回 dict { pathway_name: [gene1, gene2, ...] }
    gene_sets = read_gmt(gmt_file)

    # 构建 pathway-guided gene index
    ordered_genes = []
    used_genes = set()

    for pathway_name, genes in gene_sets.items():
        pathway_genes = [g for g in genes if g in common_genes and g not in used_genes]
        ordered_genes.extend(pathway_genes)
        used_genes.update(pathway_genes)

    # 加上剩余未出现的基因
    remaining_genes = [g for g in common_genes if g not in used_genes]
    ordered_genes.extend(remaining_genes)

    # 按顺序重排每个 h5ad 文件的基因并保存
    for i, adata in enumerate(adatas):
        existing_genes = adata.var_names.tolist()
        missing_genes = [g for g in ordered_genes if g not in existing_genes]

        # 创建缺失基因全 0 矩阵
        zero_expr = np.zeros((adata.n_obs, len(missing_genes)))
        df_zero = pd.DataFrame(zero_expr, index=adata.obs_names, columns=missing_genes)

        # 构造现有表达数据的 DataFrame
        df_expr = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                               index=adata.obs_names,
                               columns=adata.var_names)

        # 合并原始 + 补零表达矩阵
        df_full = pd.concat([df_expr, df_zero], axis=1)
        df_full = df_full[ordered_genes]  # 重新排序

        # 创建新的 AnnData 对象
        new_adata = sc.AnnData(X=df_full.values)
        new_adata.obs = adata.obs.copy()
        new_adata.var = pd.DataFrame(index=ordered_genes)

        out_path = f"{output_prefix}{i+1}.h5ad"
        new_adata.write(out_path)
        print(f"Saved {out_path} with shape {new_adata.shape}")


if __name__ == '__main__':
    pathway_guided(
        [
            'E:\\workplace\\SingleCellSequencing\\bulk_scdata\\blood\\sc_data\\sc_data.h5ad',
        ],
        './pathway/h.all.v2024.1.Hs.symbols.gmt',
        './processed/pathway_guided'
    )

