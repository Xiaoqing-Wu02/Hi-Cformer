# Hi-Cformer

**Hi-Cformer** is a transformer-based model designed to analyze single-cell Hi-C (scHi-C) data, addressing the inherent challenges of sparsity and uneven contact distribution. Hi-Cformer targets the **complex, multi-scale, and local patterns** in scHi-C contact maps by modeling chromatin interactions across diverse genomic distances.

Built upon a multi-scale attention framework, Hi-Cformer simultaneously captures both broad and fine-grained chromatin interaction features. It delivers **low-dimensional representations** of single cells that are highly informative for tasks such as clustering, cell type annotation, and imputation of 3D genomic signals. Its robust design supports generalization across different datasets and resolutions, offering a versatile tool for 3D genome analysis at single-cell resolution.

<p align="center">
  <img src="https://raw.githubusercontent.com/Xiaoqing-Wu02/Hi-Cformer/main/assets/hicformer.png" width="600" height="1200" alt="Hi-Cformer Model Architecture">
</p>


---

## 🚀 Highlights

- **Multi-scale modeling of scHi-C maps**: Captures interaction blocks across genomic distances with specialized attention modules.
- **Representation learning**: Derives low-dimensional embeddings that reflect chromatin structure heterogeneity.
- **Accurate imputation of 3D genomic features**: Recovers interaction signals like TAD-like boundaries and A/B compartments from sparse data.
- **Generalizable cell type annotation**: Embeddings can be used for robust classification across datasets.


---

## Tutorial

This repository includes a tutorial example to get you started quickly with Hi-Cformer.

- **Demo Notebook:**  
  The file [`demo/Embedding with Hi-Cformer.ipynb`](demo/Embedding%20with%20Hi-Cformer.ipynb) contains a step-by-step example.  
  It explains the input file format, runs a training process, and performs visualization of the results.

- **Quick Start with Training Script:**  
  You can also use the executable Python script `hicformer/train_validation.py` to compute clustering metrics and quickly test the model.  
  **Note:** You need to modify the input file path at the beginning of the script before running.

  The script supports several command-line arguments. Example usage:

  ```bash
  python hicformer/train_validation.py --cuda 1 --mask_ratio 0.4 --num 0 --lr 0.0005 --atlr 0.0005 --chr_single_loss True --chr_single_ratio 0.5 --loss_ratio 0.02 --weight 0.1 --epochs 180
---

## 📖 Citation

If you use **Hi-Cformer** in your research, please cite:

> Xiaoqing Wu1,†, Xiaoyang Chen1,†, Rui Jiang1,*. *Hi-Cformer enables multi-scale chromatin contact map modeling for single-cell Hi-C data analysis*.

---

## 📬 Contact

For questions or collaborations, please contact:

**Email**: xq-wu24@mails.tsinghua.edu.cn
**GitHub**: [@Xiaoqing-Wu02](https://github.com/Xiaoqing-Wu02)
