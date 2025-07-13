# Hi-Cformer

**Hi-Cformer** is a transformer-based model designed to analyze single-cell Hi-C (scHi-C) data, addressing the inherent challenges of sparsity and uneven contact distribution. Hi-Cformer targets the **complex, multi-scale, and local patterns** in scHi-C contact maps by modeling chromatin interactions across diverse genomic distances.

Built upon a multi-scale attention framework, Hi-Cformer simultaneously captures both broad and fine-grained chromatin interaction features. It delivers **low-dimensional representations** of single cells that are highly informative for tasks such as clustering, cell type annotation, and imputation of 3D genomic signals. Its robust design supports generalization across different datasets and resolutions, offering a versatile tool for 3D genome analysis at single-cell resolution.

<p align="center">
  <img src="https://raw.githubusercontent.com/Xiaoqing-Wu02/Hi-Cformer/main/assets/hicformer.png" width="400" height="1000" alt="Hi-Cformer Model Architecture">
</p>


---

## 🚀 Highlights

- **Multi-scale modeling of scHi-C maps**: Captures interaction blocks across genomic distances with specialized attention modules.
- **Representation learning**: Derives low-dimensional embeddings that reflect chromatin structure heterogeneity.
- **Accurate imputation of 3D genomic features**: Recovers interaction signals like TAD-like boundaries and A/B compartments from sparse data.
- **Generalizable cell type annotation**: Embeddings can be used for robust classification across datasets.


---

## 📖 Citation

If you use **Hi-Cformer** in your research, please cite:

> Xiaoqing Wu1,†, Xiaoyang Chen1,†, Rui Jiang1,*. *Hi-Cformer enables multi-scale chromatin contact map modeling for single-cell Hi-C data analysis*.

---

## 📬 Contact

For questions or collaborations, please contact:

**Email**: xq-wu24@mails.tsinghua.edu.cn
**GitHub**: [@Xiaoqing-Wu02](https://github.com/Xiaoqing-Wu02)
