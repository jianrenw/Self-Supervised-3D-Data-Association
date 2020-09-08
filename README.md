# Uncertainty-aware Self-supervised 3D Data Association

![](images/us3da.jpg)

This is the official repo of "Uncertainty-aware Self-supervised 3D Data Association", IROS 2020 <br/> \
Project page: <url>https://jianrenw.github.io/Self-Supervised-3D-Data-Association/</url> \
Paper: <url>https://arxiv.org/pdf/2008.08173.pdf</url>

### Data Preprocessing:
First generate sudo label for self-supervised embedding training:
<pre>
bash ./tracking/main.sh
python ./embedding/dataset/nuscenes_preprocessing.py
</pre>
### Self-supervised embedding training:
<pre>
bash ./embedding/main.sh
</pre>
### Combining appearance embedding with motion priors for MOT:
<pre>
python ./combination/generate_data.py
python ./combination/logistic_regression.py
bash ./tracking/main.sh
</pre>

## Acknowlegement
This project is not possible without multiple great opensourced codebases. We list some notable examples below.  

* [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)
* [pointnet](https://github.com/charlesq34/pointnet)

**SS3DA is deeply influenced by the following projects. Please consider citing the relevant papers.**

```
@article{Weng2020_AB3DMOT, 
    author = {Weng, Xinshuo and Wang, Jianren and Held, David and Kitani, Kris}, 
    journal = {IROS}, 
    title = {{3D Multi-Object Tracking: A Baseline and New Evaluation Metrics}}, 
    year = {2020} 
}

@inproceedings{jianren20s3da,
    Author = {Wang, Jianren and Ancha, Siddharth and Chen, Yi-Ting and Held, David},
    Title = {Self-supervised 3D Data Association},
    Booktitle = {IROS},
    Year = {2020}
}
```