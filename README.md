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