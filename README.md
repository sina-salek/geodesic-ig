# Geodesic Integrated Gradients
<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="images/half_moons_y.png" width="400"/>
        <br>
        <em>IG (left) vs Geodesic IG (right)</em>
      </td>
      <td align="center">
        <img src="images/voc_compare.png" width="540"/>
        <br>
        <em>A comparison of IG vs Geodesic IC for image classification</em>
      </td>
    </tr>
  </table>
</p>

## Installation

### Using pip
```bash
pip install geodesic-ig
```
### From source
```bash
git clone https://github.com/yourusername/geodesic-ig.git
cd geodesic-ig
pip install .
```
### Quick Start

```python
from geodesic.attr import GeodesicIGSVI, GeodesicIGKNN
knn_explainer = GeodesicIGKNN(net)
knn_attributions = explainer.attribute(
                    x_test,
                    baselines=baselines,
                    target=target,
                    n_neighbors=n,
                    internal_batch_size=200,
                    )

svi_expliner = GeodesicIGSVI(net)
svi_attributions = explainer.attribute(
                    x_test,
                    baselines=baselines,
                    target=terget,
                    num_iterations=num_iterations,
                    beta=beta,
                    n_steps=n_steps,
                    do_linear_interp=li,
                    use_endpoints_matching=em, learning_rate=learning_rat,
                    )
```

For more detailed examples see:
- [VOC Image Classification Experiment](./experiments/voc/)
- [Half Moons Experiment](./experiments/moons/)