<p align="center">
  <img width="250" src="./Images/WAVE.png">
</p>
<h2 align="center">Cough Signal Processing ( csp ) </h2>



<p align="center">A micro framework for cough singal processing </p>

[![GitHub license](https://img.shields.io/badge/License-Creative%20Commons%20Attribution%204.0%20International-blue)](https://github.com/coughresearch/Cough-signal-processing/blob/master/LICENSE)
[![GitHub commit](https://img.shields.io/github/last-commit/coughresearch/Cough-signal-processing)](https://github.com/coughresearch/Cough-signal-processing/commits/master)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

### Features

- Spectrogram features extraction
- Contiguous features
- Cough event detection
- Experiments on noise removal, Silence in cough sounds
- Applying different types of filters
- Audio augmentation techniques


### Quick Start

```python
from csp import Spectrogram_features

# path of the cough audio
sp   = Spectrogram_features('cough_sound_9412.m4a')
data = sp.spectrogram_data()

```

#### output

<img width="350" src="./Images/spectrogram_one.png">


### Audio augmentation techniques
#### Speed tuning

```python
from csp import Audio_augmentation

# Audio_augmentation speed tuning
Audio_aug = Audio_augmentation.speed_tuning(data['signal'])

```

#### output

<img width="350" src="./Images/spectrogram_two.png">


#### Time shifting
```python

# Audio augmentation time shifting
aug = Audio_augmentation.time_shifting(data['signal'])

```

#### output

<img width="350" src="./Images/spectrogram_three.png">
