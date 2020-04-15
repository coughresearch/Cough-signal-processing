<p align="center">
  <img width="350" src="./Images/WAVE.png">
</p>
<h2 align="center">Cough Signal Processing ( csp ) </h2>



<p align="center">A micro framework for cough singal processing </p>


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

<img width="650" src="./Images/spectrogram_one.png">


```python
from csp import Spectrogram_features

# path of the cough audio
sp   = Spectrogram_features('cough_sound_9412.m4a')
data = sp.spectrogram_data()

```


### Audio augmentation techniques
#### Speed tuning

```python
from csp import Audio_augmentation

# Audio_augmentation speed tuning
Audio_aug = Audio_augmentation.speed_tuning(data['signal'])

```

#### output

<img width="650" src="./Images/spectrogram_two.png">


#### Time shifting
```python

# Audio augmentation time shifting
aug = Audio_augmentation.time_shifting(data['signal'])

```

#### output

<img width="650" src="./Images/spectrogram_three.png">
