# Multi-path enabled private audio with noise. <a href = 'https://arxiv.org/pdf/1811.07065.pdf'>[Paper]</a><a href = 'https://swing-research.github.io/private-audio/icassp19_poster.pdf'>[Poster]</a> <a href = 'https://swing-research.github.io/private-audio/'>[Project page]</a> 
<b>Authors:</b> Anadi Chaman, Yu-Jeh Liu, Jonah Casebeer, Ivan Dokmanić.

We present a private audio communication system between a set of centrally coordinated loudspeakers and microphones in a reverberant room.

Traditional multi-zone sound field reproduction systems focused on sending linearly filtered signals to boost SNR in target zones, while silencing the sound elsewhere. However, silencing sound everywhere else is hard and requires a large number of speakers.

We instead use noise to reduce SNR everywhere outside the target locations. In particular, we emit structured noise signals from loudspeakers which after echoing across the room yield intelligible audio only at the target locations. An eavesdropper at anywhere else hears nothing but noise!

The iPython notebook, main.ipynb, in this repository contains an example that illustrates two methods that we proposed to construct the noise signals. main.ipynb saves its simulation results in folders mccs_reconstructions and null_space_method_reconstructions.

# Dependencies
This code uses packages—<a href = 'https://github.com/LCAV/pyroomacoustics'>Pyroomacoustics</a> and <a href = 'https://github.com/mpariente/pystoi'>pystoi</a>—for simulating room impulse responses and computing sound intelligibility scores respectively. These can be installed using pip as follows.

```
pip install pyroomacoustics
pip install pystoi
```

