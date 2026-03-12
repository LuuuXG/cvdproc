# Containers used in cvdproc

The containers used in cvdproc are listed below:

## Docker Images

- deepmi/lit:0.5.0
- kilianhett/chp_seg:1.0.1
- leonyichencai/synb0-disco:v3.1
- nipreps/fmriprep:v25.1.4
- pennlinc/aslprep:v25.0.0
- pennlinc/qsiprep:v1.0.1
- pennlinc/qsirecon:v1.0.0
- pennlinc/xcp_d:v0.11.0
- segcsvd_rc03:latest
- ytzero/synbold-disco:v1.4

## Docker Notes

Load custom docker images (.tar.gz)

```
# e.g. qsiprep_1.0.1-custom.tar.gz
gunzip -c qsiprep_1.0.1-custom.tar.gz | docker load
```

The modified QSIPrep image: [BaiduNetdisk Link](https://pan.baidu.com/s/1JH2R0IAoXlawIjWZ0VEgyw). Extraction code: 0721