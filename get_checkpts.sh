#!/bin/bash
#pretrained.pt
gdown --id '1Y8IO2_OqeT85P1kks9I9eeAq--S65YFb' -O "./checkpts/spk_encoder/"
# generator
gdown --id '10khlrM645pTbQ4rc2aNEYPba8RFDBkW-' -O "./checkpts/vocoder/"
#vc_libritts_wodyn.pt
gdown --id '18Xbme0CTVo58p2vOHoTQm8PBGW7oEjAy' -O "./checkpts/vc/"
#vc_vctk_wodyn.pt
gdown --id '12s9RPmwp9suleMkBCVetD8pub7wsDAy4' -O "./checkpts/vc/"