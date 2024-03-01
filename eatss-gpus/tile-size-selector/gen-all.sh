#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Need GPU architecture: a100 | xavier | 2080ti, warpfrac=1.0,0.5,etc"
  exit 1
fi

GPU=$1
WARPFRAC=$2
TAG=`echo $WARPFRAC | sed -e 's/\.//g'`

RAWFILE=${GPU}-wf${TAG}.raw
CAPLIST=`cat cap-config/$GPU.caps`

OUTFILE=${GPU}-wf${TAG}.tiles

if [ ! -f gpu-spec-files/$GPU.gpu ]; then
  echo "GPU descriptive file not found. Aborting ..."
  exit 1
fi

# clean files if already created
if [ -f "$RAWFILE" ]; then
  rm -rf $RAWFILE
fi
if [ -f "$OUTFILE" ]; then
  rm -rf $OUTFILE
fi

for bm in `find ./inputs/*.rels  -printf "%f\n"`; do
  if [ $bm == "doitgen.rels" ]; then
    continue
  fi
  if [ $bm == "fdtd-apml.rels" ]; then
    continue
  fi
  if [ $bm == "conv-2d.rels" ]; then
    continue
  fi
  if [ $bm == "mttkrp.rels" ]; then
    continue
  fi
  if [ $bm == "heat-3d.rels" ]; then
    continue
  fi
  bm_str=`echo $bm | sed 's/.rels//g'`
  echo -e "generating tile size for $bm_str"
  for cap in $CAPLIST; do
    python gpu-tss-energy.py ./inputs/$bm gpu-spec-files/${GPU}.gpu $cap $WARPFRAC >> $RAWFILE
  done
done 

echo "#TILE_CONF:benchmark:shmem-frac:wafrac:tile-sizes" > $OUTFILE
cat $RAWFILE | grep 'TILE_CONF' >> $OUTFILE
