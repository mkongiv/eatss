#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Need GPU architecture: a100 | xavier | 2080ti"
  exit 1
fi

GPU=$1
OUTFILE=${GPU}-heavy.tiles
RAWFILE=${GPU}-heavy.raw

if [ ! -f gpu-spec-files/$GPU.gpu ]; then
  echo "GPU descriptive file not found. Aborting ..."
  exit 1
fi

CAPLIST=`cat cap-config/$GPU.caps`

# clean files if already created
if [ -f "$RAWFILE" ]; then
  rm -rf $RAWFILE
fi
if [ -f "$OUTFILE" ]; then
  rm -rf $OUTFILE
fi

#bmlist="heat-3d.rels conv-2d.rels"
#bmlist="mttkrp.rels"
bmlist="heat-3d.rels conv-2d.rels mttkrp.rels"

#for pair in 'doitgen.rels:0.25' 'fdtd-apml.rels:0.25'; do
for bm in $bmlist; do
  bm_str=`echo $bm | sed 's/.rels//g'`
  echo -e "generating tile size for $bm_str"
  d0=`date`
  for cap in $CAPLIST; do
    for wfrac in 1.0 0.5 0.25 0.125 0.0; do
      python gpu-tss-energy.py ./inputs/$bm gpu-spec-files/${GPU}.gpu $cap $wfrac >> $RAWFILE
    done
  done
  d1=`date`
  echo "$bm##$d0##$d1" >> heavy.timestamps.txt
done 

cat $RAWFILE | grep 'TILE_CONF' > $OUTFILE
