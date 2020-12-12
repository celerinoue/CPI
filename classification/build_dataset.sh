#!/bin/sh
cd `dirname $0`

kgcn-chem --assay_dir ChEMBL_sample/gpcr -a 50 --assay_num_limit 100 --output multitask.jbl
kgcn-chem --assay_dir ChEMBL_sample/gpcr -a 50 --assay_num_limit 100 --output multimodal.jbl --multimodal

