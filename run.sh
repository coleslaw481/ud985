#!/bin/bash

for Y in `seq 200 100 50000` ; do 
   echo $Y
   /usr/bin/time -p ./generate_subgraph.py -v --numnodes $Y --numnetworks 1000 bioplex.cx $Y
done
