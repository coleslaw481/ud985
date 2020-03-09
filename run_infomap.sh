#!/bin/bash

for Y in `find . -name "*.tsv" -type f` ; do
    outfile="`pwd`/${Y}.infomap.cdresult"
    if [ -f $outfile ] ; then
       if [ -s $outfile ] ; then
          continue
       fi
       echo "file $outfile is empty. Rerunning: `ls -la $outfile`"
    fi
    echo "Running against $Y"
    echo "docker run --rm -v `pwd`:`pwd`:z coleslawndex/cdinfomap:0.1.0 `pwd`/$Y > $outfile"
    docker run --rm -v `pwd`:`pwd`:z coleslawndex/cdinfomap:0.1.0 `pwd`/$Y > $outfile
done
