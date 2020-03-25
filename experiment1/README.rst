Experiment 1
============

.. _Bioplex: http://ndexbio.org/#/network/98ba6a19-586e-11e7-8f50-0ac135e8bacf

This directory contains the results of the experiment with
99,000 networks from `Bioplex`_

Steps to reproduce
-------------------

In addition to having required software mentioned in this repo
this **Docker** must also be installed and working

.. note::

   ``*.csv`` files in this directory are gzip compressed and have ``.gz`` suffix. Be sure to uncompress before using

#. Generate training data

   This step creates subnetworks as edge files (``*.tsv``) which
   can be done by running this script from base of repo
   (which assumes ``bioplex.cx`` is the CX file of `Bioplex`_ network)

   .. code-block::

      ./run.sh > run.out

   The above command will create numbered directories with ``*.tsv`` files
   containing edge lists of networks.

#. Run Infomap on training data

   Once the above step is complete run this:

   .. warning::

      Running Infomap requires **Docker** to be installed and correctly configured

   .. code-block::

      ./run_infomap.sh

   The above command will create a bunch of ``*.infomap.csresult`` files

#. Gather stats into single CSV file

   On a terminal run a for loop like this (can be put in a shell script):

   .. code-block::

      #!/bin/bash

      for Y in `find . -name "*.tsv" -type f` ; do
          outfile="`pwd`/${Y}.infomap.cdresult"
          if [ ! -f $outfile ] ; then
              continue
          fi
          ./get_hierarchystats.py $Y $outfile >> rawinfomapstats.csv
      done


#. Extract a training dataset

   Create ``test.csv, train.csv, valid.csv`` files
   under ``experiment1`` directory by running this command:

   .. code-block::

      ./extract_trainingdata.py rawinfomapstats.csv experiment1/


   .. image:: train_csv_histograms.png

   .. image:: train_csv_plots.png

#. Train

   Train the model which will be saved to path specified
   by ``--save``

   .. code-block::

      cd experiment1
      ../predict_clusters.py --mode train \
         --save bioplexpredictclusters.100epoch.pt \
         --traindata train.csv --validationdata valid.csv \
         --num_epochs 100

   .. note::

      Added --plotgraphs will cause script display matplotlib of
      learning at completion which will cause script to hang until
      plot is closed

   .. image:: training.png

#. Predict

   Run prediction using trained model specified by
   ``--model`` flag

  .. code-block::

     for Y in `cat test.csv` ; do
         ../predict_clusters.py --mode predict \
            --model bioplexpredictclusters.100epoch.pt \
            --predict $Y >> ppredict.jsons



  Should output something like this:

  .. code-block::

     {"numberNodes": 1290.0,
      "numberEdges": 1884.0,
      "density": 0.0022660436249481303,
      "degreeMean": 2.92093023255814,
      "degreeStddev": 2.9468068230708377,
      "predictedNumberOfClusters": 342,
      "actualNumberOfClusters": 349,
      "inputTSVFile": "2200.708.tsv.infomap.cdresult"}

  .. note::

     The ``--predict`` flag detects if the data has the extra columns
     and automatically outputs those in the resulting json


  To check the accuracy with ``test.csv`` I ran

  .. code-block::

     ./analyze_predict.py ppredict.jsons

  And got this:

  .. code-block::

     Percent exact matches: 3.0%
     Percent within 1% : 17.4%
     Percent within 5% : 71.7%
     Percent within 10% : 96.1%


