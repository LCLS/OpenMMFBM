#!/bin/bash

module load python
module load tessa/devel_rjn

for i in *.dcd;
do
    name=`basename $i .dcd`;
    catdcd -o $name.trr -otype trr -s wwfip35_folded.pdb -dcd $i
    trjconv -f $name.trr -o $name.xtc -s wwfip35_folded.pdb -n system.ndx
done

for i in *.xtc;
do
    name=`basename $i .xtc`
    tessa -P $name.png -O $name.rmsd -T $i -x $i -n ww.ndx -m gromacs rmsd -r ww.pdb -d 1000 -Y 30
done
