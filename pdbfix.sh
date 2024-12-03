for file in /home/pbuser/Desktop/Jessi/data_prep/PDB/*.ent; do

    pdb=${file: -8:4}
    echo $pdb
    python3 /home/pbuser/Desktop/Jessi/data_prep/fix_pdb.py "$pdb" #pdbfixer
    reduce -BUILD -NUC -NOFLIP -Quiet /home/pbuser/Desktop/Jessi/data_prep/PDB/$pdb-fixed.pdb > /home/pbuser/Desktop/Jessi/data_prep/PDB/$pdb-red.pdb
    pdb2pqr --ff AMBER --with-ph 7.0 /home/pbuser/Desktop/Jessi/data_prep/PDB/$pdb-red.pdb /home/pbuser/Desktop/Jessi/data_prep/PDB/$pdb.pqr 

done




